# transformer_risk_model.py + integrated buffer-based training
import torch
import torch.nn as nn
import numpy as np
import random
import math
from algos.risk_training import RiskTrainingMixin
# ------------------------------
# 1. Experience Buffer
# ------------------------------
# transformer_risk_model.py

import collections # 导入collections

# 定义序列长度
SEQUENCE_LENGTH = 10 # 比如，我们关心过去10个时间步的序列

class RiskExperienceBuffer:
    def __init__(self, max_size=5000):
        self.to_use_buffer = {}

        self.active_buffer = collections.deque(maxlen=max_size)
        self.risk_active_buffer = collections.deque(maxlen=max_size)

    def add(self, vehicle_id, state, action, context=None, step=None):
        # Retain delayed terminal supervision, but never concatenate a new
        # control segment, coordinate system or nonconsecutive simulation step.
        previous = self.to_use_buffer.get(vehicle_id)
        changed = previous is not None and (
            previous.get('context') != context or
            (step is not None and previous.get('step') is not None
             and step != previous['step'] + 1))
        if previous is None or changed:
            self.to_use_buffer[vehicle_id] = {
                "state_seq": collections.deque(maxlen=SEQUENCE_LENGTH),
                "action_seq": collections.deque(maxlen=SEQUENCE_LENGTH),
                "context": context,
            }
        history = self.to_use_buffer[vehicle_id]
        history['step'] = step
        history["state_seq"].append(np.array(state, dtype=np.float32, copy=True))
        history["action_seq"].append(np.array(action, dtype=np.float32, copy=True))

    def clear_pending(self):
        """Discard unfinished histories when SUMO starts a new episode."""
        self.to_use_buffer.clear()

    @staticmethod
    def pad_sequence(state_seq, action_seq):
        valid_count = len(state_seq)
        if not 1 <= valid_count <= SEQUENCE_LENGTH or len(action_seq) != valid_count:
            raise ValueError('state/action histories must have the same length in [1,10]')
        valid_mask = np.zeros(SEQUENCE_LENGTH, dtype=np.bool_)
        valid_mask[-valid_count:] = True
        state_dim = len(state_seq[0])
        action_dim = len(action_seq[0])
        missing = SEQUENCE_LENGTH - valid_count
        states = [np.zeros(state_dim, dtype=np.float32) for _ in range(missing)] + state_seq
        actions = [np.zeros(action_dim, dtype=np.float32) for _ in range(missing)] + action_seq
        return (np.asarray(states, dtype=np.float32),
                np.asarray(actions, dtype=np.float32), valid_mask)

    def update_data(self, vehicle_id, label, junction_id=None):
        if vehicle_id in self.to_use_buffer:
            sample_queues = self.to_use_buffer.pop(vehicle_id)
            context = sample_queues.get('context')
            if junction_id is not None and context is not None and context[1] != junction_id:
                # A collision at another junction is not an outcome for this window.
                return
            state_seq = list(sample_queues["state_seq"])
            action_seq = list(sample_queues["action_seq"])

            if not state_seq:
                return

            original_state_seq, original_action_seq, valid_mask = self.pad_sequence(state_seq, action_seq)

            original_data = {
                "state_seq": original_state_seq,
                "action_seq": original_action_seq,
                "valid_mask": valid_mask,
                "label": label,
                "context": context,
                "last_step": sample_queues.get('step'),
            }

            if label == 1:
                self.risk_active_buffer.append(original_data)
            else:
                self.active_buffer.append(original_data)

    def sample(self, batch_size):
        if batch_size < 2 or batch_size % 2:
            raise ValueError('balanced sampling requires a positive even batch size >= 2')
        # 确保有足够的数据可供采样
        if len(self.active_buffer) < batch_size // 2 or len(self.risk_active_buffer) < batch_size // 2:
            return None, None, None, None

        # 平衡采样
        risk_batch = random.sample(self.risk_active_buffer, batch_size // 2)
        active_batch = random.sample(self.active_buffer, batch_size // 2)
        batch = risk_batch + active_batch
        random.shuffle(batch) # 打乱批次顺序

        # 收集数据
        # 使用np.stack可以直接将样本列表转换为一个批次
        state_batch = np.stack([item['state_seq'] for item in batch])
        action_batch = np.stack([item['action_seq'] for item in batch])
        mask_batch = np.stack([item['valid_mask'] for item in batch])
        label_batch = np.array([item['label'] for item in batch], dtype=np.float32)

        return state_batch, action_batch, mask_batch, label_batch

    def __len__(self):
        return len(self.active_buffer) + len(self.risk_active_buffer)

    def get_sequence(self, vehicle_id):
        """为奖励计算提供获取当前序列的接口"""
        if vehicle_id not in self.to_use_buffer:
            return None, None, None

        state_seq = list(self.to_use_buffer[vehicle_id]["state_seq"])
        action_seq = list(self.to_use_buffer[vehicle_id]["action_seq"])

        if not state_seq:
            return None, None, None

        states, actions, valid_mask = self.pad_sequence(state_seq, action_seq)
        return states[None, ...], actions[None, ...], valid_mask[None, ...]


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)

        # 将pe注册为模型的buffer，它不是模型参数，但会随着模型移动(如.to(device))
        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Tensor, shape [batch_size, seq_len, d_model]
               (注意: PyTorch Transformer默认是 [seq_len, batch_size, d_model]，
                但我们通常使用batch_first=True，所以这里适配更常见的形状)
        """
        # x的形状是[N, S, E] (Batch, Seq, Embedding)，而pe是[max_len, 1, E]
        # 我们需要将pe调整为 [1, S, E] 以便广播相加
        x = x + self.pe.squeeze(1)[:x.size(1), :].unsqueeze(0)
        return self.dropout(x)


# ------------------------------
# 2. Transformer Risk Model
# ------------------------------
class TransformerRiskPredictor(RiskTrainingMixin, nn.Module):
    def __init__(self, state_dim, action_dim, d_model=128, nhead=4, num_layers=2, buffer_size=10000):
        super().__init__()
        self.input_dim = state_dim + action_dim
        self.embedding = nn.Linear(self.input_dim, d_model)

        self.pos_encoder = PositionalEncoding(d_model)

        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)


        self.classifier = nn.Sequential(
            nn.Linear(d_model, 128),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )

        self.buffer = RiskExperienceBuffer(max_size=buffer_size)
        self.optimizer = torch.optim.AdamW(self.parameters(), lr=1e-5, weight_decay=1e-5)
        self.last_train_updates = 0

        # --- 新增代码：创建并注册注意力偏置 ---
        # register_buffer将张量注册为模型的一部分，它会被.to(device)移动，但不会被视为可训练参数
        self.register_buffer("attention_bias", self._create_attention_bias(SEQUENCE_LENGTH))

    def _create_attention_bias(self, seq_len, decay_factor=0.2):
        """
        创建一个注意力偏置矩阵，用于引导模型更关注序列的后半部分。
        矩阵形状为 (seq_len, seq_len)。
        bias[i, j] 的值会加到“查询(query)位置i”对“键(key)位置j”的注意力分数上。
        我们希望j越大（越靠后），偏置值也越大（惩罚越小）。
        """
        # 创建一个基础向量，值为 [-(S-1), -(S-2), ..., -1, 0]
        base_bias = torch.arange(seq_len, dtype=torch.float) - (seq_len - 1)

        # 乘以一个衰减因子来控制偏置的强度
        base_bias = base_bias * decay_factor

        # 将这个向量广播成 (seq_len, seq_len) 的矩阵
        # 每一行都是相同的，这意味着每个位置在评估其他位置时，都使用相同的“时间偏好”
        bias_matrix = base_bias.unsqueeze(0).repeat(seq_len, 1)

        # print(f"创建的注意力偏置 (Attention Bias) 示例 (第一行): {bias_matrix[0].numpy()}")
        return bias_matrix

    def forward(self, state_seq, action_seq, valid_mask=None):
        # 输入已经是 [batch_size, seq_len, feature_dim]
        # 连接状态和动作
        x = torch.cat([state_seq, action_seq], dim=-1) # [batch_size, seq_len, input_dim]

        # 嵌入层
        x = self.embedding(x)  # [batch_size, seq_len, d_model] (N, S, E)

        # --- 新增代码：应用位置编码 ---
        x = self.pos_encoder(x)

        x = x.permute(1, 0, 2)  # (N, S, E) 换为 (S, N, E)

        # Transformer编码器
        padding_mask = None if valid_mask is None else ~valid_mask.bool()
        x = self.transformer(x, mask=self.attention_bias,
                             src_key_padding_mask=padding_mask)

        x = x.permute(1, 0, 2) # (N, S, E)

        # 取序列中最后一个时间步的输出来进行分类
        x_last = x[:, -1, :]  # [batch_size, d_model]

        # 分类器
        out = self.classifier(x_last)  # [batch_size, 1]

        return out.squeeze(-1)  # [batch_size]

    def save_model(self, save_path):
        torch.save(self.state_dict(), save_path)
        print(f"Model saved to {save_path}")

    def load_model(self, save_path, map_location=None):
        self.load_state_dict(torch.load(save_path, map_location=map_location))
        print(f"Model loaded from {save_path}")
