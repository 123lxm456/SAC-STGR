"""Shared legacy BCE training schedule for B0 and the online P3 adapter."""

import torch
from torch import nn


class RiskTrainingMixin:
    def train_model(self, batch_size=128, epochs=3, lr=1e-5, device='cpu', save_path='risk_model.pt'):
        self.last_train_updates = 0
        for group in self.optimizer.param_groups:
            group['lr'] = lr
        criterion = nn.BCELoss()

        total_loss = 0.0
        num_successful_batches = 0

        for epoch in range(epochs):
            self.train()
            epoch_loss = 0.0
            epoch_batches = 0
            num_batches = max(1, len(self.buffer) // batch_size)

            for _ in range(num_batches):
                state_batch, action_batch, mask_batch, label_batch = self.buffer.sample(batch_size)
                if state_batch is None:
                    continue

                # 转换为tensor并移动到设备
                s = torch.tensor(state_batch, dtype=torch.float32).to(device)
                a = torch.tensor(action_batch, dtype=torch.float32).to(device)
                mask = torch.tensor(mask_batch, dtype=torch.bool).to(device)
                y = torch.tensor(label_batch, dtype=torch.float32).to(device)

                # 前向传播
                preds = self.forward(s, a, mask)
                loss = criterion(preds, y)

                # 反向传播
                self.optimizer.zero_grad()
                loss.backward()

                # 梯度裁剪，防止梯度爆炸
                torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=1.0)

                self.optimizer.step()
                self.last_train_updates += 1

                batch_loss = loss.item() * s.size(0)
                epoch_loss += batch_loss
                epoch_batches += 1

                # 清理内存
                del s, a, mask, y, preds, loss
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()


            # 计算每个epoch的平均损失
            if epoch_batches > 0:
                avg_epoch_loss = epoch_loss / (epoch_batches * batch_size)
                total_loss += avg_epoch_loss
                num_successful_batches += 1
                # print(f"Epoch {epoch + 1}/{epochs}, Loss: {avg_epoch_loss:.4f}")

        # 计算所有epoch的平均损失
        avg_loss = total_loss / max(1, num_successful_batches)
        return avg_loss

