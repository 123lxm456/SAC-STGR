"""Nine-token spatial and temporal risk predictor for the P2 architecture."""

import math

import torch
from torch import nn


class SpatioTemporalRiskPredictor(nn.Module):
    """Score one ego vehicle from ten state/action observations.

    The eight flow tokens are aggregate traffic groups, not individual cars.
    ``valid_mask`` marks observed steps in a left-padded window; its final step
    must be valid. The module does not collect labels or connect to SUMO.
    """

    SEQUENCE_LENGTH = 10
    STATE_DIM = 98
    ACTION_DIM = 1
    FLOW_COUNT = 8
    TOKEN_COUNT = 9

    def __init__(self, d_model=64, nhead=4, spatial_layers=1,
                 temporal_layers=2, feedforward_dim=128, dropout=0.1,
                 use_recency_bias=True):
        super().__init__()
        if d_model <= 0 or nhead <= 0 or d_model % nhead:
            raise ValueError("d_model must be positive and divisible by nhead")
        if spatial_layers <= 0 or temporal_layers <= 0 or feedforward_dim <= 0:
            raise ValueError("encoder depths and feedforward_dim must be positive")

        self.d_model = d_model
        self.use_recency_bias = use_recency_bias
        self.ego_projection = nn.Linear(3, d_model)
        self.flow_projection = nn.Linear(12, d_model)
        self.token_embedding = nn.Embedding(self.TOKEN_COUNT, d_model)

        spatial_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=feedforward_dim,
            dropout=dropout, batch_first=True,
        )
        self.spatial_encoder = nn.TransformerEncoder(
            spatial_layer, num_layers=spatial_layers,
        )
        temporal_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=feedforward_dim,
            dropout=dropout, batch_first=True,
        )
        self.temporal_encoder = nn.TransformerEncoder(
            temporal_layer, num_layers=temporal_layers,
        )
        self.readout_attention = nn.MultiheadAttention(
            embed_dim=d_model, num_heads=nhead, dropout=dropout,
            batch_first=True,
        )
        self.classifier = nn.Sequential(
            nn.Linear(2 * d_model, 64), nn.ReLU(),
            nn.Linear(64, 1), nn.Sigmoid(),
        )

        positions = torch.arange(self.SEQUENCE_LENGTH, dtype=torch.float32)
        frequencies = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float32)
            * (-math.log(10000.0) / d_model)
        )
        temporal_position = torch.zeros(self.SEQUENCE_LENGTH, d_model)
        temporal_position[:, 0::2] = torch.sin(positions[:, None] * frequencies)
        temporal_position[:, 1::2] = torch.cos(
            positions[:, None] * frequencies[:d_model // 2]
        )
        self.register_buffer("temporal_position", temporal_position)
        # Every query gets the same preference for recent valid keys.
        recency = 0.2 * (positions - (self.SEQUENCE_LENGTH - 1))
        if not use_recency_bias:
            recency.zero_()
        self.register_buffer(
            "temporal_attention_bias",
            recency.unsqueeze(0).repeat(self.SEQUENCE_LENGTH, 1),
        )

    @classmethod
    def _check_inputs(cls, state_seq, action_seq, valid_mask):
        if state_seq.ndim != 3 or state_seq.shape[1:] != (
                cls.SEQUENCE_LENGTH, cls.STATE_DIM):
            raise ValueError("state_seq must have shape [B,10,98]")
        if action_seq.shape != (state_seq.shape[0], cls.SEQUENCE_LENGTH,
                                cls.ACTION_DIM):
            raise ValueError("action_seq must have shape [B,10,1]")
        if valid_mask.shape != (state_seq.shape[0], cls.SEQUENCE_LENGTH):
            raise ValueError("valid_mask must have shape [B,10]")
        if state_seq.shape[0] == 0:
            raise ValueError("the batch must contain at least one vehicle")
        if not state_seq.is_floating_point() or not action_seq.is_floating_point():
            raise TypeError("state_seq and action_seq must be floating tensors")
        if valid_mask.dtype != torch.bool:
            raise TypeError("valid_mask must be a boolean tensor")
        if state_seq.device != action_seq.device or state_seq.device != valid_mask.device:
            raise ValueError("all inputs must be on the same device")
        if not bool(valid_mask[:, -1].all()) or bool(
                (valid_mask[:, :-1] & ~valid_mask[:, 1:]).any()):
            raise ValueError("valid_mask must describe a nonempty left-padded window")

    @staticmethod
    def split_fields(state_seq, action_seq):
        """Return ego [B,T,3] and flows [B,T,8,12] in observed group order."""
        ego = torch.cat((state_seq[..., :2], action_seq), dim=-1)
        queues = state_seq[..., 2:10].unsqueeze(-1)
        waits = state_seq[..., 10:18].unsqueeze(-1)
        occupancy = state_seq[..., 18:98].reshape(
            state_seq.shape[0], state_seq.shape[1], 8, 10,
        )
        flows = torch.cat((queues, waits, occupancy), dim=-1)
        return ego, flows

    def encode_fields(self, state_seq, action_seq):
        ego, flows = self.split_fields(state_seq, action_seq)
        tokens = torch.cat((
            self.ego_projection(ego).unsqueeze(2),
            self.flow_projection(flows),
        ), dim=2)
        token_ids = torch.arange(self.TOKEN_COUNT, device=tokens.device)
        return tokens + self.token_embedding(token_ids)[None, None, :, :]

    def encode_spatial(self, tokens):
        batch, steps, token_count, width = tokens.shape
        encoded = self.spatial_encoder(tokens.reshape(batch * steps, token_count, width))
        return encoded.reshape(batch, steps, token_count, width)

    def encode_temporal(self, spatial, valid_mask):
        batch, steps, token_count, width = spatial.shape
        sequences = spatial.permute(0, 2, 1, 3).reshape(
            batch * token_count, steps, width,
        )
        sequences = sequences + self.temporal_position[None, :, :]
        padding_mask = (~valid_mask[:, None, :]).expand(
            batch, token_count, steps,
        ).reshape(batch * token_count, steps)
        # PyTorch 1.13 requires a boolean key-padding mask when an additive
        # attention bias is supplied. True excludes that historical key.
        encoded = self.temporal_encoder(
            sequences, mask=self.temporal_attention_bias,
            src_key_padding_mask=padding_mask,
        )
        return encoded.reshape(batch, token_count, steps, width).permute(0, 2, 1, 3)

    def readout(self, temporal):
        last = temporal[:, -1, :, :]
        ego = last[:, :1, :]
        flows = last[:, 1:, :]
        context, _ = self.readout_attention(
            ego, flows, flows, need_weights=False,
        )
        return self.classifier(torch.cat((ego[:, 0], context[:, 0]), dim=-1)).squeeze(-1)

    def forward(self, state_seq, action_seq, valid_mask):
        self._check_inputs(state_seq, action_seq, valid_mask)
        tokens = self.encode_fields(state_seq, action_seq)
        spatial = self.encode_spatial(tokens)
        temporal = self.encode_temporal(spatial, valid_mask)
        return self.readout(temporal)

    def save_model(self, path):
        torch.save(self.state_dict(), path)

    def load_model(self, path, map_location=None):
        self.load_state_dict(torch.load(path, map_location=map_location))
