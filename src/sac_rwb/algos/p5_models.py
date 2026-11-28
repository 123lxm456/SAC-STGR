"""P5-only ablations; P0-P4 factories and checkpoint layouts stay unchanged."""

import torch
from torch import nn
from algos.online_risk import create_risk_predictor, OnlineSpatioTemporalRiskPredictor
from algos.transformer_risk_model import RiskExperienceBuffer

RISK_VARIANTS = ('b0', 'm1', 'a_b')
VARIANTS = RISK_VARIANTS + ('s0',)


class AblationPredictor(OnlineSpatioTemporalRiskPredictor):
    def __init__(self, variant):
        if variant != 'a_b':
            raise ValueError('Unknown ablation: ' + variant)
        super().__init__(use_recency_bias=False)
        self.variant = variant
        # Preserve the existing A-B optimizer and checkpoint layout.
        self.optimizer = torch.optim.AdamW(self.parameters(), lr=1e-5, weight_decay=1e-5)


class CandidatePredictor(OnlineSpatioTemporalRiskPredictor):
    """Post-P5 candidate: retain the nine-token architecture, learn temporal recency.

    A-B's better aggregate means motivate a zero initial coefficient. A bounded
    scalar permits the common corpus to determine its direction without adding
    a new attention block. This is an unvalidated candidate, not historical M1.
    """
    def __init__(self):
        super().__init__(use_recency_bias=False)
        del self.temporal_attention_bias
        self.recency_logit = nn.Parameter(torch.zeros(()))
        self.optimizer = torch.optim.AdamW(self.parameters(), lr=1e-5, weight_decay=1e-5)
        self.network_config = dict(d_model=64, nhead=4, spatial_layers=1,
                                   temporal_layers=2, feedforward_dim=128,
                                   recency='learned bounded scalar, initialized at zero')

    def encode_temporal(self, spatial, valid_mask):
        batch, steps, token_count, width = spatial.shape
        sequences = spatial.permute(0, 2, 1, 3).reshape(batch * token_count, steps, width)
        sequences = sequences + self.temporal_position[None, :, :]
        padding_mask = (~valid_mask[:, None, :]).expand(batch, token_count, steps).reshape(
            batch * token_count, steps)
        positions = torch.arange(steps, device=spatial.device, dtype=spatial.dtype)
        beta = 0.2 * torch.tanh(self.recency_logit)
        attention_bias = beta * (positions - (steps - 1))[None, :].expand(steps, steps)
        encoded = self.temporal_encoder(sequences, mask=attention_bias,
                                        src_key_padding_mask=padding_mask)
        return encoded.reshape(batch, token_count, steps, width).permute(0, 2, 1, 3)

class NoRiskPredictor(nn.Module):
    """S0 preserves physical action/history collection, with exactly zero risk."""
    def __init__(self):
        super().__init__()
        self.buffer = RiskExperienceBuffer()

    def forward(self, states, actions, mask):
        return states.new_zeros(states.shape[0])


def create_p5_predictor(variant):
    if variant in ('b0', 'm1'):
        model = create_risk_predictor(variant)
    elif variant == 's0':
        model = NoRiskPredictor()
    elif variant == 'm2':
        model = CandidatePredictor()
    else:
        model = AblationPredictor(variant)
    model.network_config = dict(getattr(model, 'network_config', {}), p5_variant=variant)
    return model
