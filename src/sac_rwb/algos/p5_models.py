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
    else:
        model = AblationPredictor(variant)
    model.network_config = dict(getattr(model, 'network_config', {}), p5_variant=variant)
    return model
