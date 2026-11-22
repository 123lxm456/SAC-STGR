"""P5-only ablations; P0-P4 factories and checkpoint layouts stay unchanged."""

import torch
from torch import nn
from algos.online_risk import create_risk_predictor, OnlineSpatioTemporalRiskPredictor
from algos.transformer_risk_model import RiskExperienceBuffer

RISK_VARIANTS = ('b0', 'm1', 'a_b', 'c1', 'c2', 'c3')
VARIANTS = RISK_VARIANTS + ('s0',)


class AblationPredictor(OnlineSpatioTemporalRiskPredictor):
    def __init__(self, variant):
        super().__init__(use_recency_bias=variant != 'a_b')
        self.variant = variant
        if variant == 'c1':
            # Same 9*64 input/output, ordered group mixing without attention.
            # 29 hidden units match the replaced spatial encoder's capacity.
            self.spatial_encoder = nn.Sequential(nn.Linear(576, 29), nn.ReLU(),
                                                 nn.Linear(29, 576))
        elif variant == 'c2':
            self.pool_projection = nn.Linear(128, 64)
            del self.readout_attention
            self.classifier = nn.Sequential(nn.Linear(64, 64), nn.ReLU(),
                                            nn.Linear(64, 1), nn.Sigmoid())
        elif variant == 'c3':
            del self.readout_attention
            self.classifier = nn.Sequential(nn.Linear(64, 64), nn.ReLU(),
                                            nn.Linear(64, 1), nn.Sigmoid())
        elif variant != 'a_b':
            raise ValueError('Unknown ablation: ' + variant)
        # Parent optimizer was built before replacing modules.
        self.optimizer = torch.optim.AdamW(self.parameters(), lr=1e-5, weight_decay=1e-5)

    def encode_spatial(self, tokens):
        if self.variant != 'c1':
            return super().encode_spatial(tokens)
        batch, steps, count, width = tokens.shape
        return self.spatial_encoder(tokens.reshape(batch, steps, -1)).reshape(batch, steps, count, width)

    def encode_temporal(self, spatial, valid_mask):
        if self.variant == 'c2':
            pooled = torch.cat((spatial[:, :, 0], spatial[:, :, 1:].mean(dim=2)), dim=-1)
            spatial = self.pool_projection(pooled).unsqueeze(2)
        return super().encode_temporal(spatial, valid_mask)

    def readout(self, temporal):
        if self.variant in ('c2', 'c3'):
            return self.classifier(temporal[:, -1, 0]).squeeze(-1)
        return super().readout(temporal)


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
