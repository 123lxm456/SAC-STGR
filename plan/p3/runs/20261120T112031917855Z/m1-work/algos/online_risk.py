"""P3 adapters: shared terminal buffer, batch scoring and latency accounting."""

from collections import deque
import time

import numpy as np
import torch

from algos.risk_training import RiskTrainingMixin
from algos.spatiotemporal_risk_model import SpatioTemporalRiskPredictor
from algos.transformer_risk_model import RiskExperienceBuffer, TransformerRiskPredictor


class OnlineSpatioTemporalRiskPredictor(RiskTrainingMixin, SpatioTemporalRiskPredictor):
    """Add the existing P1 buffer/optimizer to the unchanged P2 network.

    Parameter names match the standalone P2 model, so their weight files are
    interchangeable. This is legacy terminal supervision, not the P4 horizon.
    """

    def __init__(self, buffer_size=10000, **network_options):
        super().__init__(**network_options)
        self.buffer = RiskExperienceBuffer(max_size=buffer_size)
        self.optimizer = torch.optim.AdamW(self.parameters(), lr=1e-5, weight_decay=1e-5)
        self.last_train_updates = 0


def create_risk_predictor(variant='b0', buffer_size=10000, b0_d_model=128,
                          b0_nhead=4, b0_layers=2):
    if variant == 'b0':
        model = TransformerRiskPredictor(98, 1, d_model=b0_d_model,
                                         nhead=b0_nhead, num_layers=b0_layers,
                                         buffer_size=buffer_size)
        model.network_config = {'d_model': b0_d_model, 'nhead': b0_nhead, 'temporal_layers': b0_layers}
    elif variant == 'm1':
        model = OnlineSpatioTemporalRiskPredictor(buffer_size=buffer_size)
        model.network_config = {'d_model': 64, 'nhead': 4, 'spatial_layers': 1,
                                'temporal_layers': 2, 'feedforward_dim': 128,
                                'use_recency_bias': True}
    else:
        raise ValueError('Unknown risk variant: {}'.format(variant))
    return model


def synchronize(device):
    if torch.device(device).type == 'cuda':
        torch.cuda.synchronize(device)


class OnlineRiskRuntime:
    """Append physical actions and score all current virtual IDs in one batch."""

    def __init__(self, predictor, device, retained_calls=10000):
        if retained_calls < 1:
            raise ValueError('retained_calls must be positive')
        self.predictor = predictor
        self.device = torch.device(device)
        self.records = deque(maxlen=retained_calls)
        self.calls = 0
        self.vehicles = 0
        self.total_ms = 0.0

    def score(self, env, states, physical_actions):
        if set(states) != set(physical_actions):
            raise ValueError('States and physical actions must have identical vehicle IDs')
        if not states:
            return {}
        keys = list(states)
        real_ids = [env.convert_virtual_id_to_real_id(key) for key in keys]
        if len(set(real_ids)) != len(real_ids):
            raise ValueError('A physical vehicle cannot have two active control IDs')
        synchronize(self.device)
        started = time.perf_counter()
        sequences = []
        for key, vehicle in zip(keys, real_ids):
            action = float(physical_actions[key])
            if not np.isfinite(action) or not env.min_acc <= action <= env.max_acc:
                raise ValueError('Risk inputs must use the bounded physical action sent to SUMO')
            self.predictor.buffer.add(vehicle, states[key], np.array([action], dtype=np.float32),
                                      context=env.risk_history_context(key), step=env.env_step)
            sequences.append(self.predictor.buffer.get_sequence(vehicle))
        states_batch = torch.as_tensor(np.concatenate([s[0] for s in sequences]),
                                       dtype=torch.float32, device=self.device)
        actions_batch = torch.as_tensor(np.concatenate([s[1] for s in sequences]),
                                        dtype=torch.float32, device=self.device)
        valid_batch = torch.as_tensor(np.concatenate([s[2] for s in sequences]),
                                      dtype=torch.bool, device=self.device)
        self.predictor.eval()
        synchronize(self.device)
        forward_start = time.perf_counter()
        with torch.no_grad():
            scores = self.predictor(states_batch, actions_batch, valid_batch)
        synchronize(self.device)
        forward_ms = 1000 * (time.perf_counter() - forward_start)
        values = scores.detach().cpu().numpy()
        if values.shape != (len(keys),) or not np.isfinite(values).all():
            raise RuntimeError('Risk scores must be finite with one score per vehicle')
        if not ((values >= 0) & (values <= 1)).all():
            raise RuntimeError('Risk scores must lie in [0,1]')
        result = {key: float(value) for key, value in zip(keys, values)}
        total_ms = 1000 * (time.perf_counter() - started)
        self.calls += 1
        self.vehicles += len(keys)
        self.total_ms += total_ms
        self.records.append({'batch_size': len(keys), 'forward_ms': forward_ms,
                             'end_to_end_ms': total_ms})
        return result

    def summary(self):
        values = [row['end_to_end_ms'] for row in self.records]
        return {
            'calls': self.calls, 'vehicles_scored': self.vehicles,
            'total_end_to_end_ms': self.total_ms,
            'mean_end_to_end_ms_per_batch': self.total_ms / self.calls if self.calls else None,
            'amortized_ms_per_vehicle': self.total_ms / self.vehicles if self.vehicles else None,
            'retained_calls': len(self.records),
            'p50_retained_batch_ms': float(np.percentile(values, 50)) if values else None,
            'p95_retained_batch_ms': float(np.percentile(values, 95)) if values else None,
            'definition': 'Synchronized wall time includes histories, tensor assembly, transfers, forward and CPU score mapping. Percentiles cover retained calls; first-call costs are included. Amortized cost is not single-vehicle latency.',
            'retained_records': list(self.records),
        }
