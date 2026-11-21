"""P4 real-outcome supervision, reusing the P1 padded histories and sampler."""

from collections import defaultdict
import hashlib
import json
from pathlib import Path

import numpy as np

from algos.transformer_risk_model import RiskExperienceBuffer


PROTOCOL = {
    'version': 'p4-h10-v1', 'history_steps': 10, 'horizon_steps': 10,
    'step_seconds': 1.0, 'interval': '(decision_step, decision_step + H]',
    'collision_scope': 'scored vehicle, physical SUMO collisions anywhere on network',
    'safe_terminal': 'SUMO arrived ID; leaving control area is not terminal',
    'censor': 'missing vehicle, reset, end of collection, or interrupted observation',
    'action': 'bounded acceleration command sent to SUMO slowDown, not measured acceleration',
    'minimum_evaluation_positives': 20, 'minimum_evaluation_negatives': 20,
    'minimum_evaluation_collision_vehicles': 5,
    'minimum_training_positives': 64, 'minimum_training_negatives': 64,
    'threshold': 0.5, 'calibration_bins': 10,
}


def route_hash(edges):
    if not edges:
        raise ValueError('A nonempty physical route is required')
    return hashlib.sha256(' '.join(edges).encode('utf-8')).hexdigest()


class HorizonRiskBuffer(RiskExperienceBuffer):
    """Stage before scoring, commit after a successful SUMO step, then label.

    Legacy terminal callbacks are intentionally ignored. Only observe() can
    resolve a window. Future labels follow the vehicle even after control ends.
    """

    def __init__(self, destination, metadata, max_size=10000, horizon=10):
        super().__init__(max_size=max_size)
        if horizon != PROTOCOL['horizon_steps']:
            raise ValueError('Changing H requires a new locked protocol version')
        self.destination = Path(destination)
        self.destination.mkdir(parents=True, exist_ok=False)
        self.metadata = dict(metadata)
        if self.metadata.get('split') not in ('train', 'validation', 'test'):
            raise ValueError('Explicit train/validation/test membership required')
        for field in ('scenario_id', 'seed', 'run_id'):
            if field not in self.metadata:
                raise ValueError('Missing metadata: ' + field)
        (self.destination / 'manifest.json').write_text(json.dumps(
            {'protocol': PROTOCOL, 'metadata': self.metadata}, indent=2), encoding='utf-8')
        self.rows = (self.destination / 'samples.jsonl').open('x', encoding='utf-8')
        self.events = (self.destination / 'events.jsonl').open('x', encoding='utf-8')
        self.horizon = horizon
        self.pending = defaultdict(list)
        self.staged = {}
        self.routes = {}
        self.counts = defaultdict(int)
        self.episode = 0
        self.last_step = None
        self.closed = False

    def set_vehicle_route(self, vehicle_id, edges):
        self.routes[vehicle_id] = route_hash(edges)

    def start_episode(self, episode):
        self.clear_pending()
        self.episode = episode
        self.last_step = None

    def add(self, vehicle_id, state, action, context=None, step=None):
        if self.closed or step is None or vehicle_id not in self.routes:
            raise ValueError('Open collector, route and simulation decision step required')
        if vehicle_id in self.staged:
            raise RuntimeError('Previous decision has not been executed')
        if self.last_step is not None and step != self.last_step:
            raise ValueError('Decision must use the latest observed simulation step')
        super().add(vehicle_id, state, action, context=context, step=step)
        states, actions, mask = self.get_sequence(vehicle_id)
        if not np.isfinite(states).all() or not np.isfinite(actions).all():
            raise ValueError('Nonfinite risk inputs')
        if not -3 <= float(actions[0, -1, 0]) <= 3:
            raise ValueError('Action is outside the actual SUMO command bounds')
        self.staged[vehicle_id] = {
            'vehicle_id': vehicle_id, 'route_hash': self.routes[vehicle_id],
            'episode': self.episode, 'decision_step': int(step),
            'context': context, 'state_seq': states[0].tolist(),
            'action_seq': actions[0].tolist(), 'valid_mask': mask[0].tolist(),
            'score': None,
        }

    def record_scores(self, scores):
        for vehicle, value in scores.items():
            if vehicle not in self.staged or not np.isfinite(value) or not 0 <= value <= 1:
                raise ValueError('Score must match a staged decision and lie in [0,1]')
            self.staged[vehicle]['score'] = float(value)

    def confirm_action(self, vehicle, action, step):
        row = self.staged.get(vehicle)
        if row is None or row['decision_step'] != step or 'executed_action' in row:
            raise ValueError('Physical command must correspond to exactly one scored decision')
        if not np.isfinite(action) or not -3 <= action <= 3 or not np.isclose(action, row['action_seq'][-1][0], atol=1e-6, rtol=0):
            raise ValueError('Sent SUMO command differs from the scored action')
        row['executed_action'] = float(action)
        self.counts['commands_sent'] += 1

    def update_data(self, vehicle_id, label, junction_id=None):
        # Env still calls the legacy API; it must not contaminate P4 labels.
        return None

    def _write(self, row, label, reason, end_step):
        result = dict(row, label=label, reason=reason, end_step=end_step,
                      horizon_steps=self.horizon, executed=True)
        self.rows.write(json.dumps(result, allow_nan=False) + '\n')
        self.rows.flush()
        self.counts['windows'] += 1
        self.counts['censored' if label is None else ('positive' if label else 'negative')] += 1
        if label is not None and self.metadata['split'] == 'train':
            sample = {key: np.asarray(result[key], dtype=dtype) for key, dtype in (
                ('state_seq', np.float32), ('action_seq', np.float32), ('valid_mask', np.bool_))}
            sample['label'] = label
            (self.risk_active_buffer if label else self.active_buffer).append(sample)

    def observe(self, step, collided, arrived, present):
        if self.closed:
            raise RuntimeError('Collector is closed')
        if self.last_step is not None and step != self.last_step + 1:
            raise ValueError('Every SUMO simulation step must be observed, including idle steps')
        collided, arrived, present = set(collided), set(arrived), set(present)
        commands = {}
        for vehicle, row in self.staged.items():
            if 'executed_action' not in row:
                # Env can skip an ID that disappeared before command dispatch.
                self.counts['unexecuted_discarded'] += 1
                self.to_use_buffer.pop(vehicle, None)
                continue
            if row['decision_step'] + 1 != step or row['score'] is None:
                raise ValueError('Only scored actions executed in this step can be committed')
            self.pending[vehicle].append(row)
            self.counts['executed_decisions'] += 1
            commands[vehicle] = row['executed_action']
        self.staged.clear()
        absent = set(self.pending) - present - collided - arrived
        event = {'episode': self.episode, 'step': step, 'collided': sorted(collided),
                 'arrived': sorted(arrived - collided), 'absent': sorted(absent),
                 'commands': commands}
        self.events.write(json.dumps(event) + '\n')
        self.events.flush()
        for vehicle in list(self.pending):
            keep = []
            for row in self.pending[vehicle]:
                # Events at exactly t+H are included. Collision takes precedence.
                if vehicle in collided:
                    self._write(row, 1, 'collision', step)
                elif vehicle in arrived:
                    self._write(row, 0, 'arrival', step)
                elif vehicle in absent:
                    self._write(row, None, 'missing_vehicle', step)
                elif step >= row['decision_step'] + self.horizon:
                    self._write(row, 0, 'horizon_observed', step)
                else:
                    keep.append(row)
            if keep:
                self.pending[vehicle] = keep
            else:
                del self.pending[vehicle]
            if vehicle in collided | arrived | absent:
                self.to_use_buffer.pop(vehicle, None)
        self.last_step = step

    def clear_pending(self, reason='episode_reset'):
        for rows in self.pending.values():
            for row in rows:
                self._write(row, None, reason, self.last_step)
        self.pending.clear()
        # Failure before simulationStep did not execute the staged action.
        self.counts['unexecuted_discarded'] += len(self.staged)
        self.staged.clear()
        super().clear_pending()

    def close(self, reason='collection_end'):
        if not self.closed:
            self.clear_pending(reason)
            self.rows.close()
            self.events.close()
            (self.destination / 'counts.json').write_text(json.dumps(dict(self.counts), indent=2), encoding='utf-8')
            self.closed = True
