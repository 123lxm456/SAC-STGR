"""P3 online wiring checks; environment stubs never launch SUMO."""

import importlib
import io
import json
import os
from pathlib import Path
import runpy
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch, Mock

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src/sac_rwb'))
from Env import Env
from algos.online_risk import create_risk_predictor, OnlineRiskRuntime
from algos.spatiotemporal_risk_model import SpatioTemporalRiskPredictor


class ControlContext:
    min_acc = -3.0
    max_acc = 3.0

    def __init__(self):
        self.env_step = 1
        self.junction = '229'

    def convert_virtual_id_to_real_id(self, key):
        return key.split('@')[0]

    def risk_history_context(self, key):
        return key, self.junction, tuple(range(8))


class P3Tests(unittest.TestCase):
    def test_training_entrypoint_runs_both_variants_with_stubbed_sumo(self):
        env_module = importlib.import_module('Env')
        sac = importlib.import_module('sac.SAC')
        buffer_type = sac.ReplayBuffer

        class StubEnv(ControlContext):
            n_obs, n_action, max_acc = 98, 1, 3.0

            def __init__(self, config):
                super().__init__()
                self.config = config
                self._max_episode_steps = config['max_episode_steps']
                self.use_safety_predictor = config['use_safety_predictor']
                self.risk_predictor = config['risk_predictor']
                self.reward_record = {}
                self.monitor = SimpleNamespace(get_data=lambda env: dict.fromkeys(
                    ('max_quene_length', 'avg_quene_length', 'quene_wait', 'conflict', 'conflict_rate', 'avg_speed'), 0))

            def reset(self):
                self.env_step = 0
                return {'v': np.zeros(98, dtype=np.float32)}, {}

            def step(self, actions, scores):
                self.env_step += 1
                self.reward_record['v'] = {'efficiency': 0, 'safety': -3 * scores['v']}
                state = np.full(98, self.env_step, dtype=np.float32)
                done = self.env_step == 2
                if done:
                    self.risk_predictor.buffer.update_data('v', 0)
                return {'v': state}, {'v': -3 * scores['v']}, {'v': done, '__all__': False}, {}, {}, {'v': state}

            def close(self):
                pass

        for variant in ('b0', 'm1'):
            with self.subTest(variant=variant), tempfile.TemporaryDirectory(dir=ROOT) as temp:
                previous_cwd = Path.cwd()
                try:
                    os.chdir(temp)
                    writer = Mock()
                    with patch.dict(sys.modules, {'torch.utils.tensorboard': SimpleNamespace(SummaryWriter=lambda path: writer)}), \
                         patch.object(env_module, 'Env', StubEnv), \
                         patch.object(sac, 'ReplayBuffer', side_effect=lambda s, a, max_size, dvc:
                                      buffer_type(s, a, 8, dvc)), \
                         patch.object(sys, 'argv', ['train_sac.py', '--risk_variant', variant,
                                      '--dvc', 'cpu', '--epochs', '1', '--max_e_steps', '2',
                                      '--save_interval', '2', '--update_every', '1', '--batch_size', '2', '--net_width', '8']):
                        namespace = runpy.run_path(str(ROOT / 'src/sac_rwb/train_sac.py'), run_name='p3_entry_test')
                        namespace['main']()
                    report = json.loads(Path('Train_logs/unsignalizedIntersections/risk_runtime_0.json').read_text(encoding='utf-8'))
                    self.assertEqual(report['risk_variant'], variant)
                    self.assertEqual(report['inference']['calls'], 2)
                    self.assertEqual(report['sac_updates'], 5)
                    self.assertEqual(report['risk_updates'], 0)
                    self.assertTrue(Path('model/SAC_model_0/unsignalizedIntersections_actor_2.pth').is_file())
                    writer.close.assert_called_once()
                finally:
                    os.chdir(previous_cwd)

    def test_factory_preserves_b0_and_specified_m1_width(self):
        self.assertEqual(create_risk_predictor('b0').embedding.out_features, 128)
        self.assertEqual(create_risk_predictor('m1').d_model, 64)
        with self.assertRaises(ValueError):
            create_risk_predictor('unknown')

    def test_m1_optimizer_is_persistent_and_counts_actual_updates(self):
        torch.manual_seed(29)
        model = create_risk_predictor('m1')
        self.assertEqual(model.train_model(batch_size=4, epochs=1), 0)
        self.assertEqual(model.last_train_updates, 0)
        for label in (0, 1):
            for index in range(2):
                vehicle = '{}-{}'.format(label, index)
                model.buffer.add(vehicle, np.full(98, label, dtype=np.float32), np.zeros(1))
                model.buffer.update_data(vehicle, label)
        optimizer = model.optimizer
        before = model.classifier[2].weight.detach().clone()
        for _ in range(2):
            loss = model.train_model(batch_size=4, epochs=1)
            self.assertTrue(np.isfinite(loss))
            self.assertEqual(model.last_train_updates, 1)
        self.assertIs(model.optimizer, optimizer)
        self.assertFalse(torch.equal(before, model.classifier[2].weight))
        self.assertEqual(next(iter(optimizer.state.values()))['step'], 2)

    def test_adapter_weights_load_into_the_standalone_p2_model(self):
        model = create_risk_predictor('m1').eval()
        standalone = SpatioTemporalRiskPredictor().eval()
        self.assertEqual(set(model.state_dict()), set(standalone.state_dict()))
        checkpoint = io.BytesIO()
        model.save_model(checkpoint)
        checkpoint.seek(0)
        standalone.load_model(checkpoint, map_location='cpu')
        states, actions = torch.randn(2, 10, 98), torch.randn(2, 10, 1)
        mask = torch.tensor([[False] * 9 + [True], [True] * 10])
        with torch.no_grad():
            self.assertTrue(torch.equal(model(states, actions, mask), standalone(states, actions, mask)))

    def test_batch_scoring_preserves_vehicle_mapping_and_physical_actions(self):
        model = create_risk_predictor('m1')
        runtime, env = OnlineRiskRuntime(model, 'cpu'), ControlContext()
        states = {'alpha@1': np.ones(98), 'beta@2': np.full(98, 2)}
        actions = {'beta@2': 1.75, 'alpha@1': -2.5}
        with patch.object(model, 'forward', wraps=model.forward) as forward:
            scores = runtime.score(env, states, actions)
        self.assertEqual(list(scores), list(states))
        self.assertEqual(forward.call_count, 1)
        args = forward.call_args[0]
        self.assertEqual(tuple(args[0].shape), (2, 10, 98))
        np.testing.assert_allclose(args[1][:, -1, 0].numpy(), [-2.5, 1.75])
        self.assertEqual(args[2].sum(dim=1).tolist(), [1, 1])
        self.assertFalse(model.training)
        self.assertEqual(set(model.buffer.to_use_buffer), {'alpha', 'beta'})
        self.assertTrue(all(0 <= value <= 1 for value in scores.values()))

    def test_consecutive_histories_and_context_resets_remain_independent(self):
        model, env = create_risk_predictor('m1'), ControlContext()
        runtime = OnlineRiskRuntime(model, 'cpu')
        runtime.score(env, {'a': np.zeros(98)}, {'a': 0})
        env.env_step = 2
        runtime.score(env, {'a': np.zeros(98), 'b': np.zeros(98)}, {'a': 1, 'b': -1})
        self.assertEqual(model.buffer.get_sequence('a')[2].sum(), 2)
        self.assertEqual(model.buffer.get_sequence('b')[2].sum(), 1)
        model.buffer.update_data('a', 1, junction_id='229')
        self.assertEqual(len(model.buffer.risk_active_buffer), 1)
        self.assertNotIn('a', model.buffer.to_use_buffer)
        env.env_step, env.junction = 20, '499'
        runtime.score(env, {'a@3': np.ones(98)}, {'a@3': 2})
        self.assertEqual(model.buffer.get_sequence('a')[2].sum(), 1)

    def test_runtime_empty_and_invalid_inputs_do_not_issue_predictions(self):
        model, env = create_risk_predictor('m1'), ControlContext()
        runtime = OnlineRiskRuntime(model, 'cpu')
        self.assertEqual(runtime.score(env, {}, {}), {})
        self.assertEqual(runtime.summary()['calls'], 0)
        with self.assertRaises(ValueError):
            runtime.score(env, {'a': np.zeros(98)}, {})
        with self.assertRaises(ValueError):
            runtime.score(env, {'a': np.zeros(98)}, {'a': 3.1})
        with self.assertRaises(ValueError):
            runtime.score(env, {'a': np.zeros(98), 'a@2': np.zeros(98)}, {'a': 0, 'a@2': 0})
        with patch.object(model, 'forward', return_value=torch.tensor([float('nan')])):
            with self.assertRaises(RuntimeError):
                runtime.score(env, {'a': np.zeros(98)}, {'a': 0})
        self.assertEqual(runtime.calls, 0)

    def test_latency_records_are_bounded_and_totals_are_not_truncated(self):
        model, env = create_risk_predictor('m1'), ControlContext()
        runtime = OnlineRiskRuntime(model, 'cpu', retained_calls=2)
        for step in range(3):
            env.env_step = step
            runtime.score(env, {'a': np.zeros(98)}, {'a': 0})
        report = runtime.summary()
        self.assertEqual((report['calls'], report['vehicles_scored'], report['retained_calls']), (3, 3, 2))
        self.assertGreater(report['p95_retained_batch_ms'], 0)
        self.assertGreater(report['total_end_to_end_ms'], sum(x['end_to_end_ms'] for x in report['retained_records']))

    def test_sac_action_risk_reward_replay_and_update_chain(self):
        sac = importlib.import_module('sac.SAC')
        buffer_type = sac.ReplayBuffer
        with patch.object(sac, 'ReplayBuffer', side_effect=lambda s, a, max_size, dvc:
                          buffer_type(s, a, max_size=8, dvc=dvc)):
            agent = sac.SAC_countinuous(state_dim=98, action_dim=1, net_width=8,
                                       dvc=torch.device('cpu'), a_lr=3e-4, c_lr=3e-4,
                                       adaptive_alpha=False, alpha=0.12, batch_size=2, gamma=0.99)
        state = np.zeros(98, dtype=np.float32)
        normalized = agent.select_action([state], deterministic=False).reshape(-1)[0]
        action = float(normalized * 3)
        model = create_risk_predictor('m1')
        score = OnlineRiskRuntime(model, 'cpu').score(ControlContext(), {'v': state}, {'v': action})['v']
        env = Env.__new__(Env)
        env.max_acc, env.reward_record = 3, {}
        vehicle = type('Vehicle', (), {'id': 'v'})()
        baseline = env.reward_compute(vehicle, np.zeros(8), action, '229', 'topstraight', 0, 0)
        reward = env.reward_compute(vehicle, np.zeros(8), action, '229', 'topstraight', score, 0)
        self.assertAlmostEqual(reward - baseline, -3 * score)
        for terminal in (False, True):
            agent.replay_buffer.add(state, np.array(normalized), reward, state, terminal)
        self.assertEqual(agent.train()['updates'], 5)
        self.assertEqual(agent.replay_buffer.dw[:2, 0].tolist(), [False, True])

    @unittest.skipUnless(torch.cuda.is_available(), 'CUDA is unavailable')
    def test_cuda_m1_online_scoring_and_optimizer(self):
        model = create_risk_predictor('m1').cuda()
        runtime = OnlineRiskRuntime(model, 'cuda')
        for label in (0, 1):
            for index in range(2):
                key = '{}-{}'.format(label, index)
                runtime.score(ControlContext(), {key: np.zeros(98)}, {key: -2.0})
                model.buffer.update_data(key, label, junction_id='229')
        self.assertTrue(np.isfinite(model.train_model(batch_size=4, epochs=1, device='cuda')))
        self.assertEqual(model.last_train_updates, 1)
        self.assertEqual(runtime.summary()['calls'], 4)


if __name__ == '__main__':
    unittest.main()
