"""Post-P3 fault handling regressions; no CUDA or live SUMO required."""

import importlib
from pathlib import Path
import runpy
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src/sac_rwb'))
from algos.transformer_risk_model import TransformerRiskPredictor
from algos.online_risk import OnlineSpatioTemporalRiskPredictor
from core.sumo_interface import SUMO
from Env import Env
from core.costomized_data_structures import Container, Vehicle


class ReviewTests(unittest.TestCase):
    def test_environment_declares_the_continuous_acceleration_it_actually_uses(self):
        module = importlib.import_module('Env')
        config = {'cfg': 'unused', 'render': False, 'map_xml': 'unused',
                  'spawn_rl_prob': {}, 'probablity_RL': 1, 'junction_list': [],
                  'traffic_light_program': {}, 'use_safety_predictor': False}
        with patch.object(module, 'SUMO'), patch.object(module, 'NetMap'), patch.object(Env, 'init_env'):
            env = Env(config)
        self.assertEqual(env.action_space.shape, (1,))
        self.assertTrue(env.action_space.contains(np.array([2.5], dtype=np.float32)))
        self.assertFalse(env.action_space.contains(np.array([3.1], dtype=np.float32)))
        self.assertEqual(env.action_space.low.tolist(), [-3])
        self.assertEqual(env.action_space.high.tolist(), [3])

    def test_arrival_removal_preserves_a_terminal_replay_transition(self):
        env = Env.__new__(Env)
        vehicle = Vehicle(id='v', type='RL')
        env.vehicles, env.rl_vehicles = Container(v=vehicle), Container(v=vehicle)
        env.previous_obs = {'v': np.zeros(98, dtype=np.float32)}
        env.previous_dones = {'v': False, '__all__': False}
        env.pre_dir_info = {'v': ('229', (), 'v')}
        env.veh_name_mapping_table = {'v': ('v', False)}
        env.min_acc, env.max_acc = -3, 3
        env.use_safety_predictor = False
        env._step, env._max_episode_steps = 0, 100
        env.inner_lane_obs = {}
        env._print_debug, env._traffic_light_program_update, env._update_obs = Mock(), Mock(), Mock()
        env.print_debug = False
        env.monitor = SimpleNamespace(step=Mock())
        env.sumo_interface = SimpleNamespace(
            set_max_speed_all=Mock(), accl_control=Mock(), step=Mock(),
            tc=SimpleNamespace(vehicle=SimpleNamespace(getIDList=lambda: ['v'])),
            get_sim_info=lambda: SimpleNamespace(departed_vehicles_ids=[],
                                                 arrived_vehicles_ids=['v'], colliding_vehicles_ids=[]))
        obs, rewards, dones, _, _, replay_next = env.step_once({'v': 0})
        self.assertTrue(dones['v'])
        self.assertEqual(rewards['v'], 0)
        self.assertNotIn('v', env.rl_vehicles)
        np.testing.assert_array_equal(obs['v'], env.previous_obs['v'])
        self.assertEqual(replay_next['v'].shape, (98,))

    def test_b0_cuda_errors_are_not_replaced_with_fabricated_scores(self):
        model = TransformerRiskPredictor(98, 1, d_model=16, num_layers=1)
        states, actions = torch.zeros(1, 10, 98), torch.zeros(1, 10, 1)
        mask = torch.ones(1, 10, dtype=torch.bool)
        for error in ('CUDA out of memory', 'an illegal memory access was encountered'):
            with self.subTest(error=error), patch.object(model.embedding, 'forward', side_effect=RuntimeError(error)):
                with self.assertRaisesRegex(RuntimeError, error):
                    model(states, actions, mask)

    def test_both_trainers_propagate_errors_instead_of_returning_zero_loss(self):
        models = [TransformerRiskPredictor(98, 1, d_model=16, num_layers=1),
                  OnlineSpatioTemporalRiskPredictor(d_model=16, feedforward_dim=32)]
        for model in models:
            for label in (0, 1):
                for index in range(2):
                    vehicle = '{}-{}'.format(label, index)
                    model.buffer.add(vehicle, np.zeros(98), np.zeros(1))
                    model.buffer.update_data(vehicle, label)
            for error in ('CUDA out of memory', 'an illegal memory access was encountered'):
                with self.subTest(model=type(model).__name__, error=error), \
                     patch.object(model, 'forward', side_effect=RuntimeError(error)):
                    with self.assertRaisesRegex(RuntimeError, error):
                        model.train_model(batch_size=4, epochs=1)
                    self.assertEqual(model.last_train_updates, 0)

    def test_sumo_startup_failure_is_bounded_and_explicit(self):
        module = importlib.import_module('core.sumo_interface')
        simulator = SUMO.__new__(SUMO)
        simulator.sumo_cmd = ['sumo']
        error = module.traci.exceptions.FatalTraCIError('test connection failure')
        with patch.object(module.sumolib.miscutils, 'getFreeSocketPort', return_value=44000), \
             patch.object(module.traci, 'start', side_effect=error) as start:
            with self.assertRaisesRegex(RuntimeError, 'failed after 3 attempts') as raised:
                simulator.start_sumo(None, tries=3)
        self.assertEqual(start.call_count, 3)
        self.assertIs(raised.exception.__cause__, error)

    def test_evaluation_closes_environment_when_checkpoint_initialization_fails(self):
        env_module = importlib.import_module('Env')
        sac = importlib.import_module('sac.SAC')
        environment = SimpleNamespace(n_obs=98, n_action=1, max_acc=3, close=Mock())
        with patch.object(env_module, 'Env', return_value=environment), \
             patch.object(sac, 'SAC_countinuous', side_effect=RuntimeError('checkpoint init failed')), \
             patch.object(sys, 'argv', ['test_sac.py', '--dvc', 'cpu', '--run_num', '0', '--timestep', '100']):
            namespace = runpy.run_path(str(ROOT / 'src/sac_rwb/test_sac.py'), run_name='review_eval')
            with self.assertRaisesRegex(RuntimeError, 'checkpoint init failed'):
                namespace['main']()
        environment.close.assert_called_once()


if __name__ == '__main__':
    unittest.main()
