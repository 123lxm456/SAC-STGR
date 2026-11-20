"""Regression checks for the P0/P1 audit findings, without starting SUMO."""

import contextlib
import hashlib
import importlib.util
import io
import json
import math
from pathlib import Path
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src/sac_rwb'))
from Env import Env
from algos.transformer_risk_model import RiskExperienceBuffer
from core.monitor import DataMonitor
from core.runtime import wait_for_observation, reserve_run


class RepairTests(unittest.TestCase):
    def test_reward_uses_only_eight_waiting_fields(self):
        env = Env.__new__(Env)
        state = np.arange(98, dtype=np.float32)
        env.previous_obs = {'v': state}
        np.testing.assert_array_equal(env.previous_waiting_times('v'), state[10:18])
        state[18:] = 10000
        np.testing.assert_array_equal(env.previous_waiting_times('v'), np.arange(10, 18))

    def test_context_comes_from_the_observation_not_later_vehicle_state(self):
        env = Env.__new__(Env)
        env.pre_dir_info = {'v@1': ('229', ['a', 'b'], 'v')}
        self.assertEqual(env.risk_history_context('v@1'), ('v@1', '229', ('a', 'b')))

    def test_history_resets_for_junction_segment_order_and_time_gap(self):
        base = ('v', '229', tuple(range(8)))
        alternatives = [('v', '499', base[2]), ('v@2', '229', base[2]),
                        ('v', '229', tuple(reversed(range(8))))]
        for context, step in [(c, 3) for c in alternatives] + [(base, 9)]:
            with self.subTest(context=context, step=step):
                buffer = RiskExperienceBuffer()
                for t in (1, 2):
                    buffer.add('v', np.ones(98), np.ones(1), context=base, step=t)
                self.assertEqual(buffer.get_sequence('v')[2].sum(), 2)
                buffer.add('v', np.full(98, 2), np.zeros(1), context=context, step=step)
                states, _, mask = buffer.get_sequence('v')
                self.assertEqual(mask.sum(), 1)
                np.testing.assert_array_equal(states[0, -1], np.full(98, 2))
                self.assertEqual(len(buffer), 0)  # no invented safe outcome

    def test_history_copies_inputs_and_preserves_delayed_outcome(self):
        buffer = RiskExperienceBuffer()
        state = np.ones(98)
        context = ('v', '229', tuple(range(8)))
        buffer.add('v', state, np.zeros(1), context=context, step=5)
        state[:] = 99
        buffer.update_data('v', 1, junction_id='229')
        sample = buffer.risk_active_buffer[0]
        self.assertEqual(sample['last_step'], 5)
        self.assertEqual(sample['context'], context)
        self.assertTrue(np.all(sample['state_seq'][-1] == 1))
        self.assertNotIn('v', buffer.to_use_buffer)

    def test_collision_elsewhere_does_not_label_an_old_junction(self):
        buffer = RiskExperienceBuffer()
        buffer.add('v', np.zeros(98), np.zeros(1), context=('v', '229', ()), step=5)
        buffer.update_data('v', 1, junction_id='499')
        self.assertEqual(len(buffer), 0)
        self.assertNotIn('v', buffer.to_use_buffer)

    def test_invalid_padding_and_imbalanced_batch_are_rejected(self):
        buffer = RiskExperienceBuffer()
        with self.assertRaises(ValueError):
            buffer.pad_sequence([], [])
        with self.assertRaises(ValueError):
            buffer.sample(3)

    def make_env(self, remaining=1):
        env = SimpleNamespace(config={'max_reset_wait_steps': 3}, _max_episode_steps=3,
                              sumo_interface=SimpleNamespace(tc=SimpleNamespace(
                                  simulation=SimpleNamespace(getMinExpectedNumber=lambda: remaining))))
        env.step = Mock(return_value=({}, {}, {}, {}, {}, {}))
        return env

    def test_wait_is_bounded_and_detects_exhausted_traffic(self):
        env = self.make_env()
        with self.assertRaises(RuntimeError):
            wait_for_observation(env, {})
        self.assertEqual(env.step.call_count, 3)
        env = self.make_env(remaining=0)
        with self.assertRaises(RuntimeError):
            wait_for_observation(env, {})
        env.step.assert_not_called()

    def test_observation_on_last_allowed_step_is_returned(self):
        env = self.make_env()
        env.step.side_effect = [({},) * 6, ({},) * 6, ({'v': 1}, {}, {}, {}, {}, {})]
        self.assertEqual(wait_for_observation(env, {}), {'v': 1})
        env.step.reset_mock()
        self.assertEqual(wait_for_observation(env, {'v': 2}), {'v': 2})
        env.step.assert_not_called()

    def test_run_reservation_respects_models_legacy_logs_and_markers(self):
        with tempfile.TemporaryDirectory(dir=ROOT) as temp:
            root = Path(temp)
            logs, models = root / 'logs', root / 'model'
            logs.mkdir()
            models.mkdir()
            (models / 'SAC_model_0').mkdir()
            (logs / 'SAC_example_log_1.csv').write_text('legacy', encoding='utf-8')
            first, _ = reserve_run(logs, models)
            second, _ = reserve_run(logs, models)
            self.assertEqual((first, second), (2, 3))

    def make_monitor(self):
        env = SimpleNamespace(junction_list=['A', 'B'], keywords_order=['straight'], env_step=40)
        return env, DataMonitor(env)

    def test_no_traffic_metrics_are_explicitly_undefined(self):
        env, monitor = self.make_monitor()
        metrics = monitor.get_data(env)
        self.assertTrue(math.isnan(metrics['conflict_rate']))
        self.assertEqual(metrics['conflict_denominator'], 0)
        self.assertEqual(metrics['sample_count'], 0)

    def test_monitor_window_uses_recorded_steps_and_separate_junctions(self):
        env, monitor = self.make_monitor()
        monitor.recorded_steps = [40, 41]
        monitor.data_record['A']['straight']['queue_wait'][40:42] = 1
        monitor.data_record['B']['straight']['queue_wait'][40:42] = 3
        with contextlib.redirect_stdout(io.StringIO()):
            summary = monitor.evaluate(40, 42)
        self.assertEqual(summary, {'A': 1, 'B': 3})
        monitor.total_data['avg_queue_length'] = 8
        self.assertEqual(monitor.get_data(env)['avg_quene_length'], 2)

    def test_monitor_extends_capacity_without_losing_old_data(self):
        _, monitor = self.make_monitor()
        record = monitor.data_record['A']['straight']
        record['queue_wait'][10] = 7
        monitor._ensure_capacity(5001)
        self.assertGreater(len(record['queue_wait']), 5001)
        self.assertEqual(record['queue_wait'][10], 7)

    def test_p0_recopy_uses_protected_reference_and_refuses_overwrite(self):
        spec = importlib.util.spec_from_file_location('p0_snapshot', ROOT / 'scripts/p0_snapshot.py')
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        with tempfile.TemporaryDirectory(dir=ROOT) as temp:
            destination = Path(temp) / 'work'
            report = Path(temp) / 'snapshot.json'
            with contextlib.redirect_stdout(io.StringIO()):
                module.main(destination, report)
            manifest = json.loads(report.read_text(encoding='utf-8'))
            self.assertEqual(len(manifest['files']), 23)
            for entry in manifest['files']:
                self.assertEqual(hashlib.sha256((ROOT / entry['copy']).read_bytes()).hexdigest(),
                                 entry['sha256'])
            before = report.read_bytes()
            with self.assertRaises(RuntimeError):
                module.main(destination, report)
            self.assertEqual(report.read_bytes(), before)


if __name__ == '__main__':
    unittest.main()
