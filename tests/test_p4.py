"""P4 boundary, leakage, censoring and calibration regressions (synthetic fixtures)."""

import json
import importlib.util
from pathlib import Path
import sys
import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src/sac_rwb'))
from algos.horizon_risk import HorizonRiskBuffer, route_hash
from algos.p4_data import (read_dataset, audit_datasets, create_scenarios,
                          score_metrics, fit_calibration, apply_calibration, write_reliability)


class P4Tests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(dir=ROOT)
        self.root = Path(self.temp.name)
        self.addCleanup(self.temp.cleanup)

    def buffer(self, name='data', split='train', seed=100):
        buffer = HorizonRiskBuffer(self.root / name, {
            'split': split, 'seed': seed, 'scenario_id': split,
            'run_id': name, 'frozen_risk_model': True,
            'risk_sha256': 'fixed-risk-fixture', 'policy_sha256': 'fixed-policy-fixture',
            'allowed_route_hashes': [route_hash([split])],
        })
        self.addCleanup(buffer.close)
        return buffer

    def stage(self, buffer, vehicle='v', step=0, score=.5, context=('segment', '229', ())):
        buffer.set_vehicle_route(vehicle, [buffer.metadata['split']])
        buffer.add(vehicle, np.ones(98), np.array([1.25]), context=context, step=step)
        buffer.record_scores({vehicle: score})
        buffer.confirm_action(vehicle, 1.25, step)

    def rows(self, buffer):
        buffer.close()
        return read_dataset(buffer.destination)[1]

    def test_collision_at_inclusive_horizon_labels_last_ten_executed_windows(self):
        b = self.buffer()
        for step in range(10):
            self.stage(b, step=step)
            b.observe(step + 1, ['v'] if step == 9 else [], [], ['v'])
        rows = self.rows(b)
        self.assertEqual(len(rows), 10)
        self.assertTrue(all(x['label'] == 1 and x['end_step'] == 10 for x in rows))
        self.assertEqual(len(b.risk_active_buffer), 10)
        self.assertEqual(rows[0]['action_seq'][-1], [1.25])

    def test_collision_after_horizon_cannot_relabel_mature_negative(self):
        b = self.buffer()
        self.stage(b)
        for step in range(1, 12):
            b.observe(step, ['v'] if step == 11 else [], [], ['v'])
        row = self.rows(b)[0]
        self.assertEqual((row['label'], row['end_step'], row['reason']), (0, 10, 'horizon_observed'))

    def test_idle_steps_advance_future_and_control_departure_is_not_safe(self):
        b = self.buffer()
        self.stage(b)
        b.observe(1, [], [], ['v'])
        # No further decisions after entering the junction, but outcome follows.
        b.observe(2, ['v'], [], ['v'])
        self.assertEqual(self.rows(b)[0]['label'], 1)

    def test_real_arrival_is_negative_and_collision_has_precedence(self):
        b = self.buffer()
        self.stage(b, 'arrival')
        self.stage(b, 'collision')
        b.observe(1, ['collision'], ['arrival', 'collision'], [])
        rows = {x['vehicle_id']: x for x in self.rows(b)}
        self.assertEqual(rows['arrival']['label'], 0)
        self.assertEqual(rows['collision']['label'], 1)

    def test_disappearance_reset_and_budget_end_are_censored(self):
        for reason in ('missing', 'reset', 'end'):
            b = self.buffer(reason)
            self.stage(b)
            b.observe(1, [], [], [] if reason == 'missing' else ['v'])
            if reason == 'reset':
                b.start_episode(1)
            rows = self.rows(b)
            self.assertEqual(len(rows), 1)
            self.assertIsNone(rows[0]['label'])
            self.assertEqual(len(b), 0)

    def test_unexecuted_action_is_not_a_training_or_dataset_sample(self):
        b = self.buffer()
        self.stage(b)
        self.assertEqual(self.rows(b), [])
        self.assertEqual(b.counts['unexecuted_discarded'], 1)

    def test_unsent_scored_action_is_excluded_after_successful_step(self):
        b = self.buffer()
        b.set_vehicle_route('missing', ['train'])
        b.add('missing', np.zeros(98), np.zeros(1), step=0)
        b.record_scores({'missing': .5})
        with self.assertRaisesRegex(ValueError, 'differs'):
            b.confirm_action('missing', 1, 0)
        b.observe(1, [], [], [])
        self.assertEqual(self.rows(b), [])
        self.assertEqual(b.counts['unexecuted_discarded'], 1)

    def test_legacy_callbacks_cannot_label_p4_and_holdout_never_enters_training(self):
        b = self.buffer(split='validation')
        self.stage(b)
        b.update_data('v', 0)
        b.observe(1, ['v'], [], ['v'])
        self.assertEqual(self.rows(b)[0]['label'], 1)
        self.assertEqual(len(b), 0)

    def test_context_change_resets_history_but_preserves_real_future(self):
        b = self.buffer()
        self.stage(b)
        b.observe(1, [], [], ['v'])
        self.stage(b, step=1, context=('new-segment', '499', ()))
        b.observe(2, ['v'], [], ['v'])
        rows = self.rows(b)
        self.assertEqual([sum(row['valid_mask']) for row in rows], [1, 1])
        self.assertEqual([row['label'] for row in rows], [1, 1])

    def test_skipped_simulation_step_and_unscored_commit_fail(self):
        b = self.buffer()
        self.stage(b)
        b.observe(1, [], [], ['v'])
        with self.assertRaises(ValueError):
            b.observe(3, [], [], ['v'])
        b2 = self.buffer('unscored')
        b2.set_vehicle_route('v', ['train'])
        b2.add('v', np.zeros(98), np.zeros(1), step=0)
        b2.confirm_action('v', 0, 0)
        with self.assertRaises(ValueError):
            b2.observe(1, [], [], ['v'])

    def make_partitions(self, positives=20):
        paths = []
        for index, split in enumerate(('train', 'validation', 'test')):
            b = self.buffer(split, split=split, seed=100 * (index + 1))
            collided, arrived = [], []
            training = split == 'train'
            for label, count in ((0, 64 if training else 20),
                                 (1, 64 if training and positives else positives)):
                for n in range(count):
                    vehicle = '{}-{}-{}'.format(split, label, n)
                    self.stage(b, vehicle, score=.25 if label else .75)
                    (collided if label else arrived).append(vehicle)
            b.observe(1, collided, arrived, collided)
            b.close()
            paths.append(b.destination)
        return paths

    def test_independent_natural_ratio_audit_and_validation_only_calibration(self):
        report = audit_datasets(self.make_partitions())
        self.assertTrue(report['isolation_passed'])
        self.assertTrue(report['data_ready'])
        self.assertEqual(report['calibration']['fit_samples'], 40)
        self.assertEqual(report['splits']['test']['raw_metrics']['prevalence'], .5)
        self.assertLess(report['calibrated_metrics']['test']['brier'], report['splits']['test']['raw_metrics']['brier'])
        self.assertEqual(report['splits']['test']['action']['count'], 40)
        import xml.etree.ElementTree as ET
        graph = self.root / 'reliability.svg'
        write_reliability(report, graph)
        self.assertEqual(ET.parse(str(graph)).getroot().tag, '{http://www.w3.org/2000/svg}svg')

    def test_zero_positive_dataset_does_not_claim_calibration_or_full_acceptance(self):
        report = audit_datasets(self.make_partitions(positives=0))
        self.assertFalse(report['data_ready'])
        self.assertIsNone(report['calibration'])
        self.assertIsNone(report['splits']['test']['raw_metrics']['auroc'])

    def test_cross_seed_route_scene_and_vehicle_leakage_are_rejected(self):
        paths = self.make_partitions()
        p = paths[1] / 'manifest.json'
        original = json.loads(p.read_text())
        for key, value in (('seed', 100), ('scenario_id', 'train'),
                           ('allowed_route_hashes', [route_hash(['train'])])):
            changed = json.loads(json.dumps(original))
            changed['metadata'][key] = value
            p.write_text(json.dumps(changed))
            with self.assertRaises(ValueError):
                audit_datasets(paths)
        p.write_text(json.dumps(original))
        rows = [json.loads(line) for line in (paths[1] / 'samples.jsonl').read_text().splitlines()]
        rows[0]['vehicle_id'] = 'train-0-0'
        (paths[1] / 'samples.jsonl').write_text('\n'.join(json.dumps(row) for row in rows) + '\n')
        with self.assertRaises(ValueError):
            audit_datasets(paths)

    def test_duplicate_collection_cannot_inflate_event_evidence(self):
        paths = self.make_partitions()
        with self.assertRaisesRegex(ValueError, 'Duplicate collection'):
            audit_datasets(paths + [paths[0]])

    def test_finalized_horizon_pools_update_both_predictors(self):
        import torch
        from algos.online_risk import create_risk_predictor
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        for variant in ('b0', 'm1'):
            model = create_risk_predictor(variant).to(device)
            b = self.buffer('optimization-' + variant)
            positive, negative = ['p0', 'p1'], ['n0', 'n1']
            for vehicle in positive + negative:
                self.stage(b, vehicle)
            # Still no resolved labels before the physical outcome is observed.
            self.assertEqual(len(b), 0)
            b.observe(1, positive, negative, positive)
            model.buffer = b
            before = next(model.parameters()).detach().clone()
            loss = model.train_model(batch_size=4, epochs=1, device=device)
            self.assertTrue(np.isfinite(loss))
            self.assertEqual(model.last_train_updates, 1)
            self.assertFalse(torch.equal(before, next(model.parameters())))

    def test_target_environment_mismatch_stops_p4_collection(self):
        from unittest.mock import Mock
        spec = importlib.util.spec_from_file_location('accept_p4_test', ROOT / 'scripts/accept_p4.py')
        module = importlib.util.module_from_spec(spec)
        with patch.object(sys, 'path', [str(ROOT / 'scripts')] + sys.path):
            spec.loader.exec_module(module)
        gate = self.root / 'plan/review'
        (self.root / 'plan').mkdir()
        gate.mkdir()
        (gate / 'p0123-target-confirmation-20261121.json').write_text(json.dumps(
            {'passed': True, 'p4_engineering_ready': True}))
        run_command = Mock(side_effect=AssertionError('Must stop before collection'))
        with patch.object(module, 'ROOT', self.root), \
             patch.object(module, 'source_manifest', return_value=[]), \
             patch.object(module, 'collect_environment', return_value={'target_errors': ['wrong GPU']}), \
             patch.object(module.subprocess, 'run', run_command), \
             patch.object(sys, 'argv', ['accept_p4.py']):
            self.assertEqual(module.main(), 1)
        result = json.loads(next((self.root / 'plan/p4/runs').glob('*/summary.json')).read_text())
        self.assertFalse(result['passed'])
        self.assertFalse(result['engineering_passed'])
        self.assertFalse(result['p5_started'])
        run_command.assert_not_called()

    def test_frozen_collector_entrypoint_commits_real_actions_and_never_optimizes(self):
        import torch
        from algos.online_risk import create_risk_predictor
        from sac.utils import Actor
        spec = importlib.util.spec_from_file_location('collect_p4_test', ROOT / 'scripts/collect_p4.py')
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)

        class StubEnv:
            max_acc, min_acc = 3, -3

            def __init__(self, config):
                self.config = config
                self._max_episode_steps = 4
                self.env_step = 0
                self.vehicles = {'v': SimpleNamespace(route=('train',))}

            def convert_virtual_id_to_real_id(self, key):
                return key

            def risk_history_context(self, key):
                return key, '229', tuple(range(8))

            def reset(self):
                return {'v': np.zeros(98, dtype=np.float32)}, {}

            def step(self, actions, scores):
                for vehicle, action in actions.items():
                    self.config['risk_event_collector'].confirm_action(vehicle, float(action), self.env_step)
                self.env_step += 1
                self.config['risk_event_collector'].observe(
                    self.env_step, ['v'] if self.env_step == 2 else [], [], ['v'])
                return {'v': np.zeros(98, dtype=np.float32)}, {}, {}, {}, {}, {}

            def close(self):
                pass

        scene = self.root / 'scenarios.json'
        scene.write_text(json.dumps({'scenarios': {'train': {
            'split': 'train', 'seed': 100, 'scenario_id': 'fixture-scene',
            'route_hashes': [route_hash(['train'])], 'config': 'unused', 'network': 'unused'}}}))
        actor = self.root / 'actor.pth'
        torch.save(Actor(98, 1, (256, 256)).state_dict(), actor)
        for variant in ('b0', 'm1'):
            risk = self.root / (variant + '.pth')
            create_risk_predictor(variant).save_model(risk)
            output = self.root / variant
            args = SimpleNamespace(scenarios=str(scene), split='train', replicate=0,
                                   actor=str(actor), risk_model=str(risk), variant=variant,
                                   device='cpu', output=str(output), steps=4)
            with patch.object(module, 'Env', StubEnv), \
                 patch.object(torch.optim.AdamW, 'step', side_effect=AssertionError('Frozen collector optimized')):
                module.collect(args)
            manifest, rows, _ = read_dataset(output)
            self.assertTrue(manifest['metadata']['frozen_risk_model'])
            self.assertEqual(len(rows), 4)
            self.assertEqual(sum(row['label'] == 1 for row in rows), 2)
            status = json.loads((output / 'collection.json').read_text())
            self.assertEqual(status['decision_steps'], 4)
            self.assertEqual(status['sac_updates'], 0)
            self.assertEqual(status['risk_updates'], 0)

    def test_tampered_label_and_incomplete_dataset_are_rejected(self):
        paths = self.make_partitions()
        p = paths[0] / 'samples.jsonl'
        lines = p.read_text().splitlines()
        row = json.loads(lines[0]); row['label'] = 1
        p.write_text('\n'.join([json.dumps(row)] + lines[1:]) + '\n')
        with self.assertRaisesRegex(ValueError, 'label disagrees'):
            read_dataset(paths[0])
        p.write_text('\n'.join(lines[1:]) + '\n')
        with self.assertRaisesRegex(ValueError, 'Unaccounted decisions'):
            read_dataset(paths[0])

    def test_metrics_have_known_values_and_order_independent_ties(self):
        report = score_metrics([1, 0, 1, 0], [.9, .8, .7, .6])
        self.assertAlmostEqual(report['auroc'], .75)
        self.assertAlmostEqual(report['auprc_average_precision'], 5 / 6)
        for labels in ([1, 0], [0, 1]):
            tied = score_metrics(labels, [.5, .5])
            self.assertEqual(tied['auroc'], .5)
            self.assertEqual(tied['auprc_average_precision'], .5)
        with self.assertRaises(ValueError):
            fit_calibration([0, 0], [.1, .2])
        self.assertTrue(np.isfinite(apply_calibration([0, 1], {'slope': 1, 'intercept': 0})).all())

    def test_generated_scenarios_have_disjoint_complete_routes_and_seeds(self):
        result = create_scenarios(ROOT / 'src/sac_rwb/real_data', self.root / 'scenes')
        scenes = list(result['scenarios'].values())
        for i, first in enumerate(scenes):
            self.assertGreater(first['demand_elements'], 0)
            self.assertTrue(Path(first['config']).is_file())
            for second in scenes[i + 1:]:
                self.assertFalse(set(first['route_hashes']) & set(second['route_hashes']))
                self.assertNotEqual(first['seed'], second['seed'])
                self.assertNotEqual(first['scenario_id'], second['scenario_id'])


if __name__ == '__main__':
    unittest.main()
