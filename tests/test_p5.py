"""Synthetic functional fixtures only; never P5 effectiveness evidence."""

import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src/sac_rwb'))
sys.path.insert(0, str(ROOT / 'scripts'))
from algos.p5_models import create_p5_predictor, VARIANTS
from algos.p5_experiment import balanced_update, fit_offline, paired_summary, seed_all
from algos.p5_metrics import PhysicalMetrics, audit_physical
from accept_p5 import validate_protocol, protocol_hash, scenarios_at, execute, smoke_protocol
from audit_p5 import audit, system_references, within


class P5Tests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(dir=ROOT)
        self.addCleanup(self.temp.cleanup)
        self.path = Path(self.temp.name)

    def batch(self):
        rng = np.random.RandomState(4)
        states = rng.normal(size=(4, 10, 98)).astype(np.float32)
        actions = rng.uniform(-3, 3, (4, 10, 1)).astype(np.float32)
        masks = np.ones((4, 10), dtype=np.bool_)
        masks[0, :-1] = False
        states[~masks] = 0
        actions[~masks] = 0
        return states, actions, masks, np.array([0, 1, 0, 1], np.float32)

    def test_all_variants_forward_padding_action_gradient_and_roundtrip(self):
        batch = self.batch()
        for variant in VARIANTS:
            with self.subTest(variant=variant):
                seed_all(7)
                model = create_p5_predictor(variant).eval()
                states, actions, mask = [torch.as_tensor(x) for x in batch[:3]]
                actions = actions.clone().requires_grad_()
                scores = model(states, actions, mask)
                self.assertEqual(tuple(scores.shape), (4,))
                self.assertTrue(torch.isfinite(scores).all())
                if variant == 's0':
                    self.assertTrue((scores == 0).all())
                    continue
                scores.sum().backward()
                self.assertGreater(actions.grad[:, -1].abs().sum().item(), 0)
                self.assertTrue((actions.grad[~mask] == 0).all())
                changed = states.clone()
                changed[~mask] = 500
                self.assertTrue(torch.allclose(scores, model(changed, actions, mask), atol=1e-6))
                path = self.path / (variant + '.pth')
                torch.save(model.state_dict(), str(path))
                copy = create_p5_predictor(variant).eval()
                copy.load_state_dict(torch.load(str(path), map_location='cpu'))
                self.assertTrue(torch.equal(scores, copy(states, actions, mask)))

    def test_c1_capacity_matches_m1_and_removed_ablation_modules_are_absent(self):
        counts = {v: sum(p.numel() for p in create_p5_predictor(v).parameters()) for v in ('m1', 'c1')}
        self.assertLessEqual(abs(counts['c1'] / counts['m1'] - 1), .1)
        for variant in ('c2', 'c3'):
            self.assertFalse(hasattr(create_p5_predictor(variant), 'readout_attention'))
        self.assertTrue((create_p5_predictor('a_b').temporal_attention_bias == 0).all())

    def test_ablation_optimizer_owns_exactly_current_parameters_and_updates(self):
        for variant in ('a_b', 'c1', 'c2', 'c3'):
            model = create_p5_predictor(variant)
            actual = {id(p) for group in model.optimizer.param_groups for p in group['params']}
            self.assertEqual(actual, {id(p) for p in model.parameters()})
            before = model.classifier[-2].weight.detach().clone()
            loss = balanced_update(model, self.batch(), 'cpu')
            self.assertTrue(np.isfinite(loss))
            self.assertFalse(torch.equal(before, model.classifier[-2].weight))

    def test_offline_training_is_fixed_budget_and_test_labels_do_not_select_weights(self):
        batch = self.batch()
        corpus = {split: batch for split in ('train', 'validation', 'test')}
        with patch('algos.p5_experiment.benchmark', return_value={'fixture': True}):
            first = fit_offline('c1', 3, corpus, 2, 4, 'cpu', self.path / 'one', corpus_id='same')
            changed = dict(corpus, test=batch[:3] + (1 - batch[3],))
            second = fit_offline('c1', 3, changed, 2, 4, 'cpu', self.path / 'two', corpus_id='same')
        self.assertEqual(first['updates'], 2)
        self.assertEqual(first['checkpoint_sha256'], second['checkpoint_sha256'])
        self.assertEqual(first['calibration'], second['calibration'])
        self.assertNotEqual(first['splits']['test'], second['splits']['test'])

    def row(self, step, pairs=(), departed=(), arrived=()):
        return dict(step=step, pairs=list(pairs), departed=list(departed), arrived=list(arrived),
                    collided=sorted({v for pair in pairs for v in pair}), vehicles=2,
                    groups=32, speed_sum=10.0, queue_sum=64.0, wait_sum=96.0)

    def test_physical_pairs_deduplicate_contiguous_reports_not_vehicle_observations(self):
        metrics = PhysicalMetrics()
        rows = [self.row(1, [('a', 'b')], ['a', 'b', 'c']), self.row(2, [('b', 'a')]),
                self.row(3), self.row(4, [('a', 'b')], arrived=['c'])]
        for row in rows:
            metrics.add(row)
        result = metrics.summary()
        self.assertEqual(result['collision_pair_events'], 2)
        self.assertEqual(result['collision_vehicles'], 2)
        self.assertEqual(result['successful_arrivals'], 1)
        self.assertEqual(result['aql'], 2)
        self.assertEqual(result['awt'], 3)
        self.assertEqual(result['mean_speed'], 5)
        path = self.path / 'synthetic_physical.jsonl'
        path.write_text(''.join(json.dumps(r) + '\n' for r in rows))
        self.assertEqual(audit_physical(path), result)

    def test_missing_physical_step_pair_details_and_departure_denominator_fail_closed(self):
        with self.assertRaises(ValueError):
            PhysicalMetrics().add(self.row(2))
        row = self.row(1)
        row['collided'] = ['a']
        with self.assertRaises(ValueError):
            PhysicalMetrics().add(row)
        metrics = PhysicalMetrics()
        metrics.add(self.row(1, arrived=['unknown']))
        with self.assertRaises(ValueError):
            metrics.summary()

    def test_locked_protocol_rejects_missing_groups_duplicate_seeds_and_test_overlap(self):
        original = json.loads((ROOT / 'plan/p5/protocol.json').read_text())
        validate_protocol(original)
        for key, value in (('training_seeds', [1, 1, 2]), ('training_seeds', [3000, 2, 3]),
                           ('policy_variants', ['b0', 'm1']), ('research_run', False)):
            changed = json.loads(json.dumps(original))
            changed[key] = value
            with self.assertRaises(ValueError):
                validate_protocol(changed)
        self.assertEqual(protocol_hash(original), protocol_hash(json.loads(json.dumps(original))))

    def test_sparse_scene_smoke_budget_can_fill_sac_replay_without_changing_research_protocol(self):
        locked = json.loads((ROOT / 'plan/p5/protocol.json').read_text())
        before = protocol_hash(locked)
        smoke = smoke_protocol(locked)
        self.assertGreaterEqual(smoke['online']['decision_steps'], 2 * 256)
        self.assertEqual(smoke['online']['episode_decisions'], smoke['online']['decision_steps'])
        self.assertFalse(smoke['research_run'])
        self.assertEqual(smoke['evaluation']['simulation_steps'], 100)
        self.assertEqual(before, protocol_hash(locked))

    def test_scenarios_keep_split_routes_and_scale_only_flow_density(self):
        protocol = json.loads((ROOT / 'plan/p5/protocol.json').read_text())
        scenes = scenarios_at(self.path, protocol)
        self.assertFalse(set(scenes['train']['route_hashes']) & set(scenes['test-high']['route_hashes']))
        self.assertEqual(scenes['test-low']['route_hashes'], scenes['test-high']['route_hashes'])
        self.assertNotEqual(scenes['test-low']['scenario_id'], scenes['test-high']['scenario_id'])

    def test_failed_worker_exitcode_and_log_are_preserved(self):
        commands = []
        with self.assertRaises(RuntimeError):
            execute(self.path, 'synthetic-fail', [sys.executable, '-c', 'raise SystemExit(7)'], 20, commands)
        self.assertEqual(commands[0]['exitcode'], 7)
        self.assertEqual((self.path / 'synthetic-fail.exitcode').read_text().strip(), '7')

    def test_local_environment_cannot_grant_target_acceptance(self):
        plan = {'protocol': {}, 'protocol_sha256': protocol_hash({})}
        (self.path / 'plan.json').write_text(json.dumps(plan))
        (self.path / 'environment.json').write_text(json.dumps({'cuda_available': False}))
        with self.assertRaisesRegex(ValueError, 'target environment'):
            audit(self.path)

    def test_system_references_are_required_and_cannot_escape_artifact_root(self):
        result = system_references(self.path, {'protocol': {'system_references_required': ['ppo', 'fcfs', 'mixed']}})
        self.assertFalse(result['passed'])
        self.assertEqual(len(result['missing']), 3)
        with self.assertRaises(ValueError):
            within(self.path, '../elsewhere')

    def test_aggregation_uses_training_seed_units_and_preserves_negative_differences(self):
        result = paired_summary([-1, 0, 1])
        self.assertEqual(result['n'], 3)
        self.assertEqual(result['mean'], 0)
        self.assertEqual(result['std'], 1)
        with self.assertRaises(ValueError):
            paired_summary([1])

    @unittest.skipUnless(torch.cuda.is_available(), 'CUDA is unavailable')
    def test_ablation_cuda_real_update(self):
        for variant in ('a_b', 'c1', 'c2', 'c3'):
            model = create_p5_predictor(variant).cuda()
            self.assertTrue(np.isfinite(balanced_update(model, self.batch(), 'cuda')))


if __name__ == '__main__':
    unittest.main()
