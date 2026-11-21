"""Train-only extension planning must preserve holdouts and reject reused seeds."""

import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]


class ExtensionTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(dir=ROOT)
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.base = self.root / 'base'
        self.base.mkdir()
        spec = importlib.util.spec_from_file_location('extend_p4_test', ROOT / 'scripts/extend_p4_train.py')
        self.module = importlib.util.module_from_spec(spec)
        with patch.object(sys, 'path', [str(ROOT / 'scripts')] + sys.path):
            spec.loader.exec_module(self.module)
        self.write(self.base / 'summary.json', {
            'mode': 'target', 'engineering_passed': True, 'data_ready': False,
            'commands': [{'name': variant + '-train-0',
                          'argv': ['collect', '--actor', variant + '-actor', '--risk-model', variant + '-risk']}
                         for variant in ('b0', 'm1')]})
        self.write(self.base / 'unit-results.json', {
            'passed': True, 'tests_run': 74, 'require_cuda': True, 'skipped': [], 'errors': [], 'failures': []})
        self.write(self.base / 'protocol.json', {'version': 'fixture-v1'})
        scenes = self.base / 'scenarios'; scenes.mkdir()
        self.write(scenes / 'scenarios.json', {'scenarios': {'train': {'seed': 100}}})
        for variant in ('b0', 'm1'):
            parent = self.base / variant; parent.mkdir()
            self.training(parent / 'train-0', variant, 100)

    def write(self, path, data):
        path.write_text(json.dumps(data), encoding='utf-8')

    def training(self, path, variant, seed):
        path.mkdir()
        self.write(path / 'manifest.json', {
            'protocol': {'version': 'fixture-v1'}, 'metadata': {
                'split': 'train', 'seed': seed, 'frozen_risk_model': True,
                'scenario_id': 'train-scene', 'policy_sha256': variant + '-actor',
                'risk_sha256': variant + '-risk', 'allowed_route_hashes': ['route-a']}})

    def test_only_unused_training_seeds_are_planned_with_frozen_holdout_paths(self):
        plan = self.module.plan_extension(self.base)
        self.assertEqual(plan['seeds'], [101, 102, 103])
        self.assertEqual(plan['replicates'], [1, 2, 3])
        self.assertEqual(plan['steps'], 2000)
        for variant in ('b0', 'm1'):
            self.assertEqual(plan['training_collections'][variant], [str(self.base / variant / 'train-0')])
            self.assertEqual(plan['evaluation_collections'][variant], [
                str(self.base / variant / 'validation-0'), str(self.base / variant / 'test-0')])
        self.assertFalse(plan['p5_started'])

    def previous(self, repeat=False, wrong_weights=False):
        prior = self.root / 'previous'; prior.mkdir()
        collections = {}
        for variant in ('b0', 'm1'):
            parent = prior / variant; parent.mkdir()
            directories = [str(self.base / variant / 'train-0')]
            for index in (1, 2, 3):
                path = parent / ('train-' + str(index))
                self.training(path, variant, 100 if repeat else 100 + index)
                if wrong_weights:
                    data = json.loads((path / 'manifest.json').read_text())
                    data['metadata']['risk_sha256'] = 'changed'
                    self.write(path / 'manifest.json', data)
                directories.append(str(path))
            collections[variant] = directories
        self.write(prior / 'summary.json', {'engineering_passed': True, 'base_run': str(self.base),
                                           'training_collections': collections})
        return prior

    def test_further_extension_preserves_prior_training_and_uses_fresh_seeds(self):
        prior = self.previous()
        plan = self.module.plan_extension(self.base, previous=prior)
        self.assertEqual(plan['seeds'], [104, 105, 106])
        self.assertEqual(len(plan['training_collections']['b0']), 4)

    def test_repeated_seed_is_rejected(self):
        with self.assertRaisesRegex(ValueError, 'Repeated training seed'):
            self.module.plan_extension(self.base, previous=self.previous(repeat=True))

    def test_changed_checkpoint_cannot_be_mixed_into_training(self):
        with self.assertRaisesRegex(ValueError, 'checkpoints changed'):
            self.module.plan_extension(self.base, previous=self.previous(wrong_weights=True))

    def test_non_target_or_skipped_base_tests_cannot_grant_extension(self):
        path = self.base / 'summary.json'
        summary = json.loads(path.read_text()); summary['mode'] = 'local'; self.write(path, summary)
        with self.assertRaises(ValueError):
            self.module.plan_extension(self.base)
        summary['mode'] = 'target'; self.write(path, summary)
        self.write(self.base / 'unit-results.json', {'passed': True, 'require_cuda': True, 'skipped': ['cuda']})
        with self.assertRaisesRegex(ValueError, 'without skips'):
            self.module.plan_extension(self.base)

    def test_invalid_budget_is_rejected(self):
        for count, steps in ((0, 2000), (50, 2000), (3, 99)):
            with self.subTest(count=count, steps=steps), self.assertRaises(ValueError):
                self.module.plan_extension(self.base, replicates=count, steps=steps)


if __name__ == '__main__':
    unittest.main()
