"""The entry verifier must not grant acceptance from local/skipped/stale proof."""

import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]


class VerifierTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(dir=ROOT)
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        spec = importlib.util.spec_from_file_location('verify_stage_test', ROOT / 'scripts/verify_p01234.py')
        self.module = importlib.util.module_from_spec(spec)
        with patch.object(sys, 'path', [str(ROOT / 'scripts')] + sys.path):
            spec.loader.exec_module(self.module)
        self.write('summary.json', {'mode': 'target', 'passed': True,
                                    'commands': [{'name': 'unit', 'exitcode': 0}]})
        self.write('unit-results.json', {'passed': True, 'require_cuda': True,
                                         'tests_run': 1, 'skipped': [], 'failures': [], 'errors': []})
        (self.root / 'unit.exitcode').write_text('0\n')
        (self.root / 'unit.log').write_text('one passing target test\n')

    def write(self, name, value):
        (self.root / name).write_text(json.dumps(value), encoding='utf-8')

    def test_only_completed_target_history_is_accepted(self):
        self.assertTrue(self.module.checked_history(self.root, 1)['passed'])
        self.write('summary.json', {'mode': 'local', 'passed': True,
                                    'commands': [{'name': 'unit', 'exitcode': 0}]})
        with self.assertRaisesRegex(ValueError, 'target historical'):
            self.module.checked_history(self.root, 1)

    def test_summary_cannot_override_a_failed_real_exitcode(self):
        (self.root / 'unit.exitcode').write_text('1\n')
        with self.assertRaisesRegex(ValueError, 'command failed'):
            self.module.checked_history(self.root, 1)

    def test_skipped_cuda_cannot_grant_gpu_acceptance(self):
        self.write('unit-results.json', {'passed': True, 'require_cuda': True,
                                         'tests_run': 1, 'skipped': ['CUDA unavailable'], 'errors': [], 'failures': []})
        with self.assertRaisesRegex(ValueError, 'incomplete or skipped'):
            self.module.checked_history(self.root, 1)

    def test_saved_paths_cannot_escape_the_project(self):
        with patch.object(self.module, 'ROOT', self.root):
            self.assertEqual(self.module.local_saved_path('/server/project/plan/x', '/server/project'),
                             (self.root / 'plan/x').resolve())
            for path in ('/server/other/x', '/server/project/../outside'):
                with self.subTest(path=path), self.assertRaises(ValueError):
                    self.module.local_saved_path(path, '/server/project')

    def test_semantic_cleanup_check_still_detects_algorithm_changes(self):
        old = 'import math\nclass timer:\n    pass\ndef score(x):\n    return x + 1\n'
        cleaned = 'def score(x):\n    return x + 1\n'
        changed = 'def score(x):\n    return x + 2\n'
        self.assertEqual(self.module.semantic_body(old), self.module.semantic_body(cleaned))
        self.assertNotEqual(self.module.semantic_body(old), self.module.semantic_body(changed))


if __name__ == '__main__':
    unittest.main()
