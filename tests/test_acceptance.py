"""The acceptance collector must fail closed and preserve existing evidence."""

import contextlib
import importlib.util
import io
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

# Import libraries before mocking subprocess.run: torch's Windows platform
# detection can itself invoke `ver` through subprocess during initial import.
import numpy
import torch

ROOT = Path(__file__).resolve().parents[1]


class AcceptanceTests(unittest.TestCase):
    def module(self):
        spec = importlib.util.spec_from_file_location('accept_p012', ROOT / 'scripts/accept_p012.py')
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module

    def exercise(self, local, environment, codes):
        module = self.module()
        with tempfile.TemporaryDirectory(dir=ROOT) as temp:
            root = Path(temp)
            (root / 'plan').mkdir()
            argv = ['accept_p012.py'] + (['--local'] if local else [])
            with patch.object(module, 'ROOT', root), \
                 patch.object(module, 'source_manifest', return_value=[]), \
                 patch.object(module, 'collect_environment', return_value=environment), \
                 patch.object(module.platform, 'platform', return_value='test-platform'), \
                 patch.object(module.subprocess, 'run', side_effect=[
                     SimpleNamespace(returncode=code) for code in codes]) as called, \
                 patch.object(sys, 'argv', argv), \
                 contextlib.redirect_stdout(io.StringIO()), \
                 contextlib.redirect_stderr(io.StringIO()):
                code = module.main()
            path = next((root / 'plan/acceptance').glob('*/summary.json'))
            return code, json.loads(path.read_text(encoding='utf-8')), called.call_count

    def test_environment_mismatch_cannot_grant_acceptance(self):
        code, summary, calls = self.exercise(False, {'target_errors': ['wrong GPU']}, [])
        self.assertEqual((code, calls), (1, 0))
        self.assertFalse(summary['passed'])
        self.assertFalse(summary['p3_started'])
        self.assertEqual(summary['stages'], {})

    def test_local_success_cannot_grant_target_acceptance(self):
        code, summary, calls = self.exercise(True, {}, [0, 0])
        self.assertEqual((code, calls), (0, 2))
        self.assertTrue(summary['local_checks_passed'])
        self.assertFalse(summary['passed'])
        self.assertEqual(summary['stages'], {})

    def test_failed_p0_stops_before_p1_or_p3(self):
        code, summary, calls = self.exercise(False, {'target_errors': []}, [0, 0, 1])
        self.assertEqual((code, calls), (1, 3))
        self.assertFalse(summary['passed'])
        self.assertFalse(summary['p3_started'])
        self.assertEqual(summary['stages'], {})

    def test_existing_evidence_is_not_overwritten(self):
        module = self.module()
        with tempfile.TemporaryDirectory(dir=ROOT) as temp:
            root = Path(temp)
            (root / 'plan').mkdir()
            (root / 'plan/acceptance').mkdir()
            run = root / 'plan/acceptance/existing'
            run.mkdir()
            evidence = run / 'summary.json'
            evidence.write_text('original evidence', encoding='utf-8')
            with patch.object(module, 'ROOT', root), \
                 patch.object(sys, 'argv', ['accept_p012.py', '--run-dir', str(run)]):
                with self.assertRaises(FileExistsError):
                    module.main()
            self.assertEqual(evidence.read_text(encoding='utf-8'), 'original evidence')


if __name__ == '__main__':
    unittest.main()
