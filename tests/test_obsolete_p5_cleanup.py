"""Cleanup must refuse research runs and retain logs and the successful run."""
import json
from pathlib import Path
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
from clean_obsolete_p5_results import OBSOLETE_RUNS, clean, candidates


class CleanupTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(dir=ROOT)
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        current = self.root
        for name in ('plan', 'p5', 'runs'):
            current = current / name
            current.mkdir()
        for name in OBSOLETE_RUNS:
            run = current / name
            run.mkdir()
            (run / 'summary.json').write_text(json.dumps(dict(mode='smoke', status='failed', passed=False)))
            (run / 'plan.json').write_text(json.dumps(dict(protocol=dict(research_run=False, online=dict(decision_steps=100)))))
            (run / 'risk.pth').write_bytes(b'synthetic checkpoint')
            (run / 'test-predictions.npz').write_bytes(b'synthetic prediction')
            (run / 'train.log').write_text('failure evidence')
        self.formal = current / '20261122T144249871603Z'
        self.formal.mkdir()
        (self.formal / 'actor.pth').write_bytes(b'formal checkpoint')

    def test_dry_run_and_apply_keep_formal_results_and_failure_logs(self):
        report = clean(self.root)
        self.assertEqual(report['file_count'], 4)
        self.assertTrue(all((self.root / item['path']).is_file() for item in report['files']))
        done = clean(self.root, True)
        self.assertTrue(done['applied'])
        self.assertTrue(all(not (self.root / item['path']).exists() for item in done['files']))
        self.assertTrue((self.formal / 'actor.pth').is_file())
        self.assertTrue(all((self.root / 'plan/p5/runs' / name / 'train.log').is_file() for name in OBSOLETE_RUNS))

    def test_reclassified_research_or_successful_run_refuses_cleanup(self):
        first = self.root / 'plan/p5/runs' / OBSOLETE_RUNS[0]
        (first / 'summary.json').write_text(json.dumps(dict(mode='research', status='passed', passed=True)))
        with self.assertRaises(ValueError):
            clean(self.root, True)
        self.assertTrue((first / 'risk.pth').is_file())


if __name__ == '__main__':
    unittest.main()
