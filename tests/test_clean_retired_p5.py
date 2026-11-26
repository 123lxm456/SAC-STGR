"""Deletion policy fixtures: preserve models, provenance and linked dependencies."""
import json
import os
from pathlib import Path
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
from clean_retired_p5 import RETIRED, candidate, inventory, apply_scan
from unittest.mock import patch


def mkdir(path):
    # Windows UNC cannot stat some deeply missing parents.
    current = ROOT
    for part in path.relative_to(ROOT).parts:
        current = current / part
        current.mkdir(exist_ok=True)


class CleanupTests(unittest.TestCase):
    def test_policy_is_scoped_to_dedicated_products(self):
        for variant in RETIRED:
            self.assertTrue(candidate('runs/run/online/{}/11/risk.pth'.format(variant)))
            self.assertTrue(candidate('runs/run/work-{}-11/file'.format(variant)))
            self.assertTrue(candidate('runs/run/train-{}-11.log'.format(variant)))
        for path in ('runs/run/online/m1/11/risk.pth', 'runs/run/source.tar.gz',
                     'runs/run/commands.json', 'protocol.json', 'best.pth',
                     'runs/run/online/a_b/11/events.jsonl'):
            self.assertFalse(candidate(path))

    def test_inventory_hashes_candidates_and_preserves_mixed_records(self):
        with tempfile.TemporaryDirectory(dir=ROOT) as folder:
            root = Path(folder)
            variant = sorted(RETIRED)[0]
            path = root / 'runs/run/online' / variant / '11'
            mkdir(path)
            (path / 'result.json').write_text(json.dumps({'variant': variant}))
            (path / 'risk.pth').write_bytes(b'weight')
            (root / 'runs/run/commands.json').write_text('[]')
            report = inventory(root)
            self.assertEqual(report['candidate_count'], 2)
            self.assertTrue(all('sha256' in r for r in report['files'] if r['delete']))
            self.assertFalse(next(r for r in report['files'] if r['path'].endswith('commands.json'))['delete'])
            (path / 'result.json').write_text(json.dumps({'variant': 'm1'}))
            with self.assertRaisesRegex(ValueError, 'identity'):
                inventory(root)

    def test_inbound_preserved_link_blocks_deletion(self):
        with tempfile.TemporaryDirectory(dir=ROOT) as folder:
            root = Path(folder)
            target = root / 'runs/run/online' / sorted(RETIRED)[0] / '11'
            mkdir(target)
            (target / 'risk.pth').write_bytes(b'important')
            link = root / 'retained-link'
            try:
                link.symlink_to(target, target_is_directory=True)
            except OSError:
                self.skipTest('OS does not permit symlink creation')
            self.assertEqual(len(inventory(root)['blocked_links']), 1)

    @unittest.skipUnless(os.name == 'posix', 'Deletion runs only on Linux')
    def test_apply_refuses_changed_inventory_and_preserves_other_weights(self):
        with tempfile.TemporaryDirectory(dir=ROOT) as folder:
            root = Path(folder)
            target = root / 'runs/run/online' / sorted(RETIRED)[0] / '11'
            mkdir(target)
            weight = target / 'risk.pth'
            weight.write_bytes(b'retired')
            kept = root / 'best.pth'
            kept.write_bytes(b'retained')
            scan = inventory(root)
            weight.write_bytes(b'changed')
            with self.assertRaisesRegex(ValueError, 'inventory changed'):
                apply_scan(root, scan, root / 'receipt.json')
            self.assertTrue(weight.is_file())
            report = apply_scan(root, inventory(root), root / 'receipt.json')
            self.assertEqual(report['status'], 'completed')
            self.assertFalse(weight.exists())
            self.assertEqual(kept.read_bytes(), b'retained')

    @unittest.skipUnless(os.name == 'posix', 'Deletion runs only on Linux')
    def test_active_worker_prevents_deletion(self):
        with tempfile.TemporaryDirectory(dir=ROOT) as folder:
            with patch('clean_retired_p5.active_processes', return_value=[123]):
                with self.assertRaisesRegex(ValueError, 'active'):
                    apply_scan(Path(folder), {}, Path(folder) / 'receipt.json')


if __name__ == '__main__':
    unittest.main()
