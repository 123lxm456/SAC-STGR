"""Only the reviewed protocol/source pair may project immutable old evidence."""
import copy
import json
from pathlib import Path
import sys
import tarfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
from p5_scope import projected, canonical_runtime, record, included_job


class ScopeTests(unittest.TestCase):
    def legacy(self):
        return json.loads((ROOT / 'plan/p5/runs/20261123T094822803672Z/plan.json').read_text(encoding='utf-8'))['protocol']

    def test_projection_keeps_budgets_seeds_and_data_definition(self):
        old = self.legacy()
        new = projected(old)
        self.assertNotEqual(old, new)
        for field in ('training_seeds', 'online', 'offline', 'evaluation', 'common_corpus',
                      'label_protocol', 'checkpoint_selection', 'calibration', 'system_references_required'):
            self.assertEqual(old[field], new[field])
        self.assertEqual(new['policy_variants'], ['b0', 'm1', 'a_b', 's0'])
        changed = copy.deepcopy(old)
        changed['online']['decision_steps'] -= 1
        self.assertEqual(projected(changed), changed)
        for variant in set(old['policy_variants']) - set(new['policy_variants']):
            self.assertFalse(included_job('train-{}-11'.format(variant), old))
        self.assertTrue(included_job('train-m1-11', old))

    def test_runtime_mapping_does_not_hide_other_source_changes(self):
        proof = record()
        path = proof['runtime_path']
        with tarfile.open(str(ROOT / 'plan/p5/runs/20261123T094822803672Z/source.tar.gz')) as archive:
            old = archive.extractfile(path).read()
        current = (ROOT / path).read_bytes()
        self.assertEqual(canonical_runtime(path, old), canonical_runtime(path, current))
        self.assertNotEqual(canonical_runtime(path, old + b'\n# unexpected change'), canonical_runtime(path, current))
        self.assertNotEqual(canonical_runtime('src/other.py', old), canonical_runtime('src/other.py', current))


if __name__ == '__main__':
    unittest.main()
