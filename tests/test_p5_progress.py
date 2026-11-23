"""Partial-review hashes tolerate only the identified Git newline conversion."""
import hashlib
from pathlib import Path
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
from review_p5_progress import physical_digest, equivalent


class ProgressTests(unittest.TestCase):
    def test_newline_copy_matches_original_but_changed_event_is_rejected(self):
        with tempfile.TemporaryDirectory(dir=ROOT) as temporary:
            path = Path(temporary) / 'synthetic.jsonl'
            original = b'{"step":1}\n'
            expected = hashlib.sha256(original).hexdigest()
            path.write_bytes(original)
            self.assertEqual(physical_digest(path, expected), 'original-bytes')
            path.write_bytes(original.replace(b'\n', b'\r\n'))
            self.assertIn('CRLF', physical_digest(path, expected))
            path.write_bytes(b'{"step":2}\r\n')
            with self.assertRaises(ValueError):
                physical_digest(path, expected)

    def test_recomputed_float_tolerance_does_not_hide_counter_or_schema_changes(self):
        self.assertTrue(equivalent({'value': .3}, {'value': .3 + 1e-12}))
        self.assertFalse(equivalent({'value': 1000}, {'value': 1001}))
        self.assertFalse(equivalent({'value': 10000000}, {'value': 10000001}))
        self.assertFalse(equivalent({'value': True}, {'value': 1}))
        self.assertFalse(equivalent({'value': .3}, {'other': .3}))


if __name__ == '__main__':
    unittest.main()
