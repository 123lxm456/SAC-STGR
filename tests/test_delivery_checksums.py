"""Delivery manifests must remain consumable by Linux sha256sum on Windows."""

import hashlib
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
import package_p5
import export_p5


class ChecksumTests(unittest.TestCase):
    def check_manifest(self, archive):
        content = archive.with_suffix(archive.suffix + '.sha256').read_bytes()
        self.assertNotIn(b'\r', content)
        self.assertEqual(content, (hashlib.sha256(archive.read_bytes()).hexdigest() +
                                   '  ' + archive.name + '\n').encode('utf-8'))

    def test_windows_delivery_checksum_has_literal_lf(self):
        with tempfile.TemporaryDirectory(dir=ROOT) as temp:
            root = Path(temp)
            for name in ('README.md', 'research_implementation_plan.md', '.gitignore', '.gitattributes'):
                (root / name).write_text('synthetic fixture')
            (root / 'plan').mkdir()
            (root / 'plan/acceptance').mkdir()
            (root / 'plan/acceptance/p012-accepted-20261120.json').write_text('{}')
            archive = root / 'synthetic-delivery.tar.gz'
            with patch.object(package_p5, 'ROOT', root), patch.object(package_p5, 'source_manifest', return_value=[]):
                package_p5.build(archive)
            self.check_manifest(archive)

    def test_windows_export_checksum_has_literal_lf(self):
        with tempfile.TemporaryDirectory(dir=ROOT) as temp:
            root = Path(temp)
            run = root / 'synthetic-run'
            run.mkdir()
            (run / 'summary.json').write_text('{}')
            destination = root / 'returns'
            export_p5.export(run, destination)
            archives = list(destination.glob('*.tar.gz'))
            self.assertEqual(len(archives), 1)
            self.check_manifest(archives[0])


if __name__ == '__main__':
    unittest.main()
