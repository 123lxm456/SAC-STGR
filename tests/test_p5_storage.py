"""Lossless migration must preserve bytes, provenance, and deletion gates."""
import gzip
import hashlib
import json
from pathlib import Path
import sys
import tempfile
import os
import shutil
from types import SimpleNamespace
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
sys.path.insert(0, str(ROOT / 'src/sac_rwb'))
import p5_storage as storage
import consolidate_p5 as migrate


def mkdir(path):
    current = Path(path.anchor)
    for part in path.parts[1:]:
        current = current / part
        if not current.exists():
            current.mkdir()


class StorageTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(dir=ROOT)
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name).resolve()
        self.primary = self.root / 'plan/p5/runs' / storage.PRIMARY
        mkdir(self.primary)
        self.raw = b'{"label":0,"value":[1,2,3]}\n{"label":1}\n'
        self.digest = hashlib.sha256(self.raw).hexdigest()
        self.object = self.primary / 'data/objects' / (self.digest + '.jsonl.gz')
        mkdir(self.object.parent)
        self.object.write_bytes(gzip.compress(self.raw))
        self.old = 'plan/p5/runs/original'
        self.new = 'plan/p5/runs/' + storage.PRIMARY + '/provenance/original'
        self.alias = self.old + '/online/b0/11/online-data/samples.jsonl'
        self.entry = dict(object=self.object.relative_to(self.root).as_posix(), sha256=self.digest, bytes=len(self.raw))
        self.data = dict(version='p5-storage-v1', original_root='/original/project',
                         relocations={self.old: self.new}, journals={self.alias: self.entry})
        self.save()

    def save(self):
        (self.root / storage.INDEX).write_text(json.dumps(self.data))

    def test_compressed_read_after_original_deleted_and_old_path_relocated(self):
        # No raw file or historical directory exists in this fixture.
        for path in (self.root / self.alias, '/original/project/' + self.alias,
                     self.root / (self.new + self.alias[len(self.old):])):
            self.assertEqual(storage.journal_digest_size(path, self.root), (self.digest, len(self.raw)))
            self.assertTrue(storage.journal_exists(path, self.root))
            self.assertEqual(list(storage.iter_journal_rows(path, self.root))[1]['label'], 1)
        self.assertEqual(storage.relocated('/original/project/' + self.old, self.root), self.root / self.new)

    def test_compressed_read_is_exercised_even_when_raw_original_still_exists(self):
        path = self.root / self.alias
        mkdir(path.parent)
        path.write_bytes(b'not the indexed compressed bytes')
        self.assertEqual(storage.journal_digest_size(path, self.root), (self.digest, len(self.raw)))

    def test_tampered_gzip_and_wrong_original_hash_are_rejected(self):
        self.object.write_bytes(gzip.compress(self.raw + b'changed'))
        with self.assertRaisesRegex(ValueError, 'SHA-256/size'):
            storage.journal_digest_size(self.root / self.alias, self.root)
        self.object.write_bytes(gzip.compress(self.raw)[:-5])
        with self.assertRaises((EOFError, OSError)):
            storage.journal_digest_size(self.root / self.alias, self.root)

    def test_partial_consumer_still_checks_entire_stream(self):
        self.object.write_bytes(gzip.compress(self.raw + b'changed'))
        with self.assertRaises(ValueError):
            with storage.open_journal(self.root / self.alias, self.root) as stream:
                stream.read(1)

    def test_traversal_and_object_name_mismatch_fail(self):
        self.entry['object'] = '../' + self.digest + '.jsonl.gz'
        self.save()
        with self.assertRaises(ValueError):
            storage.journal_digest_size(self.root / self.alias, self.root)
        self.entry['object'] = 'data/wrong.jsonl.gz'
        self.save()
        with self.assertRaises(ValueError):
            storage.journal_digest_size(self.root / self.alias, self.root)

    def test_raw_reader_without_index_preserves_default_behavior(self):
        (self.root / storage.INDEX).unlink()
        path = self.root / 'ordinary.jsonl'
        path.write_bytes(self.raw)
        self.assertEqual(storage.journal_digest_size(path, self.root), (self.digest, len(self.raw)))

    def test_dataset_adapter_uses_verified_bytes_and_removes_temporary_copy(self):
        directory = self.root / self.new / 'online/b0/11/online-data'
        mkdir(directory)
        for file in ('manifest.json', 'counts.json'):
            (directory / file).write_text('{}')
        self.data['journals'][self.alias.replace('samples.jsonl', 'events.jsonl')] = dict(self.entry)
        self.save()
        consumed = []
        def reader(temp):
            consumed.append(temp)
            self.assertEqual((temp / 'samples.jsonl').read_bytes(), self.raw)
            self.assertEqual((temp / 'events.jsonl').read_bytes(), self.raw)
            return 'strict reader result'
        with patch('algos.p4_data.read_dataset', side_effect=reader):
            self.assertEqual(storage.read_online_dataset(self.root / self.old / 'online/b0/11/online-data', self.root),
                             'strict reader result')
        self.assertFalse(consumed[0].exists())

    def test_export_skips_compressed_objects_and_keeps_storage_manifest(self):
        import export_p5
        import tarfile
        with patch.object(export_p5, 'relocated', side_effect=lambda p: p):
            export_p5.export(self.primary, self.root / 'returns')
        archive = next((self.root / 'returns').glob('*.tar.gz'))
        with tarfile.open(str(archive)) as stream:
            names = stream.getnames()
        self.assertTrue(any(n.endswith('storage-index.json') for n in names))
        self.assertFalse(any('/data/objects/' in n for n in names))

    def test_finalize_without_successful_target_proof_cannot_delete(self):
        with patch.object(migrate, 'CONTROL', self.primary / 'migration'), self.assertRaisesRegex(ValueError, 'No successful'):
            migrate.finalize()
        self.assertTrue(self.object.exists())

    def test_snapshot_detects_source_replacement_before_deletion(self):
        with patch.object(migrate, 'ROOT', self.root):
            path = self.root / 'source'
            path.write_bytes(b'original')
            row = migrate.snapshot(path)
            path.write_bytes(b'changed!')
            with self.assertRaises(ValueError):
                migrate.unchanged(row)

    def test_prepare_verify_finalize_fixture_preserves_all_bytes_and_one_run(self):
        # Synthetic migration fixture, not a SUMO or research acceptance run.
        fixture = self.root / 'fixture'
        mkdir(fixture)
        primary = fixture / 'plan/p5/runs' / storage.PRIMARY
        store = fixture / 'plan/p5/storage/primary-20261123T094822803672Z'
        mkdir(primary)
        mkdir(store / 'objects')
        stage_entries = []
        for name in (*migrate.OLD, storage.PRIMARY):
            run = fixture / 'plan/p5/runs' / name
            mkdir(run / 'online/b0/11/online-data')
            (run / 'plan.json').write_text('{"synthetic":true}')
            raw = run / 'online/b0/11/online-data/samples.jsonl'
            raw.write_bytes(self.raw)
            stage_entries.append(dict(fully_decompressed_verified=True, object=self.object.name,
                                      sha256=self.digest, bytes=len(self.raw), gzip_bytes=self.object.stat().st_size,
                                      aliases=[raw.relative_to(fixture).as_posix()]))
        shutil.copyfile(self.object, store / 'objects' / self.object.name)
        stage = store / 'stage-fixture.json'
        stage.write_text(json.dumps(dict(completed=True, sources_unchanged=True, objects=stage_entries)))
        baseline = dict(primary_matrix_integrity_passed=True, missing_natural_risk_updates=[],
                        passed=False, status='synthetic_fixture')
        patches = [patch.object(migrate, 'ROOT', fixture), patch.object(migrate, 'RUN', primary),
                   patch.object(migrate, 'STORE', store), patch.object(migrate, 'CONTROL', primary / 'migration'),
                   patch('audit_p5.audit', return_value=baseline),
                   patch.object(migrate, 'storage_usage', return_value=0),
                   patch.object(migrate.subprocess, 'call', return_value=0)]
        if os.name == 'nt':
            patches += [patch.object(migrate.os, 'link', side_effect=shutil.copyfile),
                        patch.object(migrate.os, 'statvfs', create=True,
                                     return_value=SimpleNamespace(f_bavail=100, f_frsize=4096))]
        for p in patches:
            p.start()
        try:
            migrate.prepare(stage)
            self.assertTrue((fixture / 'plan/p5/runs' / migrate.OLD[0]).exists())
            plan_path = primary / 'migration/plan.json'
            saved = json.loads(plan_path.read_text())
            saved['code_signature'] = {'stale': 'reviewed verifier repair'}
            plan_path.write_text(json.dumps(saved))
            with self.assertRaisesRegex(ValueError, 'Migration code changed'):
                migrate.verify()
            migrate.rebind()
            migrate.verify()
            migrate.finalize()
            self.assertEqual([p.name for p in (fixture / 'plan/p5/runs').iterdir()], [storage.PRIMARY])
            data = json.loads((fixture / storage.INDEX).read_text())
            for alias in data['journals']:
                self.assertEqual(storage.journal_digest_size(fixture / alias, fixture), (self.digest, len(self.raw)))
            for name in migrate.OLD:
                self.assertEqual((primary / 'provenance' / name / 'plan.json').read_text(), '{"synthetic":true}')
        finally:
            for p in reversed(patches):
                p.stop()


if __name__ == '__main__':
    unittest.main()
