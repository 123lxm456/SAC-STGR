"""Task continuation preserves actual execution provenance and immutable data."""
import copy
import gzip
import hashlib
import json
from pathlib import Path
import sys
import shutil
import tarfile
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
sys.path.insert(0, str(ROOT / 'src/sac_rwb'))
import p5_reuse
from algos.p5_experiment import make_directory
from accept_p5 import active_worker_pids
from p5_storage import INDEX


class ReuseTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(dir=ROOT)
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.patch_root = patch.object(p5_reuse, 'ROOT', self.root)
        self.patch_root.start()
        self.addCleanup(self.patch_root.stop)
        self.origin = self.root / 'plan/p5/runs/origin'
        self.run = self.root / 'plan/p5/runs/continuation'
        make_directory(self.origin, new=True)
        make_directory(self.run, new=True)
        self.plan = dict(protocol={'budget': 5000}, protocol_sha256='locked', common_corpus_sha256='frozen',
                         scenarios={'train': {'scenario_id': 'same', 'route_hashes': ['route']}})
        for run in (self.origin, self.run):
            (run / 'plan.json').write_text(json.dumps(self.plan))
        self.worker = self.root / 'worker-fixture.py'
        self.worker.write_text('def unchanged():\n    return 1\n')
        for run in (self.origin, self.run):
            with tarfile.open(str(run / 'source.tar.gz'), 'w:gz') as archive:
                archive.add(str(self.worker), arcname='scripts/p5_worker.py')
        self.name = 'offline-b0-11'
        self.relative = 'offline/b0/11'
        self.original = dict(name=self.name, argv=['python', 'p5_worker.py', 'offline', '--seed', '11'], exitcode=0)
        (self.origin / 'commands.json').write_text(json.dumps([self.original]))
        (self.run / 'commands.json').write_text('[]')
        (self.origin / (self.name + '.log')).write_text('original successful execution\n')
        (self.origin / (self.name + '.exitcode')).write_text('0\n')
        source = self.origin / self.relative
        make_directory(source, new=True)
        (source / 'risk.pth').write_bytes(b'synthetic weights')
        (source / 'result.json').write_text('{"synthetic_fixture": true}')
        self.proposal = dict(jobs={self.name: self.relative})

    def manager(self, proposal=None, plan=None):
        with patch.object(p5_reuse, 'inspect', return_value=proposal or self.proposal):
            return p5_reuse.ReuseManager(self.origin, self.run, plan or self.plan)

    def test_only_complete_commands_and_matching_training_dependencies_are_selected(self):
        report = dict(online=[dict(variant='b0', seed=11, execution_confirmed=True),
                              dict(variant='s0', seed=11, execution_confirmed=False)],
                      offline=[dict(variant='m1', seed=11, execution_confirmed=False)],
                      closed_loop=[dict(variant='s0', seed=11, density='low', evaluation_seed=3000, execution_confirmed=True)])
        self.assertEqual(p5_reuse.completed_jobs(report), {'train-b0-11': 'online/b0/11'})

    def test_complete_task_is_copied_with_original_argv_log_exitcode_and_hashes(self):
        manager = self.manager()
        commands = []
        self.assertTrue(manager.reuse(self.name, self.run / self.relative, commands))
        self.assertEqual(commands[0]['argv'], self.original['argv'])
        self.assertEqual(commands[0]['reused_from'], str(self.origin.resolve()))
        self.assertEqual((self.run / (self.name + '.log')).read_bytes(), (self.origin / (self.name + '.log')).read_bytes())
        p5_reuse.verify_reuse(self.run)
        self.assertEqual((self.origin / 'commands.json').read_text(), json.dumps([self.original]))

    def test_changed_protocol_or_common_corpus_refuses_reuse(self):
        for field, value in (('protocol', {'budget': 2}), ('common_corpus_sha256', 'different')):
            changed = copy.deepcopy(self.plan)
            changed[field] = value
            with self.subTest(field=field), self.assertRaises(ValueError):
                self.manager(plan=changed)

    def test_missing_raw_journals_cannot_turn_a_review_snapshot_into_training_evidence(self):
        proposal = dict(jobs={'train-b0-11': 'online/b0/11'})
        make_directory(self.origin / 'online/b0/11/online-data', new=True)
        manager = self.manager(proposal=proposal)
        with self.assertRaisesRegex(ValueError, 'Raw training journals absent'):
            manager.reuse('train-b0-11', self.run / 'online/b0/11', [])

    def test_online_reuse_keeps_raw_journal_links_and_origin_metadata(self):
        name, relative = 'train-b0-11', 'online/b0/11'
        folder = self.origin / relative
        make_directory(folder / 'online-data', new=True)
        for file in ('samples.jsonl', 'events.jsonl'):
            (folder / 'online-data' / file).write_text('synthetic fixture\n')
        (folder / 'result.json').write_text('{"synthetic_fixture": true}')
        command = dict(name=name, argv=['python', 'worker', 'train'], exitcode=0)
        (self.origin / 'commands.json').write_text(json.dumps([command]))
        (self.origin / (name + '.exitcode')).write_text('0')
        (self.origin / (name + '.log')).write_text('actual prior command fixture')
        manager = self.manager(proposal=dict(jobs={name: relative}))
        # Exercise transport on Windows without requiring symlink privileges.
        with patch.object(p5_reuse, 'link_journal', side_effect=lambda source, target: shutil.copy2(str(source), str(target))) as linker:
            self.assertTrue(manager.reuse(name, self.run / relative, []))
        self.assertEqual(linker.call_count, 2)
        p5_reuse.verify_reuse(self.run)
        self.assertEqual((folder / 'online-data/samples.jsonl').read_text(), 'synthetic fixture\n')

    def test_relocated_origin_and_compressed_journals_keep_provenance_checks(self):
        name, relative = 'train-b0-11', 'online/b0/11'
        folder = self.origin / relative
        make_directory(folder / 'online-data', new=True)
        raw = b'synthetic fixture\n'
        for filename in ('samples.jsonl', 'events.jsonl'):
            (folder / 'online-data' / filename).write_bytes(raw)
        (folder / 'result.json').write_text('{"synthetic_fixture": true}')
        command = dict(name=name, argv=['python', 'worker', 'train'], exitcode=0)
        (self.origin / 'commands.json').write_text(json.dumps([command]))
        (self.origin / (name + '.exitcode')).write_text('0')
        (self.origin / (name + '.log')).write_text('actual prior command fixture')
        manager = self.manager(proposal=dict(jobs={name: relative}))
        with patch.object(p5_reuse, 'link_journal', side_effect=lambda a, b: shutil.copy2(str(a), str(b))):
            manager.reuse(name, self.run / relative, [])
        registry = self.root / INDEX
        make_directory(registry.parent, new=True)
        sha = hashlib.sha256(raw).hexdigest()
        obj = registry.parent / (sha + '.jsonl.gz')
        obj.write_bytes(gzip.compress(raw))
        journals = {}
        for directory in (self.origin, self.run):
            for filename in ('samples.jsonl', 'events.jsonl'):
                path = directory / relative / 'online-data' / filename
                journals[path.relative_to(self.root).as_posix()] = dict(
                    object=obj.relative_to(self.root).as_posix(), bytes=len(raw), sha256=sha)
                path.unlink()
        moved = self.run / 'provenance/origin'
        make_directory(moved.parent, new=True)
        self.origin.resolve().relative_to(self.root.resolve())
        moved.resolve().relative_to(self.root.resolve())
        self.origin.rename(moved)
        registry.write_text(json.dumps(dict(version='p5-storage-v1', original_root=str(self.root),
            relocations={self.origin.relative_to(self.root).as_posix(): moved.relative_to(self.root).as_posix()},
            journals=journals)))
        p5_reuse.verify_reuse(self.run)
        obj.write_bytes(gzip.compress(raw + b'changed'))
        with self.assertRaises(ValueError):
            p5_reuse.verify_reuse(self.run)

    def test_orphan_worker_detection_matches_exact_project_script(self):
        proc = self.root / 'synthetic-proc'
        make_directory(proc / '123', new=True)
        make_directory(proc / '456', new=True)
        (proc / '123/cmdline').write_bytes(b'python\0/project/scripts/p5_worker.py\0train\0')
        (proc / '456/cmdline').write_bytes(b'python\0/another-project/p5_worker.py\0')
        self.assertEqual(active_worker_pids(proc, '/project/scripts/p5_worker.py'), [123])

    def test_modified_inherited_weights_or_original_failure_are_rejected(self):
        manager = self.manager()
        manager.reuse(self.name, self.run / self.relative, [])
        (self.run / self.relative / 'risk.pth').write_bytes(b'changed')
        with self.assertRaisesRegex(ValueError, 'Inherited artifact changed'):
            p5_reuse.verify_reuse(self.run)
        (self.run / self.relative / 'risk.pth').write_bytes(b'synthetic weights')
        (self.origin / (self.name + '.exitcode')).write_text('1')
        with self.assertRaisesRegex(ValueError, 'not confirmed successful'):
            p5_reuse.verify_reuse(self.run)

    def test_runtime_change_and_missing_provenance_are_rejected(self):
        self.manager().reuse(self.name, self.run / self.relative, [])
        self.worker.write_text('def changed():\n    return 2\n')
        with tarfile.open(str(self.run / 'source.tar.gz'), 'w:gz') as archive:
            archive.add(str(self.worker), arcname='scripts/p5_worker.py')
        with self.assertRaisesRegex(ValueError, 'implementation differs'):
            p5_reuse.verify_reuse(self.run)
        (self.run / 'reuse-manifest.json').unlink()
        with self.assertRaisesRegex(ValueError, 'lack provenance'):
            p5_reuse.verify_reuse(self.run)


if __name__ == '__main__':
    unittest.main()
