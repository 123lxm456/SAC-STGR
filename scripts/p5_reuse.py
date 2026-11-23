"""Task-level P5 continuation: copy verified completed jobs, never resume SGD.

Original evidence is read-only. Missing command proof (including old S0) means
that task is rerun from scratch in the new run. Raw training streams are linked
on Linux, not copied or deleted. Default CLI only inspects the proposed reuse.
"""

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys
import tarfile
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src/sac_rwb'))
from accept_p5 import read, protocol_hash
from algos.p5_experiment import make_directory


def require(value, message):
    if not value:
        raise ValueError(message)


def digest(path):
    value = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            value.update(block)
    return value.hexdigest()


def runtime_signature(archive_path):
    with tarfile.open(str(archive_path), 'r:gz') as archive:
        return {member.name: hashlib.sha256(archive.extractfile(member).read().replace(b'\r\n', b'\n')).hexdigest()
                for member in archive if member.isfile() and
                (member.name.startswith('src/') or member.name == 'scripts/p5_worker.py')}


def link_journal(source, target):
    require(os.name != 'nt', 'Raw-journal linking requires original Linux environment')
    target.symlink_to(source.resolve())


def completed_jobs(report):
    jobs = {}
    for row in report['online']:
        if row.get('execution_confirmed'):
            jobs['train-{}-{}'.format(row['variant'], row['seed'])] = 'online/{}/{}'.format(row['variant'], row['seed'])
    for row in report['offline']:
        if row.get('execution_confirmed'):
            jobs['offline-{}-{}'.format(row['variant'], row['seed'])] = 'offline/{}/{}'.format(row['variant'], row['seed'])
    for row in report['closed_loop']:
        # Recompute evaluation when its original actor training is not reused.
        dependency = 'train-{}-{}'.format(row['variant'], row['seed'])
        if row.get('execution_confirmed') and dependency in jobs:
            jobs['eval-{}-{}-{}-{}'.format(row['variant'], row['seed'], row['density'], row['evaluation_seed'])] = (
                'closed-loop/{}/{}/{}/{}'.format(row['variant'], row['seed'], row['density'], row['evaluation_seed']))
    return jobs


def inspect(origin):
    from review_p5_progress import review
    origin = Path(origin).resolve()
    origin.relative_to((ROOT / 'plan/p5/runs').resolve())
    report = review(origin)
    plan = read(origin / 'plan.json')
    require(plan['protocol']['research_run'], 'Only formal research tasks can be reused')
    require(not report['local_line_ending_copies'], 'Reuse on original Linux bytes, not converted Git copies')
    with tarfile.open(str(origin / 'source.tar.gz'), 'r:gz') as archive:
        saved = archive.extractfile('scripts/p5_worker.py').read()
        require(saved.replace(b'\r\n', b'\n') == (ROOT / 'scripts/p5_worker.py').read_bytes().replace(b'\r\n', b'\n'),
                'Worker changed; cannot mix prior and new training')
    verify_reuse(origin)
    jobs = completed_jobs(report)
    expected = report['expected']
    return dict(origin=str(origin), protocol_sha256=plan['protocol_sha256'],
                common_corpus_sha256=plan['common_corpus_sha256'], jobs=jobs,
                pending_proofs=report['pending_command_proofs'],
                reused_online=sum(n.startswith('train-') for n in jobs),
                remaining_online=expected['online'] - sum(n.startswith('train-') for n in jobs),
                remaining_offline=expected['offline'] - sum(n.startswith('offline-') for n in jobs),
                remaining_closed_loop=expected['closed_loop'] - sum(n.startswith('eval-') for n in jobs),
                raw_journals_present_for_reused_online=all(row['raw_online_streams_present'] for row in report['online']
                    if 'train-{}-{}'.format(row['variant'], row['seed']) in jobs),
                scope='Verified complete tasks only. Missing-proof tasks rerun from scratch. No optimizer/replay resumption.')


def verify_reuse(run, visited=None):
    """Validate provenance whenever an acceptance command was inherited."""
    run = Path(run).resolve()
    visited = set() if visited is None else set(visited)
    require(str(run) not in visited, 'Cyclic continuation provenance')
    visited.add(str(run))
    commands = read(run / 'commands.json')
    inherited = {c['name']: c for c in commands if c.get('reused_from')}
    path = run / 'reuse-manifest.json'
    if not path.is_file():
        require(not inherited, 'Inherited commands lack provenance')
        return
    run.relative_to((ROOT / 'plan/p5/runs').resolve())
    manifest = read(path)
    require(manifest['protocol_sha256'] == read(run / 'plan.json')['protocol_sha256'], 'Reuse protocol mismatch')
    # Extra prepared records after an interrupted copy are not accepted tasks.
    for name, command in inherited.items():
        record = manifest['jobs'][name]
        origin = Path(record['origin']).resolve()
        origin.relative_to((ROOT / 'plan/p5/runs').resolve())
        require(command['reused_from'] == str(origin) and origin != run, 'Reuse origin mismatch')
        origin_plan, current_plan = read(origin / 'plan.json'), read(run / 'plan.json')
        require(origin_plan['protocol_sha256'] == current_plan['protocol_sha256'] and
                origin_plan['common_corpus_sha256'] == current_plan['common_corpus_sha256'], 'Mixed reuse protocol/data')
        original = [c for c in read(origin / 'commands.json') if c['name'] == name]
        require(len(original) == 1 and original[0] == record['original_command'] and original[0]['exitcode'] == 0 and
                int((origin / (name + '.exitcode')).read_text()) == 0, 'Original command was not confirmed successful')
        require(command['argv'] == original[0]['argv'] and command['exitcode'] == 0 and
                (run / (name + '.log')).read_bytes() == (origin / (name + '.log')).read_bytes(), 'Inherited command/log mismatch')
        require(digest(origin / 'source.tar.gz') == record['origin_source_sha256'], 'Original source archive changed')
        require(runtime_signature(origin / 'source.tar.gz') == runtime_signature(run / 'source.tar.gz'),
                'Reused model/worker implementation differs from current execution snapshot')
        for entry in record['files']:
            target = run / entry['path']
            target.resolve().relative_to(ROOT.resolve())
            require(target.stat().st_size == entry['bytes'] and digest(target) == entry['sha256'],
                    'Inherited artifact changed: ' + entry['path'])
    for origin in {Path(manifest['jobs'][n]['origin']).resolve() for n in inherited}:
        verify_reuse(origin, visited)


class ReuseManager:
    def __init__(self, origin, run, plan):
        self.origin, self.run = Path(origin).resolve(), Path(run).resolve()
        self.proposal = inspect(self.origin)
        old = read(self.origin / 'plan.json')
        require(old['protocol'] == plan['protocol'] and old['common_corpus_sha256'] == plan['common_corpus_sha256'],
                'Continuation changed locked budget or common data')
        for key in plan['scenarios']:
            require(old['scenarios'][key]['scenario_id'] == plan['scenarios'][key]['scenario_id'] and
                    old['scenarios'][key]['route_hashes'] == plan['scenarios'][key]['route_hashes'],
                    'Continuation demand scene changed')
        self.commands = {c['name']: c for c in read(self.origin / 'commands.json')}
        self.manifest = dict(protocol_sha256=plan['protocol_sha256'], jobs={},
                             semantics='Copied complete tasks, original actual argv/log/exitcodes retained; no SGD resumption')
        self._save()

    def _save(self):
        (self.run / 'reuse-manifest.json').write_text(json.dumps(self.manifest, indent=2), encoding='utf-8')

    def reuse(self, name, destination, commands):
        if name not in self.proposal['jobs']:
            return False
        print('Validating reusable task: ' + name, flush=True)
        relative = self.proposal['jobs'][name]
        source = self.origin / relative
        destination = Path(destination)
        require(destination.relative_to(self.run).as_posix() == relative, 'Wrong reuse destination')
        files = []
        # Complete raw streams are needed on server; never silently reuse a
        # review-only copy with its online-data journals omitted.
        if name.startswith('train-'):
            require(all((source / 'online-data' / file).is_file() for file in ('samples.jsonl', 'events.jsonl')),
                    'Raw training journals absent; use the original experiment server')
        inventory = sorted(source.rglob('*'))
        require(not any(p.is_dir() and p.is_symlink() for p in inventory), 'Directory symlinks are not accepted')
        for item in inventory:
            if item.is_file():
                item.resolve().relative_to(ROOT.resolve())
                files.append(dict(path=(destination / item.relative_to(source)).relative_to(self.run).as_posix(),
                                  bytes=item.stat().st_size, sha256=digest(item)))
        make_directory(destination, new=True)
        for item in inventory:
            if not item.is_file():
                continue
            target = destination / item.relative_to(source)
            make_directory(target.parent)
            if item.name in ('samples.jsonl', 'events.jsonl') and 'online-data' in item.parts:
                link_journal(item, target)
            else:
                shutil.copy2(str(item), str(target))
                expected = next(row for row in files if row['path'] == target.relative_to(self.run).as_posix())
                require(digest(target) == expected['sha256'], 'Copied artifact mismatch')
        for suffix in ('.log', '.exitcode'):
            shutil.copy2(str(self.origin / (name + suffix)), str(self.run / (name + suffix)))
        self.manifest['jobs'][name] = dict(origin=str(self.origin), original_command=self.commands[name],
                                           origin_source_sha256=digest(self.origin / 'source.tar.gz'), files=files)
        self._save()
        inherited = dict(self.commands[name], reused_from=str(self.origin))
        commands.append(inherited)
        (self.run / 'commands.json').write_text(json.dumps(commands, indent=2), encoding='utf-8')
        print('Reused verified task: ' + name, flush=True)
        return True


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('origin', type=Path)
    args = parser.parse_args()
    print(json.dumps(inspect(args.origin), ensure_ascii=False, indent=2))
