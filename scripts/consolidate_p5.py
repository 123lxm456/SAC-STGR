"""Prepare, verify, then finalize a lossless single-directory P5 migration.

Linux server only. Preparation leaves every original file intact. Finalization
requires a fresh target regression, an equal full audit, and checked inventories.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

from p5_storage import PRIMARY, INDEX, safe_path, journal_digest_size
from stage_p5_storage import check_gzip

ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT / 'plan/p5/runs' / PRIMARY
OLD = ('20261122T141242292188Z', '20261122T144249871603Z')
STORE = ROOT / 'plan/p5/storage/primary-20261123T094822803672Z'
CONTROL = RUN / 'migration'


def require(value, message):
    if not value:
        raise ValueError(message)


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def ensure_directory(path):
    current = Path(path.anchor)
    for part in path.parts[1:]:
        current = current / part
        if not current.exists():
            current.mkdir()


def write(path, data):
    with path.open('x', encoding='utf-8') as f:
        json.dump(data, f, indent=2)


def write_once_equal(path, data):
    if path.exists():
        require(read(path) == data, 'Previously staged metadata differs: ' + str(path))
    else:
        write(path, data)


def snapshot(path, digest=None):
    s = path.lstat()
    return dict(path=path.relative_to(ROOT).as_posix(), size=s.st_size,
                device=s.st_dev, inode=s.st_ino, mtime_ns=s.st_mtime_ns,
                link=os.readlink(str(path)) if path.is_symlink() else None,
                sha256=digest or sha(path))


def unchanged(row):
    path = safe_path(ROOT, row['path'])
    current = snapshot(path)
    require(all(current[key] == value for key, value in row.items()),
            'Source changed since preparation: ' + str(path))
    return current


def regular_files(directory):
    for parent, dirs, names in os.walk(str(directory), followlinks=False):
        require(not any((Path(parent) / d).is_symlink() for d in dirs), 'Directory link not allowed')
        for name in sorted(names):
            yield Path(parent) / name


def work_directory(directory):
    return directory.name.startswith(('work-', 'eval-work-'))


def copy_verified(source, target, dedup):
    ensure_directory(target.parent)
    digest = sha(source)
    if target.exists():
        require(sha(target) == digest, 'Existing destination differs: ' + str(target))
    else:
        # Only immutable evidence files participate; Linux hard links avoid
        # duplicated model/scenario bytes. The original path remains intact.
        os.link(str(dedup.get(digest, source)), str(target))
    require(sha(target) == digest, 'Destination hash differs')
    dedup.setdefault(digest, target)
    return dict(path=target.relative_to(ROOT).as_posix(), sha256=digest)


def code_signature():
    return {p.relative_to(ROOT).as_posix(): hashlib.sha256(p.read_bytes().replace(b'\r\n', b'\n')).hexdigest()
            for folder in ('scripts', 'src', 'tests') for p in (ROOT / folder).rglob('*.py')
            if '__pycache__' not in p.parts}


def storage_usage():
    inodes = {}
    for directory in (ROOT / 'plan/p5/runs', ROOT / 'plan/p5/storage'):
        if directory.exists():
            for p in regular_files(directory):
                s = p.lstat()
                inodes.setdefault((s.st_dev, s.st_ino), s.st_blocks * 512)
    return sum(inodes.values())


def no_workers():
    require(os.name == 'posix', 'Run only on original Linux experiment server')
    for proc in Path('/proc').iterdir():
        if proc.name.isdigit() and int(proc.name) != os.getpid():
            try:
                argv = (proc / 'cmdline').read_bytes().split(b'\0')
            except (OSError, PermissionError):
                continue
            forbidden = {str(ROOT / 'scripts' / n).encode() for n in
                         ('accept_p5.py', 'p5_worker.py', 'stage_p5_storage.py', 'consolidate_p5.py')}
            require(not any(a in forbidden or a in {b'scripts/' + Path(x.decode()).name.encode() for x in forbidden}
                            for a in argv), 'An experiment/storage worker is still active: ' + proc.name)


def prepare(stage):
    require(not (ROOT / INDEX).exists(), 'Migration already prepared; use --verify or --finalize')
    require({p.name for p in (ROOT / 'plan/p5/runs').iterdir() if p.is_dir()} == set((*OLD, PRIMARY)),
            'Unexpected run directories; inspect before consolidation')
    report = read(stage)
    require(report.get('completed') and report.get('sources_unchanged'), 'Incomplete compression stage')
    # Freeze a fresh reference audit before enabling compressed reads.
    from audit_p5 import audit
    print('Computing original-layout full audit ...', flush=True)
    baseline = audit(RUN)
    require(baseline['primary_matrix_integrity_passed'] and not baseline['missing_natural_risk_updates'],
            'Original matrix is not intact')
    CONTROL.mkdir(exist_ok=True)
    write_once_equal(CONTROL / 'baseline-audit.json', baseline)
    journals, objects, sources = {}, {}, []
    for entry in report['objects']:
        require(entry['fully_decompressed_verified'] is True and
                entry['object'] == entry['sha256'] + '.jsonl.gz', 'Invalid staged object')
        source = safe_path(STORE / 'objects', entry['object'])
        require(source.stat().st_size == entry['gzip_bytes'], 'Changed staged object size')
        if entry['object'] not in objects:
            print('Verifying staged object: ' + entry['object'], flush=True)
            check_gzip(source, entry['sha256'], entry['bytes'])
            target = RUN / 'data/objects' / entry['object']
            ensure_directory(target.parent)
            if target.exists():
                require(sha(target) == sha(source), 'Existing object differs')
            else:
                os.link(str(source), str(target))
            objects[entry['object']] = dict(path=target.relative_to(ROOT).as_posix(),
                                            gzip_sha256=sha(target), bytes=entry['bytes'], sha256=entry['sha256'])
        for alias in entry['aliases']:
            path = safe_path(ROOT, alias)
            require(alias.startswith('plan/p5/runs/') and Path(alias).parts[3] in (*OLD, PRIMARY) and
                    path.name in ('samples.jsonl', 'events.jsonl') and 'online-data' in path.parts,
                    'Unexpected staged alias')
            require(path.is_file(), 'Original journal missing')
            print('Checking original: ' + alias, flush=True)
            require(path.stat().st_size == entry['bytes'] and sha(path) == entry['sha256'], 'Original no longer matches gzip')
            require(alias not in journals, 'Duplicate journal alias')
            journals[alias] = dict(object=objects[entry['object']]['path'], bytes=entry['bytes'], sha256=entry['sha256'])
            sources.append(snapshot(path, entry['sha256']))
    current = {p.relative_to(ROOT).as_posix() for name in (*OLD, PRIMARY)
               for p in (ROOT / 'plan/p5/runs' / name).glob('online/*/*/online-data/*.jsonl')
               if p.name in ('samples.jsonl', 'events.jsonl')}
    require(current == set(journals), 'Stage does not cover all original journals')
    dedup, retained, deleted = {}, [], []
    # Use existing primary files as de-duplication targets without altering them.
    for p in regular_files(RUN):
        rel = p.relative_to(RUN)
        if p.is_symlink() or rel.parts[0] in ('data', 'migration', 'provenance') or work_directory(RUN / rel.parts[0]):
            continue
        if p.relative_to(ROOT).as_posix() in journals:
            continue
        digest = sha(p)
        dedup.setdefault(digest, p)
        retained.append(dict(path=p.relative_to(ROOT).as_posix(), sha256=digest))
    relocations = {}
    for name in OLD:
        old = ROOT / 'plan/p5/runs' / name
        target = RUN / 'provenance' / name
        relocations[old.relative_to(ROOT).as_posix()] = target.relative_to(ROOT).as_posix()
        for p in regular_files(old):
            rel = p.relative_to(old)
            if p.relative_to(ROOT).as_posix() in journals:
                continue
            require(not p.is_symlink(), 'Unexpected non-journal symbolic link')
            if work_directory(old / rel.parts[0]):
                require(len(rel.parts) == 2 and rel.name == 'sumolog', 'Unknown work files; inspect manually')
            else:
                retained.append(copy_verified(p, target / rel, dedup))
            deleted.append(snapshot(p))
    for child in RUN.iterdir():
        if child.is_dir() and work_directory(child):
            for p in regular_files(child):
                require(p.name == 'sumolog' and p.parent == child and not p.is_symlink(), 'Unknown primary work content')
                deleted.append(snapshot(p))
    # Preserve staging receipts inside the unified run before retiring storage.
    for p in STORE.iterdir():
        if p.is_file():
            retained.append(copy_verified(p, RUN / 'provenance/storage-stage' / p.name, dedup))
            deleted.append(snapshot(p))
    for p in regular_files(STORE / 'objects'):
        require(p.name in objects and not p.is_symlink(), 'Unrecognized staged object; inspect manually')
        deleted.append(snapshot(p, objects[p.name]['gzip_sha256']))
    data = dict(version='p5-storage-v1', original_root=str(ROOT), relocations=relocations, journals=journals)
    write_once_equal(CONTROL / 'plan.json', dict(stage=str(stage), objects=list(objects.values()),
                                    retained=retained, remove=sources + deleted,
                                    journals=journals, relocations=relocations,
                                    retired_runs=list(OLD), code_signature=code_signature()))
    write(ROOT / INDEX, data)
    print('PREPARED: all original files remain; compressed reads are now enabled', flush=True)


def check_migration(allow_code_change=False):
    plan = read(CONTROL / 'plan.json')
    if not allow_code_change:
        require(code_signature() == plan['code_signature'], 'Migration code changed; prepare/verify a reviewed revision')
    for row in plan['retained']:
        require(sha(safe_path(ROOT, row['path'])) == row['sha256'], 'Retained evidence changed: ' + row['path'])
    for row in plan['objects']:
        p = safe_path(ROOT, row['path'])
        require(sha(p) == row['gzip_sha256'], 'Compressed object changed')
    # Exercise the public reader for every original alias, including all three
    # reuse links and S0 hard links; do not trust manifest hashes alone.
    data = read(ROOT / INDEX)
    require(data['journals'] == plan['journals'] and data['relocations'] == plan['relocations'],
            'Storage mapping changed after preparation')
    checked = set()
    for alias, row in data['journals'].items():
        if row['object'] not in checked:
            print('Verifying compressed reader {}/{}: {}'.format(len(checked) + 1, len(plan['objects']), alias), flush=True)
            require(journal_digest_size(ROOT / alias, ROOT) == (row['sha256'], row['bytes']), 'Reader round trip differs')
            checked.add(row['object'])
    from audit_p5 import audit
    print('Recomputing compressed-layout full audit ...', flush=True)
    result = audit(RUN)
    require(result == read(CONTROL / 'baseline-audit.json'), 'Metrics/evidence verdict changed after migration')
    return plan, result


def rebind():
    """Revalidate a prepared migration after a reviewed verifier repair.

    Only available before deletion. The old snapshots, original journal hashes,
    copied evidence and full audit must still agree before new proof is bound.
    """
    require((ROOT / INDEX).is_file() and (CONTROL / 'plan.json').is_file(), 'Migration is not prepared')
    require(not list(CONTROL.glob('deletion-*.json')), 'Deletion has started; do not rebind')
    plan, _ = check_migration(allow_code_change=True)
    for row in plan['remove']:
        current = unchanged(row)
        digest = current['sha256']
        if row['path'] in plan['journals']:
            require(digest == plan['journals'][row['path']]['sha256'], 'Original journal changed')
        else:
            for old, new in plan['relocations'].items():
                if row['path'].startswith(old + '/'):
                    copy = safe_path(ROOT, new + row['path'][len(old):])
                    if copy.is_file():
                        require(digest == sha(copy), 'Preserved evidence differs from original')
                    break
        row['sha256'] = digest
    plan['code_signature'] = code_signature()
    target = CONTROL / 'plan.json'
    temporary = CONTROL / 'plan.rebind.tmp'
    require(not temporary.exists(), 'Previous rebind temp file exists; inspect it')
    write(temporary, plan)
    os.replace(str(temporary), str(target))
    print('REBOUND: originals intact; rerun --verify before any deletion', flush=True)


def verify():
    plan, result = check_migration()
    stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
    logdir = CONTROL / ('verification-' + stamp)
    logdir.mkdir()
    for name, args in (
        ('unit', ['scripts/run_unit_checks.py', '--require-cuda', '--output', str(logdir / 'unit.json')]),
        ('static', ['scripts/verify_plan.py', '--report', str(logdir / 'static.json')]),
        ('p01234', ['scripts/verify_p01234.py', '--target']),
    ):
        print('Running target ' + name, flush=True)
        with (logdir / (name + '.log')).open('w') as output:
            code = subprocess.call([sys.executable, '-B'] + args, cwd=str(ROOT), stdout=output, stderr=subprocess.STDOUT)
        require(code == 0, name + ' failed; see ' + str(logdir))
    write(logdir / 'audit.json', result)
    write(logdir / 'passed.json', dict(passed=True, index_sha256=sha(ROOT / INDEX),
                                      plan_sha256=sha(CONTROL / 'plan.json'),
                                      code_signature=code_signature(), audit_equal=True))
    print('VERIFIED: ' + str(logdir), flush=True)


def finalize():
    proofs = sorted(CONTROL.glob('verification-*/passed.json'))
    require(proofs, 'No successful target migration verification')
    proof = read(proofs[-1])
    require(proof['passed'] and proof['audit_equal'] and proof['code_signature'] == code_signature() and
            proof['index_sha256'] == sha(ROOT / INDEX) and proof['plan_sha256'] == sha(CONTROL / 'plan.json'),
            'Verification no longer matches this migration')
    plan, _ = check_migration()
    # Validate every removal before deleting any source, including directory
    # coverage. Unexpected new files abort rather than recursively disappearing.
    removals = {row['path']: row for row in plan['remove']}
    roots = [ROOT / 'plan/p5/runs' / name for name in OLD] + [STORE]
    roots += [p for p in RUN.iterdir() if p.is_dir() and work_directory(p)]
    for directory in roots:
        directory.resolve().relative_to(ROOT.resolve())
        require(all(p.relative_to(ROOT).as_posix() in removals for p in regular_files(directory)),
                'Unplanned file in removal directory')
    for row in removals.values():
        unchanged(row)
    before = os.statvfs(str(ROOT))
    allocated_before = storage_usage()
    receipt = CONTROL / ('deletion-' + datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ') + '.json')
    write(receipt, dict(completed=False, remove=list(removals)))
    for row in removals.values():
        safe_path(ROOT, row['path']).unlink()
    for directory in roots:
        for parent, dirs, _ in os.walk(str(directory), topdown=False):
            for child in dirs:
                (Path(parent) / child).rmdir()
        directory.rmdir()
    from audit_p5 import audit
    result = audit(RUN)
    require(result == read(CONTROL / 'baseline-audit.json'), 'Post-removal audit differs; compressed objects remain recoverable')
    after = os.statvfs(str(ROOT))
    done = dict(completed=True, removed_paths=len(removals), removed_logical_bytes=sum(r['size'] for r in removals.values()),
                allocated_before=allocated_before, allocated_after=storage_usage(),
                free_space_delta=after.f_bavail * after.f_frsize - before.f_bavail * before.f_frsize,
                full_audit_unchanged=True, research_passed=result['passed'], research_status=result['status'])
    receipt.write_text(json.dumps(done, indent=2), encoding='utf-8')
    print('FINALIZED: ' + json.dumps(done), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--prepare', type=Path, metavar='STAGE_MANIFEST')
    group.add_argument('--verify', action='store_true')
    group.add_argument('--rebind', action='store_true')
    group.add_argument('--finalize', action='store_true')
    args = parser.parse_args()
    no_workers()
    # Serialize migration operations, including across SSH sessions.
    import fcntl
    lock = ROOT / 'plan/p5/.storage-migration.lock'
    with lock.open('a') as handle, (ROOT / 'plan/p5/.experiment.lock').open('a') as experiment:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        fcntl.flock(experiment.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        if args.prepare:
            prepare(args.prepare)
        elif args.verify:
            verify()
        elif args.rebind:
            rebind()
        else:
            finalize()
