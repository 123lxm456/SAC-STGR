"""Stage verified lossless P5 journals. Never remove or modify source runs."""
import argparse
from datetime import datetime, timezone
import gzip
import hashlib
import json
import os
from pathlib import Path
import tempfile

ROOT = Path(__file__).resolve().parents[1]
RUNS = {'20261122T141242292188Z', '20261122T144249871603Z', '20261123T094822803672Z'}


def fingerprint(path):
    s = path.stat()
    return (s.st_dev, s.st_ino, s.st_size, s.st_mtime_ns)


def verified_gzip(source, destination):
    """Stream once to gzip, then decompress fully and verify original bytes."""
    before = fingerprint(source)
    fd, name = tempfile.mkstemp(prefix='staging-', suffix='.tmp', dir=str(destination))
    tmp = Path(name)
    try:
        digest = hashlib.sha256()
        size = 0
        with os.fdopen(fd, 'wb') as output:
            with gzip.GzipFile(filename='', mode='wb', fileobj=output, compresslevel=6, mtime=0) as zipped:
                with source.open('rb') as stream:
                    for block in iter(lambda: stream.read(1024 * 1024), b''):
                        zipped.write(block)
                        digest.update(block)
                        size += len(block)
            output.flush()
            os.fsync(output.fileno())
        expected = digest.hexdigest()
        require(size == before[2] and fingerprint(source) == before, 'Source changed during staging')
        check_gzip(tmp, expected, size)
        target = destination / (expected + '.jsonl.gz')
        if target.exists():
            check_gzip(target, expected, size)
        else:
            # Atomic publication that refuses to replace an existing object.
            os.link(str(tmp), str(target))
        return dict(sha256=expected, bytes=size, gzip_bytes=target.stat().st_size,
                    object=target.name, fully_decompressed_verified=True)
    finally:
        if tmp.exists():
            tmp.unlink()


def require(value, message):
    if not value:
        raise ValueError(message)


def check_gzip(path, expected, size):
    digest = hashlib.sha256()
    actual_size = 0
    with gzip.open(str(path), 'rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
            actual_size += len(block)
    require(actual_size == size and digest.hexdigest() == expected, 'Decompressed hash/size mismatch: ' + str(path))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--inventory', type=Path, required=True)
    args = parser.parse_args()
    require(os.name == 'posix', 'Run staging on the original Linux server')
    base = ROOT / 'plan/p5/runs'
    inventory = json.loads(args.inventory.read_text())
    selected, groups = [], {}
    for row in inventory:
        relative = Path(row['path'])
        if relative.name not in ('samples.jsonl', 'events.jsonl') or 'online-data' not in relative.parts:
            continue
        path = ROOT / relative
        require(relative.parts[:3] == ('plan', 'p5', 'runs') and relative.parts[3] in RUNS,
                'Unexpected source run')
        path.resolve().relative_to(base.resolve())
        require(path.is_file(), 'Source absent: ' + str(path))
        if row['kind'] == 'file':
            require(fingerprint(path)[:3] == (row['device'], row['inode'], row['bytes']),
                    'Inventory no longer matches source: ' + str(path))
        else:
            require(path.is_symlink() and str(path.resolve()) == row['resolved'], 'Source link changed')
        key = fingerprint(path)[:2]
        groups.setdefault(key, []).append(path)
        selected.append(path)
    require(groups, 'No online journals found')
    current = {p for name in RUNS for p in (base / name).glob('online/*/*/online-data/*.jsonl')
               if p.name in ('samples.jsonl', 'events.jsonl')}
    require(set(selected) == current, 'Inventory does not cover all current online journals')
    store = ROOT / 'plan/p5/storage/primary-20261123T094822803672Z'
    objects = store / 'objects'
    objects.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
    manifest = store / ('stage-' + stamp + '.json')
    report = dict(completed=False, sources_unchanged=True, inventory=str(args.inventory), objects=[])
    free_before = os.statvfs(str(store)).f_bavail * os.statvfs(str(store)).f_frsize
    print('MANIFEST: ' + str(manifest), flush=True)
    for index, paths in enumerate(groups.values(), 1):
        print('Staging {}/{}: {}'.format(index, len(groups), paths[0].relative_to(ROOT)), flush=True)
        entry = verified_gzip(paths[0], objects)
        entry['aliases'] = [p.relative_to(ROOT).as_posix() for p in paths]
        report['objects'].append(entry)
        manifest.write_text(json.dumps(report, indent=2), encoding='utf-8')
        print('Verified {} -> {} bytes'.format(entry['bytes'], entry['gzip_bytes']), flush=True)
    report.update(completed=True, source_paths=len(selected), unique_source_inodes=len(groups),
                  original_bytes=sum(e['bytes'] for e in report['objects']),
                  gzip_bytes=sum(p.stat().st_size for p in objects.glob('*.jsonl.gz')),
                  filesystem_free_before=free_before,
                  filesystem_free_after=os.statvfs(str(store)).f_bavail * os.statvfs(str(store)).f_frsize)
    manifest.write_text(json.dumps(report, indent=2), encoding='utf-8')
    print('SUMMARY: ' + json.dumps({k: v for k, v in report.items() if k != 'objects'}), flush=True)


if __name__ == '__main__':
    main()
