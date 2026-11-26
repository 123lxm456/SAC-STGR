"""Inventory P5 disk usage; delete only dedicated retired experiment products.

Standard library only, Python 3.7+. Mixed provenance, retained models and raw
streams are never rewritten. Default is read-only. Apply requires a saved scan
and refuses active experiment processes, changed inventories and inbound links.
"""
import argparse
from collections import defaultdict
import hashlib
import json
import os
from pathlib import Path
import re
import stat

RETIRED = {'c1', 'c2', 'c3'}  # Deletion policy only; never experiment options.
ROOT = Path(__file__).resolve().parents[1]


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def candidate(relative):
    p = Path(relative).parts
    if len(p) < 3 or p[0] != 'runs':
        return False
    if p[2] in ('online', 'offline', 'closed-loop'):
        return len(p) >= 4 and p[3] in RETIRED
    if re.fullmatch(r'work-(c[123])-\d+', p[2]):
        return True
    return len(p) == 3 and bool(re.fullmatch(
        r'(train|offline|eval)-c[123]-\d+(?:-[a-z]+-\d+)?\.(log|exitcode)', p[2]))


def active_processes(project):
    found = []
    proc = Path('/proc')
    if os.name != 'posix' or not proc.is_dir():
        return found
    for folder in proc.glob('[0-9]*'):
        try:
            argv = (folder / 'cmdline').read_bytes().decode(errors='replace').split('\0')
            cwd = (folder / 'cwd').resolve()
            if any(Path(a).name in ('accept_p5.py', 'p5_worker.py') for a in argv if a) and (
                    cwd == project or any(str(project) in a for a in argv)):
                found.append(int(folder.name))
        except (OSError, ValueError):
            continue
    return found


def inventory(root):
    root = Path(root).resolve()
    files, directories, totals, links = [], [], defaultdict(lambda: [0, 0]), []
    seen = set()
    for parent, dirs, names in os.walk(str(root), followlinks=False):
        parent = Path(parent)
        for name in list(dirs):
            path = parent / name
            if path.is_symlink():
                dirs.remove(name)
                names.append(name)
            else:
                directories.append(path.relative_to(root).as_posix())
        for name in sorted(names):
            path = parent / name
            rel = path.relative_to(root).as_posix()
            s = path.lstat()
            if not (stat.S_ISREG(s.st_mode) or stat.S_ISLNK(s.st_mode)):
                raise ValueError('Unsupported file type: ' + rel)
            link = os.readlink(str(path)) if path.is_symlink() else None
            allocated = getattr(s, 'st_blocks', 0) * 512 if hasattr(s, 'st_blocks') else s.st_size
            inode = (s.st_dev, s.st_ino)
            exclusive = allocated if inode not in seen else 0
            seen.add(inode)
            row = dict(path=rel, bytes=s.st_size, allocated_bytes=exclusive,
                       mtime_ns=s.st_mtime_ns, inode=s.st_ino, device=s.st_dev,
                       nlink=s.st_nlink, link=link, delete=candidate(rel))
            if row['delete'] and link is None:
                row['sha256'] = sha(path)
                if path.name == 'result.json':
                    value = json.loads(path.read_text(encoding='utf-8-sig'))
                    if value.get('variant') not in RETIRED:
                        raise ValueError('Unexpected result identity: ' + rel)
            files.append(row)
            if link is not None:
                links.append((rel, str(path.resolve())))
            for depth in range(min(5, len(Path(rel).parts))):
                key = '/'.join(Path(rel).parts[:depth]) or '.'
                totals[key][0] += s.st_size
                totals[key][1] += exclusive
    selected = {r['path'] for r in files if r['delete']}
    blocked = []
    for source, target in links:
        try:
            rel = Path(target).relative_to(root).as_posix()
        except ValueError:
            continue
        if source not in selected and (rel in selected or any(p.startswith(rel + '/') for p in selected)):
            blocked.append(dict(link=source, target=rel))
    large = sorted((r for r in files if not r['delete']), key=lambda r: r['bytes'], reverse=True)[:40]
    return dict(root=str(root), files=sorted(files, key=lambda r: r['path']),
                directories=sorted(directories), blocked_links=blocked,
                totals={k: dict(logical_bytes=v[0], allocated_bytes=v[1]) for k, v in sorted(totals.items())},
                largest_preserved=large, candidate_count=len(selected),
                candidate_logical_bytes=sum(r['bytes'] for r in files if r['delete']),
                candidate_allocated_bytes=sum(r['allocated_bytes'] for r in files if r['delete']))


def save(path, value):
    path = Path(path)
    with path.open('x', encoding='utf-8', newline='\n') as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2)


def apply_scan(root, saved, receipt):
    root = Path(root).resolve()
    if os.name != 'posix':
        raise ValueError('Apply on original Linux server only')
    if active_processes(root.parent.parent):
        raise ValueError('Experiment is active; no deletion performed')
    # Global runner lock prevents concurrent start after the process check.
    import fcntl
    with (root / '.experiment.lock').open('a+') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if active_processes(root.parent.parent):
            raise ValueError('Experiment is active; no deletion performed')
        current = inventory(root)
        if current['root'] != saved['root'] or current['blocked_links']:
            raise ValueError('Root changed or preserved files depend on deletion candidates')
        old = [r for r in saved['files'] if r['delete']]
        new = [r for r in current['files'] if r['delete']]
        # Never trust the scan to expand the deletion policy.
        if old != new or any(not candidate(r['path']) for r in old):
            raise ValueError('Candidate inventory changed; rescan before deletion')
        receipt = Path(receipt)
        save(receipt, dict(status='started', root=str(root), files=old))
        before = os.statvfs(str(root))
        for row in old:
            path = root / row['path']
            path.parent.resolve().relative_to(root)
            path.unlink()  # Removes links themselves, never their targets.
        for rel in sorted(current['directories'], key=lambda p: len(Path(p).parts), reverse=True):
            if candidate(rel):
                path = root / rel
                try:
                    path.rmdir()
                except OSError:
                    pass
        after = os.statvfs(str(root))
        result = dict(status='completed', root=str(root), files=old,
                      deleted_logical_bytes=sum(r['bytes'] for r in old),
                      filesystem_free_delta=(after.f_bfree-before.f_bfree)*after.f_frsize,
                      note='Free-space delta can include concurrent filesystem activity; hardlinks may retain blocks.')
        receipt.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding='utf-8')
        return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root', type=Path, default=ROOT / 'plan/p5')
    p.add_argument('--scan', type=Path)
    p.add_argument('--apply', type=Path, help='Apply a previously saved scan')
    p.add_argument('--receipt', type=Path)
    args = p.parse_args()
    root = args.root.resolve()
    allowed = {str((ROOT / 'plan' / name).resolve()) for name in ('p5', 'P5')}
    if str(root) not in allowed or not root.is_dir():
        p.error('Root must be existing project plan/p5 or plan/P5')
    if args.apply:
        if not args.receipt:
            p.error('--receipt required with --apply')
        report = apply_scan(root, json.loads(args.apply.read_text(encoding='utf-8')), args.receipt)
        print(json.dumps({k:v for k,v in report.items() if k != 'files'}, indent=2))
    else:
        if not args.scan:
            p.error('--scan required for read-only inventory')
        report = inventory(root)
        report['active_processes'] = active_processes(ROOT)
        save(args.scan, report)
        print(json.dumps(dict(root=report['root'], active_processes=report['active_processes'],
                              blocked_links=report['blocked_links'],
                              candidate_count=report['candidate_count'],
                              candidate_logical_bytes=report['candidate_logical_bytes'],
                              candidate_allocated_bytes=report['candidate_allocated_bytes'],
                              total=report['totals'].get('.'),
                              largest_preserved=report['largest_preserved'][:10]), indent=2))


if __name__ == '__main__':
    main()
