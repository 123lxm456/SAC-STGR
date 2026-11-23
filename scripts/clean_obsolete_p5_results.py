"""Remove fixed, retired local snapshots and failed-smoke artifacts.

Default is a dry run. --apply keeps logs, metadata and every successful
or research run. Fixed historical run IDs cannot expand to formal training.
"""

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OBSOLETE_RUNS = ('20261122T123630783825Z', '20261122T132941429168Z')
LOCAL_RUNS = ('20261122T091531727475Z', '20261122T092014428044Z',
              '20261122T092249404846Z', '20261122T140654903738Z')
SUFFIXES = ('.pth', '.pt', '.npz', '.pkl')


def candidates(root):
    root = Path(root).resolve()
    result = []
    present = {path.name: path for path in (root / 'plan/p5/runs').iterdir() if path.is_dir()}
    for name in OBSOLETE_RUNS + LOCAL_RUNS:
        if name not in present:
            continue
        run = present[name]
        run.resolve().relative_to(root)
        summary = json.loads((run / 'summary.json').read_text(encoding='utf-8-sig'))
        if name in LOCAL_RUNS:
            if summary.get('mode') != 'local':
                raise ValueError('Refusing to clean repurposed local run: ' + name)
        else:
            protocol = json.loads((run / 'plan.json').read_text(encoding='utf-8-sig'))['protocol']
            if not (summary.get('mode') == 'smoke' and summary.get('status') == 'failed' and
                not summary.get('passed') and protocol.get('research_run') is False and
                protocol['online']['decision_steps'] == 100):
                raise ValueError('Refusing to clean non-obsolete run: ' + name)
        for path in sorted(run.rglob('*')):
            snapshot = path.name == 'source.tar.gz'
            failed_result = name in OBSOLETE_RUNS and (path.suffix in SUFFIXES or path.name == 'physical.jsonl')
            if not (snapshot or failed_result) or not path.is_file():
                continue
            resolved = path.resolve()
            resolved.relative_to(run.resolve())
            data = path.read_bytes()
            if snapshot:
                expected = json.loads((run / 'source-snapshot.json').read_text(encoding='utf-8-sig'))['sha256']
                if hashlib.sha256(data).hexdigest() != expected:
                    raise ValueError('Retired source snapshot has changed: ' + name)
            elif path.name == 'physical.jsonl':
                expected = json.loads(path.with_name('result.json').read_text(encoding='utf-8-sig'))['physical_sha256']
                if expected not in (hashlib.sha256(data).hexdigest(), hashlib.sha256(data.replace(b'\r\n', b'\n')).hexdigest()):
                    raise ValueError('Retired physical log has changed: ' + name)
            result.append(dict(path=path.relative_to(root).as_posix(), bytes=len(data),
                               sha256=hashlib.sha256(data).hexdigest()))
    return result


def clean(root, apply=False):
    root = Path(root).resolve()
    files = candidates(root)
    report = dict(applied=False, files=files, file_count=len(files),
                  total_bytes=sum(item['bytes'] for item in files),
                  preserved='All research/successful smoke runs, failure logs, metadata and original training journals; Git history unchanged')
    if apply:
        path = root / 'plan/p5' / ('obsolete-result-cleanup-' + datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ') + '.json')
        with path.open('x', encoding='utf-8') as stream:
            json.dump(report, stream, ensure_ascii=False, indent=2)
        for item in files:
            target = (root / item['path']).resolve()
            target.relative_to(root)
            if hashlib.sha256(target.read_bytes()).hexdigest() != item['sha256']:
                raise ValueError('Candidate changed since inspection: ' + item['path'])
            target.unlink()
        report['applied'] = True
        path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
        report['manifest'] = str(path)
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--apply', action='store_true')
    args = parser.parse_args()
    report = clean(ROOT, args.apply)
    print(json.dumps({key: value for key, value in report.items() if key != 'files'}, ensure_ascii=False, indent=2))
