"""Remove only superseded failed 100-decision smoke weights/predictions.

Default is a dry run. --apply keeps logs, metadata, data and every successful
or research run. Fixed historical run IDs cannot expand to formal training.
"""

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OBSOLETE_RUNS = ('20261122T123630783825Z', '20261122T132941429168Z')
SUFFIXES = ('.pth', '.pt', '.npz', '.pkl')


def candidates(root):
    root = Path(root).resolve()
    result = []
    for name in OBSOLETE_RUNS:
        run = root / 'plan/p5/runs' / name
        run.resolve().relative_to(root)
        summary = json.loads((run / 'summary.json').read_text(encoding='utf-8-sig'))
        protocol = json.loads((run / 'plan.json').read_text(encoding='utf-8-sig'))['protocol']
        if not (summary.get('mode') == 'smoke' and summary.get('status') == 'failed' and
                not summary.get('passed') and protocol.get('research_run') is False and
                protocol['online']['decision_steps'] == 100):
            raise ValueError('Refusing to clean non-obsolete run: ' + name)
        for path in sorted(run.rglob('*')):
            if path.suffix not in SUFFIXES or not path.is_file():
                continue
            resolved = path.resolve()
            resolved.relative_to(run.resolve())
            data = path.read_bytes()
            result.append(dict(path=path.relative_to(root).as_posix(), bytes=len(data),
                               sha256=hashlib.sha256(data).hexdigest()))
    return result


def clean(root, apply=False):
    root = Path(root).resolve()
    files = candidates(root)
    report = dict(applied=False, files=files, file_count=len(files),
                  total_bytes=sum(item['bytes'] for item in files),
                  preserved='All research/successful smoke runs, failure logs, source snapshots, metadata and raw data; Git history unchanged')
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
