"""Remove only retired C1/C2/C3 evaluation work logs; keep run evidence."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re

ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT / 'plan/p5/runs/20261123T094822803672Z'
PATTERN = re.compile(r'eval-work-c[123]-(11|23|37)-(low|nominal|high)-(3000|3001|3002)')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--apply', action='store_true')
    args = parser.parse_args()
    RUN.resolve().relative_to(ROOT.resolve())
    targets, files = [], []
    for directory in sorted(RUN.iterdir()):
        if not PATTERN.fullmatch(directory.name):
            continue
        if directory.is_symlink() or not directory.is_dir():
            raise ValueError('Unexpected target: ' + str(directory))
        directory.resolve().relative_to(RUN.resolve())
        children = list(directory.iterdir())
        if any(p.is_symlink() or not p.is_file() or p.name != 'sumolog' for p in children):
            raise ValueError('Unexpected work content: ' + str(directory))
        targets.append(directory)
        for path in children:
            data = path.read_bytes()
            files.append(dict(path=path.relative_to(ROOT).as_posix(), bytes=len(data),
                              sha256=hashlib.sha256(data).hexdigest()))
    report = dict(applied=False, directories=[p.relative_to(ROOT).as_posix() for p in targets],
                  files=files, logical_bytes=sum(p['bytes'] for p in files))
    if args.apply:
        stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
        receipt = ROOT / ('plan/p5/evidence/retired-eval-work-' + stamp + '.json')
        # Write inventory before mutation; never delete an unexpected child.
        receipt.write_text(json.dumps(report, indent=2), encoding='utf-8')
        for directory in targets:
            for path in directory.iterdir():
                if path.is_symlink() or not path.is_file() or path.name != 'sumolog':
                    raise ValueError('Work content changed during cleanup')
                path.unlink()
            directory.rmdir()
        report['applied'] = True
        receipt.write_text(json.dumps(report, indent=2), encoding='utf-8')
        print('Receipt: ' + str(receipt))
    print(json.dumps(dict(applied=report['applied'], directories=len(targets),
                          files=len(files), logical_bytes=report['logical_bytes'])))


if __name__ == '__main__':
    main()
