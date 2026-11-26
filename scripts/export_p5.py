"""Export review evidence; large online/common-corpus JSONL remains on server."""

import argparse
from datetime import datetime, timezone
import hashlib
from pathlib import Path
import tarfile
from p5_storage import relocated


def export(run, destination):
    run = relocated(run).resolve()
    destination = Path(destination)
    destination.mkdir(exist_ok=True, parents=True)
    archive = destination / ('p5-review-' + run.name + '-' + datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ') + '.tar.gz')
    with tarfile.open(str(archive), 'x:gz') as stream:
        for path in sorted(run.rglob('*')):
            if (not path.is_file() or path.name in ('samples.jsonl', 'events.jsonl') or
                    path.relative_to(run).parts[:2] == ('data', 'objects')):
                continue
            stream.add(str(path), arcname=run.name + '/' + path.relative_to(run).as_posix(), recursive=False)
    with archive.open('rb') as stream:
        digest = hashlib.sha256()
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    archive.with_suffix(archive.suffix + '.sha256').write_bytes((digest.hexdigest() + '  ' + archive.name + '\n').encode('utf-8'))
    print(str(archive))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('run', type=Path)
    parser.add_argument('--destination', type=Path, default=Path('plan/p5/returns'))
    args = parser.parse_args()
    export(args.run, args.destination)
