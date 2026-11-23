"""Package active source/docs only. Excludes original references and run data."""

import argparse
import hashlib
import io
import json
from pathlib import Path
import tarfile
from accept_p012 import ROOT, source_manifest


def build(output):
    output = Path(output)
    output.parent.mkdir(exist_ok=True, parents=True)
    if output.exists() or output.with_suffix(output.suffix + '.sha256').exists():
        raise ValueError('Refusing to overwrite delivery artifact')
    files = {ROOT / entry['path'] for entry in source_manifest()}
    files.update(ROOT / name for name in ('README.md', 'research_implementation_plan.md', '.gitignore', '.gitattributes'))
    files.update((ROOT / 'chapters').glob('*.md'))
    files.update((ROOT / 'chapters').glob('*.docx'))
    excluded = {'runs', 'remote-run', 'transfer', 'archive', 'acceptance'}
    files.update(path for path in (ROOT / 'plan').rglob('*') if path.is_file() and
                 path.suffix in ('.md', '.json', '.txt') and not excluded.intersection(path.relative_to(ROOT / 'plan').parts))
    files.add(ROOT / 'plan/acceptance/p012-accepted-20261120.json')
    entries = []
    with tarfile.open(str(output), 'w:gz') as archive:
        for path in sorted(files):
            relative = path.relative_to(ROOT).as_posix()
            data = path.read_bytes()
            entries.append(dict(path=relative, bytes=len(data), sha256=hashlib.sha256(data).hexdigest()))
            info = tarfile.TarInfo(relative)
            info.size, info.mode = len(data), 0o644
            archive.addfile(info, io.BytesIO(data))
        data = json.dumps(entries, ensure_ascii=False, indent=2).encode('utf-8')
        info = tarfile.TarInfo('p5-delivery-manifest.json')
        info.size, info.mode = len(data), 0o644
        archive.addfile(info, io.BytesIO(data))
    # Validate member bytes after compression, not merely the archive checksum.
    with tarfile.open(str(output), 'r:gz') as archive:
        for entry in entries:
            data = archive.extractfile(entry['path']).read()
            if hashlib.sha256(data).hexdigest() != entry['sha256']:
                raise ValueError('Delivery member mismatch')
    checksum = hashlib.sha256(output.read_bytes()).hexdigest()
    output.with_suffix(output.suffix + '.sha256').write_bytes((checksum + '  ' + output.name + '\n').encode('utf-8'))
    print(json.dumps(dict(path=str(output.resolve()), files=len(entries), bytes=output.stat().st_size, sha256=checksum)))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    build(parser.parse_args().output)
