"""Read immutable P5 evidence after lossless storage consolidation.

Only the migration index redirects historical paths. Original JSON evidence
is never rewritten. Compressed streams are checked against original hashes.
"""
from contextlib import contextmanager
import gzip
import hashlib
import json
import os
import shutil
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PRIMARY = '20261123T094822803672Z'
INDEX = 'plan/p5/runs/' + PRIMARY + '/storage-index.json'


def index(root=None):
    root = Path(root or ROOT)
    path = root / INDEX
    if not os.path.isfile(str(path)):
        return None
    data = json.loads(path.read_text(encoding='utf-8'))
    if data.get('version') != 'p5-storage-v1':
        raise ValueError('Unknown P5 storage index')
    return data


def relative(path, root, data):
    value = str(path).replace('\\', '/')
    for prefix in (str(root).replace('\\', '/').rstrip('/'), data['original_root'].rstrip('/')):
        if value.startswith(prefix + '/'):
            return value[len(prefix) + 1:]
    return value if not Path(value).is_absolute() else None


def safe_path(root, rel):
    value = Path(rel)
    if value.is_absolute() or '..' in value.parts:
        raise ValueError('Unsafe storage path: ' + str(rel))
    result = root / value
    result.resolve().relative_to(root.resolve())
    return result


def relocated(path, root=None):
    root = Path(root or ROOT)
    data = index(root)
    if not data:
        return Path(path)
    rel = relative(path, root, data)
    if rel is not None:
        for old, new in data['relocations'].items():
            if rel == old or rel.startswith(old + '/'):
                return safe_path(root, new + rel[len(old):])
    return Path(path)


def journal_record(path, root=None):
    root = Path(root or ROOT)
    data = index(root)
    if not data:
        return None
    rel = relative(path, root, data)
    if rel is None:
        return None
    for old, new in data['relocations'].items():
        if rel.startswith(new + '/'):
            rel = old + rel[len(new):]
            break
    entry = data['journals'].get(rel)
    if entry:
        expected = entry['sha256'] + '.jsonl.gz'
        if Path(entry['object']).name != expected:
            raise ValueError('Journal object/name mismatch')
        safe_path(root, entry['object'])
    return entry


def journal_exists(path, root=None):
    root = Path(root or ROOT)
    entry = journal_record(path, root)
    return safe_path(root, entry['object']).is_file() if entry else Path(path).is_file()


@contextmanager
def open_journal(path, root=None):
    """Read binary original bytes; gzip CRC plus full hash at EOF is mandatory.

    Consume the stream completely for verification. Early close drains it so
    a partial analysis cannot hide damage later in the compressed stream.
    """
    root = Path(root or ROOT)
    entry = journal_record(path, root)
    if not entry:
        with Path(path).open('rb') as stream:
            yield stream
        return
    with gzip.open(str(safe_path(root, entry['object'])), 'rb') as stream:
        wrapper = CheckedReader(stream, entry)
        yield wrapper
        while wrapper.read(1024 * 1024):
            pass
        wrapper.verify()


class CheckedReader:
    def __init__(self, stream, entry):
        self.stream, self.entry = stream, entry
        self.digest, self.size = hashlib.sha256(), 0

    def read(self, size=-1):
        data = self.stream.read(size)
        self.digest.update(data)
        self.size += len(data)
        return data

    def __iter__(self):
        for line in self.stream:
            self.digest.update(line)
            self.size += len(line)
            yield line

    def verify(self):
        if self.size != self.entry['bytes'] or self.digest.hexdigest() != self.entry['sha256']:
            raise ValueError('Original journal SHA-256/size mismatch')


def journal_digest_size(path, root=None):
    digest, size = hashlib.sha256(), 0
    with open_journal(path, root) as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
            size += len(block)
    return digest.hexdigest(), size


def iter_journal_rows(path, root=None):
    """Stream JSON records without materializing a multi-GB string."""
    with open_journal(path, root) as stream:
        for line in stream:
            yield json.loads(line)


def read_online_dataset(directory, root=None):
    """Compatibility with the unchanged strict P4 dataset reader.

    Materializes one dataset in temporary storage, then removes it. Prefer
    iter_journal_rows for streaming analyses that do not need full arrays.
    """
    root = Path(root or ROOT)
    directory = relocated(directory, root)
    from algos.p4_data import read_dataset
    if not journal_record(directory / 'samples.jsonl', root):
        return read_dataset(directory)
    with tempfile.TemporaryDirectory(prefix='p5-analysis-', dir=str(root)) as name:
        temp = Path(name)
        for file in ('manifest.json', 'counts.json'):
            shutil.copyfile(str(directory / file), str(temp / file))
        for file in ('samples.jsonl', 'events.jsonl'):
            with open_journal(directory / file, root) as source, (temp / file).open('wb') as target:
                shutil.copyfileobj(source, target, 1024 * 1024)
        return read_dataset(temp)
