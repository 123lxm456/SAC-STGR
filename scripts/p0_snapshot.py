"""Create a byte-identical source work copy from protected reference files."""

from collections import Counter
from hashlib import sha256
import json
from pathlib import Path
import xml.etree.ElementTree as ET
import argparse
import tarfile


ROOT = Path(__file__).resolve().parents[1]
REFERENCE = ROOT / "references" / "SAC-RWB-main"
DESTINATION = ROOT / "src" / "sac_rwb"
REPORT = ROOT / "plan" / "p0" / "snapshot.json"
ARCHIVE = ROOT / 'plan/p0/transfer/reference-inputs-20261120.tar.gz'


def digest(path: Path) -> str:
    return sha256(path.read_bytes()).hexdigest()


def reference_inputs(archive=None):
    """Read verified bytes without extracting or writing into references/."""
    manifest = json.loads((ROOT / 'plan/review/protected-integrity.json').read_text(encoding='utf-8'))
    all_entries = [item for item in manifest['files'] if item['path'].startswith('references/')]
    entries = [item for item in all_entries if item['path'].startswith('references/SAC-RWB-main/')]
    if archive is None and (not (ROOT / 'references').exists() or not REFERENCE.exists()):
        archive = ARCHIVE
    if archive is not None:
        archive = Path(archive).resolve()
        if not archive.is_file():
            raise FileNotFoundError('P0 original inputs missing. Upload the verified input archive to {}. '
                                    'Do not copy revised src/ into references/.'.format(archive))
        with tarfile.open(str(archive), 'r:gz') as package:
            members = package.getmembers()
            expected = {item['path']: item['sha256'] for item in all_entries}
            if len(members) != len(expected) or {m.name for m in members} != set(expected):
                raise ValueError('Reference archive inventory differs from protected manifest')
            payloads = {}
            for member in members:
                if not member.isfile():
                    raise ValueError('Reference archive must contain only regular files')
                payload = package.extractfile(member).read()
                if sha256(payload).hexdigest() != expected[member.name]:
                    raise ValueError('Reference archive hash mismatch: ' + member.name)
                payloads[member.name] = payload
        provenance = {'kind': 'verified_archive', 'path': str(archive), 'sha256': digest(archive)}
    else:
        payloads = {}
        for item in entries:
            source = ROOT / item['path']
            payload = source.read_bytes()
            if sha256(payload).hexdigest() != item['sha256']:
                raise ValueError('Protected source differs from baseline: {}'.format(source))
            payloads[item['path']] = payload
        provenance = {'kind': 'read_only_reference', 'path': str(REFERENCE)}
    return entries, payloads, provenance


def main(destination=DESTINATION, report_path=REPORT, archive=None) -> None:
    destination = Path(destination).resolve()
    report_path = Path(report_path).resolve()
    destination.relative_to(ROOT)
    report_path.relative_to(ROOT)
    for protected_root in (ROOT / 'references', ROOT / '.git'):
        for candidate in (destination, report_path):
            try:
                candidate.relative_to(protected_root)
            except ValueError:
                continue
            raise ValueError('Snapshot outputs cannot be inside a protected directory')
    if destination.exists() and any(destination.rglob("*")):
        raise RuntimeError(f"Refusing to overwrite nonempty work copy: {destination}")
    if report_path.exists():
        raise RuntimeError(f"Refusing to overwrite existing snapshot report: {report_path}")

    entries, payloads, provenance = reference_inputs(archive)
    checked = [(ROOT / item['path'], item, item['sha256']) for item in entries]

    for source, _, actual in checked:
        target = destination / source.relative_to(REFERENCE)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(payloads[source.relative_to(ROOT).as_posix()])
        if digest(target) != actual:
            raise RuntimeError(f"Copy differs from source: {target}")

    route = ET.fromstring(payloads['references/SAC-RWB-main/real_data/newroute1220_start0.xml'])
    network = ET.fromstring(payloads['references/SAC-RWB-main/real_data/CSeditClean_1.net_threelegs.xml'])
    routes = Counter(element.tag for element in route)
    junction_types = {
        element.get("id"): element.get("type")
        for element in network.findall("junction")
    }
    report = {
        "scope": "P0 reference and work-copy inventory; no simulation run",
        "work_copy_file_count": len(checked),
        "copy_matches_reference": True,
        "input_provenance": provenance,
        "files": [
            {
                "source": item["path"],
                "copy": (destination / source.relative_to(REFERENCE))
                .relative_to(ROOT).as_posix(),
                "sha256": actual,
                "bytes": len(payloads[item['path']]),
            }
            for source, item, actual in checked
        ],
        "route_counts": dict(routes),
        "target_junction_types": {
            key: junction_types.get(key) for key in ("229", "499", "332", "334")
        },
        "requirements_overlay": "plan/p0/requirements-server.txt",
    }
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"Verified and copied {len(checked)} files; wrote {report_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--destination', type=Path, default=DESTINATION)
    parser.add_argument('--report', type=Path, default=REPORT)
    parser.add_argument('--archive', type=Path, help='Read verified original inputs without extracting into references/')
    args = parser.parse_args()
    main(args.destination, args.report, args.archive)
