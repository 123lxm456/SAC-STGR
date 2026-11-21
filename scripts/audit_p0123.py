"""Read-only structural/dependency audit; does not grant target acceptance.

Historical sources are compared after LF normalization because transferred
text files may use CRLF. Binary artifacts are always checked byte-for-byte.
No pickle is executed and no archive is extracted onto the filesystem.
"""

import argparse
import ast
import gzip
import hashlib
import json
from pathlib import Path
import pickletools
import re
import tarfile
import xml.etree.ElementTree as ET
import zipfile

ROOT = Path(__file__).resolve().parents[1]
BASELINE = 'plan/p3/runs/20261120T132252406987Z'


def sha(data):
    return hashlib.sha256(data).hexdigest()


def audit(output=None):
    errors = []
    counts = {}
    imports = {}
    files = [p for directory in ('src', 'scripts', 'tests', 'plan', 'chapters', 'references')
             for p in (ROOT / directory).rglob('*')
             if p.is_file() and '__pycache__' not in p.parts]
    for p in sorted(files):
        rel = p.relative_to(ROOT).as_posix()
        try:
            suffix = p.suffix.lower()
            if suffix == '.py':
                tree = ast.parse(p.read_text(encoding='utf-8-sig'), filename=rel)
                counts['python_ast'] = counts.get('python_ast', 0) + 1
                if rel.startswith(('src/', 'scripts/', 'tests/')):
                    imports[rel] = sorted(set(
                        node.module for node in ast.walk(tree)
                        if isinstance(node, ast.ImportFrom) and node.module))
                    imports[rel] += sorted(set(alias.name for node in ast.walk(tree)
                                              if isinstance(node, ast.Import)
                                              for alias in node.names))
            elif suffix in ('.xml', '.sumocfg'):
                tree = ET.parse(str(p))
                counts['xml'] = counts.get('xml', 0) + 1
                if suffix == '.sumocfg':
                    for element in tree.getroot().findall('./input/*'):
                        for name in element.get('value', '').split(','):
                            if name and not (p.parent / name).is_file():
                                raise ValueError('missing SUMO input ' + name)
            elif suffix == '.json':
                json.loads(p.read_text(encoding='utf-8-sig'))
                counts['json'] = counts.get('json', 0) + 1
            elif suffix in ('.pth', '.pt', '.docx'):
                with zipfile.ZipFile(str(p)) as archive:
                    if archive.testzip() is not None:
                        raise ValueError('ZIP CRC mismatch')
                    if suffix == '.docx':
                        for name in archive.namelist():
                            if name.endswith('.xml'):
                                ET.fromstring(archive.read(name))
                counts['zip_crc'] = counts.get('zip_crc', 0) + 1
            elif suffix == '.pkl':
                list(pickletools.genops(p.read_bytes()))
                counts['pickle_structure_only'] = counts.get('pickle_structure_only', 0) + 1
            elif suffix == '.gz' and not p.name.endswith('.tar.gz'):
                with gzip.open(str(p), 'rb') as stream:
                    while stream.read(1024 * 1024):
                        pass
                counts['gzip_crc'] = counts.get('gzip_crc', 0) + 1
        except Exception as error:
            errors.append({'path': rel, 'error': str(error)})

    # Check recoverability of the previously removed expanded run copies.
    cold = json.loads((ROOT / 'plan/review/cold-snapshot-cleanup-20261120.json').read_text(encoding='utf-8'))
    archive_path = ROOT / cold['archive']['path']
    try:
        assert sha(archive_path.read_bytes()) == cold['archive']['sha256'], 'cold archive SHA-256'
        with tarfile.open(str(archive_path), 'r:gz') as archive:
            for item in cold['files']:
                data = archive.extractfile(item['path']).read()
                assert len(data) == item['bytes'] and sha(data) == item['sha256'], item['path']
        counts['cold_archive_members_verified'] = len(cold['files'])
    except Exception as error:
        errors.append({'path': str(archive_path.relative_to(ROOT)), 'error': str(error)})

    changed, matched = [], []
    baseline = json.loads((ROOT / BASELINE / 'source-manifest.json').read_text(encoding='utf-8'))
    for entry in baseline:
        p = ROOT / entry['path']
        status = 'missing' if not p.is_file() else (
            'matched' if sha(p.read_bytes().replace(b'\r\n', b'\n')) == entry['sha256'] else 'changed')
        (matched if status == 'matched' else changed).append(
            entry['path'] if status == 'matched' else {'path': entry['path'], 'status': status})

    evidence = {}
    try:
        for name, directory, expected_count in (
                ('p012', 'plan/acceptance/20261120T104147769112Z', 35),
                ('p3_latest', BASELINE, 51)):
            run = ROOT / directory
            summary = json.loads((run / 'summary.json').read_text(encoding='utf-8'))
            unit = json.loads((run / 'unit-results.json').read_text(encoding='utf-8'))
            assert summary['mode'] == 'target' and summary['passed'] is True, name
            assert unit['passed'] and unit['tests_run'] == expected_count, name
            assert not any(unit[key] for key in ('skipped', 'failures', 'errors')), name
            for command in summary['commands']:
                assert command['exitcode'] == 0, command['name']
                assert int((run / (command['name'] + '.exitcode')).read_text()) == 0, command['name']
            evidence[name] = {'passed': True, 'tests_run': expected_count,
                              'commands': len(summary['commands'])}
        for variant in ('b0', 'm1'):
            run = ROOT / BASELINE
            runtime = json.loads((run / (variant + '-runtime.json')).read_text(encoding='utf-8'))
            assert runtime['decision_steps'] == runtime['inference']['calls'] == 100, variant
            assert runtime['sac_updates'] == 5 and runtime['risk_updates'] == 0, variant
            for entry in baseline:
                if not entry['path'].startswith('src/sac_rwb/'):
                    continue
                relative = entry['path'][len('src/sac_rwb/'):]
                p = run / (variant + '-work') / relative
                if relative == 'real_data/osm.sumocfg':
                    config = json.loads((run / (variant + '-config.json')).read_text(encoding='utf-8'))
                    assert sha(p.read_bytes().replace(b'\r\n', b'\n')) == config['sha256'], str(p)
                else:
                    assert sha(p.read_bytes().replace(b'\r\n', b'\n')) == entry['sha256'], str(p)
            evidence[variant] = {'snapshot_verified': True, 'sac_updates': 5, 'risk_updates': 0}
    except Exception as error:
        errors.append({'path': 'target acceptance evidence', 'error': str(error)})

    # Active documents, not source README snapshots embedded in historical runs.
    broken = []
    pages = [ROOT / 'README.md', ROOT / 'research_implementation_plan.md']
    pages += list((ROOT / 'chapters').glob('*.md'))
    pages += [p for p in (ROOT / 'plan').rglob('*.md')
              if not any(part in ('runs', 'acceptance', 'remote-run') for part in p.parts)]
    for p in pages:
        for target in re.findall(r'(?<!!)\[[^\]]+\]\(([^)]+)\)', p.read_text(encoding='utf-8')):
            if target.startswith(('https:', 'http:', 'mailto:', '#')):
                continue
            local = target.split('#', 1)[0]
            destination = (p.parent / local).resolve()
            # The current report is written after this read-only audit finishes.
            if local and not destination.exists() and destination != output:
                broken.append(p.relative_to(ROOT).as_posix() + ' -> ' + target)
    if broken:
        errors.append({'path': 'active document links', 'error': broken})
    return {'scope': 'Structure, input dependencies and historical recovery; no SUMO or GPU run',
            'structural_checks_passed': not errors, 'counts': counts, 'errors': errors,
            'historical_target_evidence': evidence,
            'baseline': BASELINE, 'baseline_matched_entries': len(matched),
            'baseline_source_delta': changed,
            'new_source_files': sorted(p.relative_to(ROOT).as_posix() for p in files
                                       if p.relative_to(ROOT).parts[0] in ('src', 'scripts', 'tests')
                                       and p.relative_to(ROOT).as_posix() not in {e['path'] for e in baseline}),
            'imports': imports,
            'target_acceptance_of_current_delta': False, 'p4_started': False}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error('Refusing to overwrite existing audit evidence')
    report = audit(args.output.resolve())
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps({k: v for k, v in report.items() if k != 'imports'}, ensure_ascii=False, indent=2))
    raise SystemExit(0 if report['structural_checks_passed'] else 1)
