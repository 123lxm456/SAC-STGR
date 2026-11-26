"""Read-only P0-P4 re-audit and P5 entry check; never implements P5.

Default: current auxiliary tests, static protection, historical execution proof.
--target: require locked Linux/RTX4090, no skipped tests, then re-read all server
P4 raw streams with the strengthened integrity audit. No new training or SUMO
trajectory is run. Old logs, data, checkpoints and acceptance records are read
only; each invocation writes to a new review directory.
"""

import argparse
import ast
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import pickletools
import gzip
import tarfile
import zipfile
import subprocess
import sys
import xml.etree.ElementTree as ET

from accept_p012 import ROOT, collect_environment, source_manifest, write_json

P012 = 'plan/acceptance/20261120T104147769112Z'
P3 = 'plan/p3/runs/20261121T092656852782Z'
P4 = 'plan/p4/runs/20261121T115438612730Z'
EXTENSION = 'plan/p4/runs/20261121T150247686103Z-train-extension'
IMPORT_CLEANUP = {
    'src/sac_rwb/Env.py', 'src/sac_rwb/core/monitor.py',
    'src/sac_rwb/core/sumo_interface.py', 'src/sac_rwb/core/costomized_data_structures.py',
    'src/sac_rwb/core/utils.py',
}


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def require(condition, message):
    if not condition:
        raise ValueError(message)


def digest(path, normalize=False):
    hasher = hashlib.sha256()
    if normalize:
        hasher.update(Path(path).read_bytes().replace(b'\r\n', b'\n'))
    else:
        with Path(path).open('rb') as stream:
            for block in iter(lambda: stream.read(1024 * 1024), b''):
                hasher.update(block)
    return hasher.hexdigest()


def checked_history(directory, tests=None, status='passed'):
    directory = Path(directory)
    report = read(directory / 'summary.json')
    require(report.get('mode') == 'target' and report.get(status) is True,
            'Missing target historical acceptance: ' + str(directory))
    require(bool(report.get('commands')), 'Historical command list missing')
    for command in report['commands']:
        require(command['exitcode'] == 0 and int((directory / (command['name'] + '.exitcode')).read_text()) == 0,
                'Historical command failed: ' + command['name'])
        require((directory / (command['name'] + '.log')).is_file(), 'Historical command log missing')
    if tests is not None:
        unit = read(directory / 'unit-results.json')
        require(unit.get('passed') and unit.get('require_cuda') and unit.get('tests_run') == tests and
                not any(unit.get(name) for name in ('skipped', 'errors', 'failures')),
                'Historical GPU test result incomplete or skipped')
    return report


def local_saved_path(value, remote_root):
    prefix = remote_root.rstrip('/') + '/'
    require(value.startswith(prefix), 'Saved path is outside the recorded server project')
    path = (ROOT / value[len(prefix):]).resolve()
    path.relative_to(ROOT.resolve())
    return path


def semantic_body(text):
    tree = ast.parse(text)
    # P5's exact optional callback leaves legacy runs unchanged. Strip only
    # this reviewed pair of statements in Env.step_once, never arbitrary code.
    callback = ast.parse("observer = self.config.get('physical_observer') if hasattr(self, 'config') else None\n"
                         "if observer is not None:\n    observer.step(self, sim_res)\n").body
    expected = [ast.dump(node, include_attributes=False) for node in callback]
    for node in tree.body:
        if isinstance(node, ast.ClassDef) and node.name == 'Env':
            for method in node.body:
                if isinstance(method, ast.FunctionDef) and method.name == 'step_once':
                    for index in range(len(method.body) - 1):
                        if [ast.dump(n, include_attributes=False) for n in method.body[index:index + 2]] == expected:
                            del method.body[index:index + 2]
                            break
    tree.body = [node for node in tree.body
                 if not isinstance(node, (ast.Import, ast.ImportFrom))
                 and not (isinstance(node, ast.ClassDef) and node.name == 'timer')]
    return ast.dump(tree, include_attributes=False)


def structural_proof():
    """Check dependency inputs and retained artifacts without executing them."""
    counts, relocated, imports = {}, [], {}
    remote_root = read(ROOT / EXTENSION / 'summary.json')['base_run'].split('/plan/p4/runs/', 1)[0]
    known_origins = (remote_root + '/', '//wsl.localhost/Ubuntu/home/lxm/codes/SAC-STGR/')
    for top in ('src', 'scripts', 'tests', 'plan', 'chapters', 'references'):
        for path in sorted((ROOT / top).rglob('*')):
            if not path.is_file() or '__pycache__' in path.parts:
                continue
            rel = path.relative_to(ROOT).as_posix()
            suffix = path.suffix.lower()
            kind = None
            if suffix == '.py':
                tree = ast.parse(path.read_text(encoding='utf-8-sig'), filename=rel)
                kind = 'python_ast'
                if top in ('src', 'scripts', 'tests'):
                    names = []
                    for node in ast.walk(tree):
                        if isinstance(node, ast.ImportFrom) and node.module:
                            names.append(node.module)
                        elif isinstance(node, ast.Import):
                            names.extend(alias.name for alias in node.names)
                    imports[rel] = sorted(set(names))
                    for name in names:
                        if name.split('.')[0] in ('algos', 'core', 'sac', 'Env'):
                            module = ROOT / 'src/sac_rwb' / name.replace('.', '/')
                            require(module.with_suffix('.py').is_file() or (module / '__init__.py').is_file(),
                                    'Missing active local import: {} -> {}'.format(rel, name))
            elif suffix in ('.xml', '.sumocfg'):
                tree = ET.parse(str(path))
                kind = 'xml'
                if suffix == '.sumocfg':
                    for item in tree.getroot().findall('./input/*'):
                        for value in item.get('value', '').split(','):
                            candidate = path.parent / value
                            if value:
                                normalized = value.replace('\\', '/')
                                for origin in known_origins:
                                    if normalized.startswith(origin):
                                        candidate = (ROOT / normalized[len(origin):]).resolve()
                                        candidate.relative_to(ROOT.resolve())
                                        relocated.append({'config': rel, 'input': candidate.relative_to(ROOT).as_posix()})
                                        break
                            require(not value or candidate.is_file(), 'Missing SUMO input: {} -> {}'.format(rel, value))
            elif suffix == '.json':
                read(path)
                kind = 'json'
            elif suffix in ('.pth', '.pt', '.docx'):
                with zipfile.ZipFile(str(path)) as archive:
                    require(archive.testzip() is None, 'ZIP CRC failed: ' + rel)
                    if suffix == '.docx':
                        for name in archive.namelist():
                            if name.endswith('.xml'):
                                ET.fromstring(archive.read(name))
                kind = 'checkpoint_or_docx_zip_crc'
            elif suffix == '.pkl':
                for _ in pickletools.genops(path.read_bytes()):
                    pass
                kind = 'pickle_structure_only'
            elif suffix == '.gz' and not path.name.endswith('.tar.gz'):
                with gzip.open(str(path), 'rb') as stream:
                    for _ in iter(lambda: stream.read(1024 * 1024), b''):
                        pass
                kind = 'gzip_crc'
            if kind:
                counts[kind] = counts.get(kind, 0) + 1
    cold = read(ROOT / 'plan/review/cold-snapshot-cleanup-20261120.json')
    archive_path = ROOT / cold['archive']['path']
    if os.path.isfile(str(archive_path)):
        require(digest(archive_path) == cold['archive']['sha256'], 'Cold recovery archive hash changed')
        expected = {item['path']: item for item in cold['files']}
        verified = set()
        # Stream once instead of repeatedly rewinding a large gzip on the UNC share.
        with tarfile.open(str(archive_path), 'r|gz') as archive:
            for member in archive:
                if member.name not in expected:
                    continue
                item = expected[member.name]
                require(member.isfile(), 'Cold archive member is not a regular file')
                stream = archive.extractfile(member)
                hasher = hashlib.sha256()
                size = 0
                for block in iter(lambda: stream.read(1024 * 1024), b''):
                    size += len(block)
                    hasher.update(block)
                require(size == item['bytes'] and hasher.hexdigest() == item['sha256'],
                        'Cold archive member changed: ' + member.name)
                verified.add(member.name)
        require(verified == set(expected), 'Cold archive members missing')
        counts['cold_archive_members_verified'] = len(cold['files'])
    else:
        receipt = read(ROOT / 'plan/p5/evidence/local-history-removal-20261126.json')
        retired = [item for item in receipt.get('targets', []) if item.get('path') == 'plan/archive']
        require(receipt.get('completed') is True and len(retired) == 1 and
                retired[0].get('bytes') == 46202434 and retired[0].get('files') == 3 and
                not os.path.exists(str(archive_path.parent)), 'Cold archive missing without reviewed retirement')
        counts['cold_archive_members_retired_unverifiable'] = len(cold['files'])
    original = ROOT / 'plan/p0/transfer/reference-inputs-20261120.tar.gz'
    require(digest(original) == original.with_name(original.name + '.sha256').read_text().split()[0],
            'Protected original-input bundle checksum failed')
    return {'passed': True, 'counts': counts, 'active_imports': imports,
            'relocated_historical_config_inputs': relocated,
            'scope': 'Static structures, active local imports and original recovery artifacts; no pickle execution'}


def historical_proof(target=False):
    p012 = checked_history(ROOT / P012, 35)
    p3 = checked_history(ROOT / P3, 53)
    base = checked_history(ROOT / P4, 74, 'engineering_passed')
    ext = checked_history(ROOT / EXTENSION)
    require(ext.get('data_ready') and ext.get('evaluation_unchanged') and not ext.get('p5_started'),
            'P4 extension is not accepted or evaluation changed')
    require(p012.get('stages') == dict.fromkeys(('P0', 'P1', 'P2'), 'passed'), 'P012 stage result missing')
    remote_root = ext['base_run'].split('/plan/p4/runs/', 1)[0]
    require(local_saved_path(ext['base_run'], remote_root) == (ROOT / P4).resolve(), 'Wrong base run')
    from algos.horizon_risk import PROTOCOL
    from algos.p4_data import audit_datasets
    require(read(ROOT / 'plan/p4/protocol.json') == read(ROOT / P4 / 'protocol.json') == PROTOCOL,
            'P4 label protocol changed')

    deltas = []
    for entry in read(ROOT / EXTENSION / 'source-manifest.json'):
        path = ROOT / entry['path']
        require(path.is_file(), 'Accepted source missing: ' + entry['path'])
        if entry['sha256'] in (digest(path), digest(path, True)):
            continue
        relative = entry['path']
        snapshot = ROOT / P4 / 'b0-work' / relative[len('src/sac_rwb/'):]
        if relative == 'src/sac_rwb/README.md':
            # Stage-status corrections do not change executable source. Keep
            # every other unreviewed source delta fail-closed below.
            deltas.append({'path': relative, 'scope': 'documentation status only'})
        elif relative == 'scripts/verify_plan.py':
            # P5 v2 replaced the retired C1/C2/C3 document terms with the
            # reviewed four-group vocabulary. Pin both sides so this cannot
            # become a general exemption for later verifier changes.
            require(entry['sha256'] == 'b082e1d45883832826dc4144c16797553401e362e0be284fe60dd856a8022175'
                    and digest(path, True) == 'a69fe6a1b1a2ff0472e4a8f4e183bba4efd873025d96b8cce20665ba19c7fc6d',
                    'Unreviewed plan verifier change')
            deltas.append({'path': relative, 'scope': 'pinned P5 four-group document checks only'})
        elif relative == 'scripts/audit_p0123.py':
            require(entry['sha256'] == 'c526b687a40a43693e2dcdf885c121b04c5e33aedae9d2d33aef5f324d72b90a'
                    and digest(path, True) == 'bbf2f3e7e30b352cef872e2233c297b9e0eaf77aac7edea6b5cf54dfa0554db2',
                    'Unreviewed historical audit change')
            deltas.append({'path': relative, 'scope': 'pinned cold-archive retirement handling only'})
        elif relative in IMPORT_CLEANUP:
            require(semantic_body(path.read_text(encoding='utf-8')) ==
                    semantic_body(snapshot.read_text(encoding='utf-8')),
                    'Runtime behavior changed beyond dead-code/import cleanup: ' + relative)
            deltas.append({'path': relative, 'scope': ('unused imports and exact optional P5 physical-observer callback; legacy default unchanged'
                                                      if relative == 'src/sac_rwb/Env.py' else 'unused imports / uncalled timer only')})
        elif relative == 'src/sac_rwb/algos/p4_data.py':
            old = ast.parse(snapshot.read_text(encoding='utf-8'))
            current = ast.parse(path.read_text(encoding='utf-8'))
            functions = {node.name: node for node in current.body if isinstance(node, ast.FunctionDef)}
            for node in old.body:
                if isinstance(node, ast.FunctionDef) and node.name != 'read_dataset':
                    require(ast.dump(node, include_attributes=False) == ast.dump(functions[node.name], include_attributes=False),
                            'P4 metrics/scenario/calibration behavior changed: ' + node.name)
            deltas.append({'path': relative, 'scope': 'reader integrity checks and line-wise parsing only'})
        else:
            raise ValueError('Unreviewed source delta against accepted P4: ' + relative)

    fingerprints = read(ROOT / EXTENSION / 'frozen-evaluation-sha256.json')
    missing_raw = []
    for value, expected in fingerprints.items():
        path = local_saved_path(value, remote_root)
        if not path.is_file() and not target and path.suffix == '.jsonl':
            missing_raw.append(str(path.relative_to(ROOT)))
            continue
        require(path.is_file(), 'Frozen evaluation artifact missing: ' + str(path))
        hashes = [digest(path)]
        if path.suffix == '.json':
            hashes.append(digest(path, True))
        require(expected in hashes, 'Frozen evaluation fingerprint changed: ' + str(path))

    variants = {}
    for variant in ('b0', 'm1'):
        original = read(ROOT / P4 / (variant + '-audit.json'))
        saved = read(ROOT / EXTENSION / (variant + '-audit.json'))
        require(saved['protocol'] == PROTOCOL and saved['isolation_passed'] and saved['data_ready'] and
                saved['training_data_ready'] and saved['calibration_data_ready'], 'P4 audit incomplete')
        for split in ('validation', 'test'):
            require(saved['splits'][split] == original['splits'][split], 'Frozen split statistics changed')
        require(saved['calibration'] == original['calibration'] and
                saved['calibrated_metrics'] == original['calibrated_metrics'], 'Frozen calibration changed')
        paths = [local_saved_path(value, remote_root) for value in
                 ext['training_collections'][variant] + ext['evaluation_collections'][variant]]
        totals = {split: {name: 0 for name in ('positive', 'negative', 'censored', 'windows')}
                  for split in ('train', 'validation', 'test')}
        identities = {name: {} for name in ('seed', 'scenario_id', 'route')}
        frozen_models = set()
        for directory in paths:
            manifest, counts, collection = (read(directory / name) for name in
                                             ('manifest.json', 'counts.json', 'collection.json'))
            meta = manifest['metadata']
            split = meta['split']
            require(manifest['protocol'] == PROTOCOL and meta['frozen_risk_model'], 'Dataset protocol mismatch')
            require(collection['completed'] and collection['sac_updates'] == collection['risk_updates'] == 0,
                    'Independent collection incomplete or models were updated')
            require(collection['decision_steps'] == (2000 if '-train-extension' in str(directory) else 1000),
                    'Wrong collection decision budget')
            require(counts['windows'] == counts['executed_decisions'] ==
                    sum(counts.get(key, 0) for key in ('positive', 'negative', 'censored')), 'Dataset counts disagree')
            for key in totals[split]:
                totals[split][key] += counts.get(key, 0)
            for name, values in (('seed', [meta['seed']]), ('scenario_id', [meta['scenario_id']]),
                                 ('route', meta['allowed_route_hashes'])):
                for value in values:
                    require(identities[name].setdefault(value, split) == split, 'Cross-split metadata leakage')
            frozen_models.add((meta['policy_sha256'], meta['risk_sha256']))
            for key, arg in (('policy_sha256', 'actor'), ('risk_sha256', 'risk_model')):
                weight = local_saved_path(ext['checkpoints'][variant][arg], remote_root)
                require(digest(weight) == meta[key], 'Frozen checkpoint hash changed')
        require(len(frozen_models) == 1, 'Mixed checkpoints in independent collection')
        for split, counters in totals.items():
            stats = saved['splits'][split]
            require(counters == {'positive': stats['raw_metrics']['positive_windows'],
                                  'negative': stats['raw_metrics']['negative_windows'],
                                  'censored': stats['censored'], 'windows': stats['windows']}, 'Merged counts disagree')
        for folder in (ROOT / P4, ROOT / EXTENSION):
            ET.parse(str(folder / (variant + '-reliability.svg')))
        if target:
            recomputed = audit_datasets(paths)
            require(recomputed == saved, 'Strict raw audit changed the accepted P4 result: ' + variant)
        variants[variant] = {'counts': totals, 'frozen_evaluation_and_models_verified': True,
                             'strict_raw_reaudit_passed': bool(target)}
    return {'historical_stages': dict.fromkeys(('P0', 'P1', 'P2', 'P3', 'P4'), 'passed'),
            'historical_gpu_test_runs': [35, 53, 74], 'reviewed_source_deltas': deltas,
            'frozen_metadata_fingerprints': len(fingerprints), 'raw_streams_absent_locally': missing_raw,
            'raw_reaudited': bool(target), 'variants': variants,
            'scope': 'Historical SUMO/GPU executions + current tests; no fresh training trajectory'}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--target', action='store_true')
    args = parser.parse_args()
    parent = ROOT / 'plan/review/p5-entry-runs'
    parent.mkdir(exist_ok=True)
    run = parent / datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
    run.mkdir()
    report = {'mode': 'target' if args.target else 'local', 'passed': False,
              'p5_ready': False, 'p5_started': False, 'p5_requires_user_confirmation': True,
              'commands': []}
    before = source_manifest()
    write_json(run / 'source-manifest.json', before)

    def command(name, argv):
        print('Running {} ...'.format(name), flush=True)
        if args.target:
            argv = ['timeout', '30m'] + argv
        with (run / (name + '.log')).open('w', encoding='utf-8') as stream:
            try:
                code = subprocess.run([str(x) for x in argv], cwd=str(ROOT), stdout=stream,
                                      stderr=subprocess.STDOUT, timeout=1900).returncode
            except subprocess.TimeoutExpired:
                code = 124
        (run / (name + '.exitcode')).write_text(str(code) + '\n', encoding='utf-8')
        report['commands'].append({'name': name, 'exitcode': code})
        require(code == 0, name + ' failed; see log')

    try:
        sys.path.insert(0, str(ROOT / 'src/sac_rwb'))
        if args.target:
            env = collect_environment(True)
            write_json(run / 'environment.json', env)
            require(not env.get('target_errors'), 'Target environment differs from lock')
        unit = [sys.executable, '-B', ROOT / 'scripts/run_unit_checks.py', '--output', run / 'unit-results.json']
        if args.target:
            unit.append('--require-cuda')
        command('unit', unit)
        command('static', [sys.executable, '-B', ROOT / 'scripts/verify_plan.py', '--report', run / 'static.json'])
        proof = historical_proof(args.target)
        write_json(run / 'historical-and-data-audit.json', proof)
        structure = structural_proof()
        write_json(run / 'structure-and-dependencies.json', structure)
        require(source_manifest() == before, 'Current source changed during verification')
        report.update(local_checks_passed=True, historical_stages=proof['historical_stages'],
                      raw_reaudited=proof['raw_reaudited'], passed=bool(args.target), p5_ready=bool(args.target))
    except Exception as error:
        report['error'] = '{}: {}'.format(type(error).__name__, error)
        print(report['error'], file=sys.stderr)
    finally:
        write_json(run / 'summary.json', report)
        print('Evidence: {}'.format(run), flush=True)
    return 0 if report.get('local_checks_passed') and not report.get('error') else 1


if __name__ == '__main__':
    sys.exit(main())
