"""Collect P0/P1/P2 acceptance evidence; never connects by SSH or starts P3."""

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import platform
import re
import shutil
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]


def write_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding='utf-8')


def source_manifest():
    files = []
    for directory in ('src', 'scripts', 'tests'):
        files.extend(p for p in (ROOT / directory).rglob('*')
                     if p.is_file() and '__pycache__' not in p.parts)
    files.extend(ROOT / name for name in ('plan/p0/requirements-server.txt',
                                         'plan/review/protected-integrity.json'))
    return [{'path': p.relative_to(ROOT).as_posix(),
             'sha256': hashlib.sha256(p.read_bytes()).hexdigest(), 'bytes': p.stat().st_size}
            for p in sorted(files)]


def collect_environment(require_target):
    import numpy
    import torch
    try:
        from importlib.metadata import version
    except ImportError:
        from importlib_metadata import version
    expected = {}
    for line in (ROOT / 'plan/p0/requirements-server.txt').read_text(encoding='utf-8').splitlines():
        if line.strip() and not line.startswith('#'):
            package, pin = line.split('==')
            expected[package] = pin
    actual = {name: version(name) for name in expected}
    info = {'platform': platform.platform(), 'python': sys.version, 'executable': sys.executable,
            'numpy': numpy.__version__, 'torch': torch.__version__,
            'cuda': torch.version.cuda, 'cuda_available': torch.cuda.is_available(),
            'packages': actual, 'target_required': require_target}
    if torch.cuda.is_available():
        info['gpu'] = torch.cuda.get_device_name(0)
    if require_target:
        info['sumo'] = subprocess.check_output(['sumo', '--version'], text=True).splitlines()[0]
        info['gpu_driver'] = subprocess.check_output(
            ['nvidia-smi', '--query-gpu=name,driver_version', '--format=csv,noheader'], text=True).strip()
        info['target_errors'] = []
        checks = [
            (platform.system() == 'Linux', 'Linux required'),
            (sys.version_info[:3] == (3, 7, 12), 'Python 3.7.12 required'),
            ('RTX 4090' in info.get('gpu', ''), 'CUDA RTX 4090 required'),
            (torch.version.cuda == '11.7', 'PyTorch CUDA 11.7 required'),
            (re.search(r'Version 1\.12\.0\b', info['sumo']) is not None, 'SUMO 1.12.0 required'),
            ('555.42.06' in info['gpu_driver'], 'recorded NVIDIA driver 555.42.06 required'),
        ]
        checks.extend((actual[name] == pin, '{} must be {}'.format(name, pin))
                      for name, pin in expected.items())
        info['target_errors'] = [message for ok, message in checks if not ok]
    return info


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--local', action='store_true', help='CPU/static checks only; cannot grant stage acceptance')
    parser.add_argument('--run-dir', type=Path)
    args = parser.parse_args()
    parent = ROOT / 'plan/acceptance'
    parent.mkdir(exist_ok=True)
    run = (args.run_dir or parent / datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')).resolve()
    run.relative_to(parent.resolve())
    run.mkdir()  # Must not already exist; no evidence is truncated on reruns.
    report = {'mode': 'local' if args.local else 'target', 'passed': False,
              'stages': {}, 'commands': [], 'p3_started': False}
    write_json(run / 'source-manifest.json', source_manifest())

    def command(name, argv, cwd=ROOT, timeout=1200):
        print('Running {} ...'.format(name), flush=True)
        with (run / (name + '.log')).open('w', encoding='utf-8') as stream:
            try:
                code = subprocess.run([str(x) for x in argv], cwd=str(cwd), stdout=stream,
                                      stderr=subprocess.STDOUT, timeout=timeout).returncode
            except subprocess.TimeoutExpired:
                code = 124
        (run / (name + '.exitcode')).write_text(str(code) + '\n', encoding='utf-8')
        report['commands'].append({'name': name, 'argv': [str(x) for x in argv],
                                   'cwd': str(cwd), 'exitcode': code})
        if code:
            raise RuntimeError('{} failed with exit code {}; see its log'.format(name, code))

    try:
        if args.local:
            # Local auxiliary dependencies can differ from the target lock.
            import numpy
            import torch
            info = {'platform': platform.platform(), 'python': sys.version,
                    'numpy': numpy.__version__, 'torch': torch.__version__,
                    'cuda_available': torch.cuda.is_available()}
        else:
            info = collect_environment(True)
        write_json(run / 'environment.json', info)
        if info.get('target_errors'):
            raise RuntimeError('Target environment differs from lock: ' + '; '.join(info['target_errors']))
        command('static', [sys.executable, '-B', ROOT / 'scripts/verify_plan.py',
                           '--report', run / 'static.json'])
        unit_command = [sys.executable, '-B', ROOT / 'scripts/run_unit_checks.py',
                        '--output', run / 'unit-results.json']
        if not args.local:
            unit_command.append('--require-cuda')
        command('unit', unit_command)
        if args.local:
            report['local_checks_passed'] = True
            report['remaining'] = 'Linux/RTX4090 P0 smoke, revised P1 train/test and default-width P2 CUDA verification'
        else:
            command('p0', ['bash', ROOT / 'scripts/p0_remote.sh', run / 'p0'])
            p0_log = (run / 'p0/train.log').read_text(encoding='utf-8', errors='replace')
            if 'Timestep: 100' not in p0_log:
                raise RuntimeError('P0 did not record 100 decision steps')
            report['stages']['P0'] = 'passed'
            work = run / 'p1-work'
            shutil.copytree(ROOT / 'src/sac_rwb', work, ignore=shutil.ignore_patterns('__pycache__'))
            command('p1-train', ['timeout', '20m', sys.executable, '-B', 'train_sac.py',
                                 '--epochs', '1', '--max_e_steps', '100', '--update_every', '50',
                                 '--save_interval', '100', '--dvc', 'cuda', '--seed', '0'], cwd=work)
            text = (run / 'p1-train.log').read_text(encoding='utf-8', errors='replace')
            if not re.search(r'Optimizer updates: SAC=[1-9]\d*, risk=\d+', text):
                raise RuntimeError('P1 smoke has no confirmed SAC optimizer update')
            command('p1-test', ['timeout', '20m', sys.executable, '-B', 'test_sac.py',
                                '--run_num', '0', '--timestep', '100', '--max_e_steps', '100',
                                '--test_logs_path', str(run / 'test_results.pkl'),
                                '--dvc', 'cuda', '--seed', '2'], cwd=work)
            for suffix in ('actor', 'q_critic'):
                if not (work / 'model/SAC_model_0/unsignalizedIntersections_{}_100.pth'.format(suffix)).is_file():
                    raise RuntimeError('Missing SAC checkpoint: ' + suffix)
            if not (run / 'test_results.pkl').is_file():
                raise RuntimeError('Missing test result')
            report['stages'].update({'P1': 'passed', 'P2': 'passed'})
            report['passed'] = True
        if source_manifest() != json.loads((run / 'source-manifest.json').read_text(encoding='utf-8')):
            report['passed'] = False
            raise RuntimeError('Project sources changed during verification')
    except Exception as error:
        report['error'] = '{}: {}'.format(type(error).__name__, error)
        print(report['error'], file=sys.stderr)
    finally:
        write_json(run / 'summary.json', report)
        print('Evidence: {}'.format(run), flush=True)
    return 0 if (report['passed'] or report.get('local_checks_passed')) and 'error' not in report else 1


if __name__ == '__main__':
    sys.exit(main())
