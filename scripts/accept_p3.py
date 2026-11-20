"""Validate P3 engineering integration after accepted P0/P1/P2; no SSH or P4."""

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import shutil
import subprocess
import sys
import xml.etree.ElementTree as ET
import hashlib

from accept_p012 import ROOT, collect_environment, source_manifest, write_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--local', action='store_true', help='CPU checks only; cannot grant P3 acceptance')
    parser.add_argument('--episodes', type=int, default=1)
    parser.add_argument('--steps', type=int, default=100)
    parser.add_argument('--require-online-risk-updates', action='store_true')
    args = parser.parse_args()
    if args.episodes < 1 or args.steps < 100:
        parser.error('episodes must be positive and steps at least 100')
    gate = json.loads((ROOT / 'plan/acceptance/p012-accepted-20261120.json').read_text(encoding='utf-8'))
    if not gate.get('accepted') or gate.get('stages') != dict.fromkeys(('P0', 'P1', 'P2'), 'passed'):
        raise RuntimeError('P0/P1/P2 must be accepted before P3')
    parent = ROOT / 'plan/p3/runs'
    parent.mkdir(exist_ok=True)
    run = parent / datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
    run.mkdir()
    report = {'mode': 'local' if args.local else 'target', 'passed': False,
              'p4_started': False, 'commands': [], 'variants': {},
              'acceptance_scope': 'Engineering integration only, not effectiveness or calibrated risk',
              'require_online_risk_updates': args.require_online_risk_updates}
    before = source_manifest()
    write_json(run / 'source-manifest.json', before)

    def command(name, argv, cwd=ROOT):
        print('Running {} ...'.format(name), flush=True)
        with (run / (name + '.log')).open('w', encoding='utf-8') as stream:
            # GNU timeout terminates the process group, including SUMO children.
            code = subprocess.run([str(x) for x in argv], cwd=str(cwd), stdout=stream,
                                  stderr=subprocess.STDOUT).returncode
        (run / (name + '.exitcode')).write_text(str(code) + '\n', encoding='utf-8')
        report['commands'].append({'name': name, 'argv': [str(x) for x in argv], 'exitcode': code})
        if code:
            raise RuntimeError('{} failed; see its log'.format(name))

    try:
        if not args.local:
            environment = collect_environment(True)
            write_json(run / 'environment.json', environment)
            if environment.get('target_errors'):
                raise RuntimeError('; '.join(environment['target_errors']))
        unit = [sys.executable, '-B', ROOT / 'scripts/run_unit_checks.py', '--output', run / 'unit-results.json']
        if not args.local:
            unit = ['timeout', '20m'] + unit + ['--require-cuda']
        command('unit', unit)
        command('static', [sys.executable, '-B', ROOT / 'scripts/verify_plan.py', '--report', run / 'static.json'])
        command('benchmark', [sys.executable, '-B', ROOT / 'scripts/benchmark_risk.py',
                              '--device', 'cpu' if args.local else 'cuda', '--output', run / 'latency.json',
                              '--repeats', '3' if args.local else '20', '--warmup', '1' if args.local else '5'])
        if args.local:
            report['local_checks_passed'] = True
        else:
            for variant in ('b0', 'm1'):
                work = run / (variant + '-work')
                shutil.copytree(ROOT / 'src/sac_rwb', work, ignore=shutil.ignore_patterns(
                    '__pycache__', 'model', 'risk_model', 'Train_logs',
                    'tensorboard_logs', 'sumolog', 'rawdump.xml'))
                # Suppress only the bulk output dump in disposable run copies.
                # All simulator behavior/configuration stays identical across variants.
                config = work / 'real_data/osm.sumocfg'
                tree = ET.parse(config)
                for output in tree.getroot().findall('output'):
                    for dump in output.findall('netstate-dump'):
                        output.remove(dump)
                tree.write(config, encoding='utf-8', xml_declaration=True)
                write_json(run / (variant + '-config.json'), {
                    'change': 'Only removed netstate-dump from disposable copy to avoid bulk rawdump.xml',
                    'sha256': hashlib.sha256(config.read_bytes()).hexdigest()})
                limit = '20m' if args.episodes == 1 and args.steps == 100 else '4h'
                command(variant + '-train', ['timeout', limit, sys.executable, '-B', 'train_sac.py',
                                             '--risk_variant', variant, '--epochs', str(args.episodes),
                                             '--max_e_steps', str(args.steps), '--save_interval', str(args.steps),
                                             '--update_every', '50', '--dvc', 'cuda', '--seed', '0'], cwd=work)
                runtime = json.loads((work / 'Train_logs/unsignalizedIntersections/risk_runtime_0.json').read_text(encoding='utf-8'))
                if runtime['risk_variant'] != variant or runtime['sac_updates'] < 1:
                    raise RuntimeError('Incorrect variant or missing SAC optimization')
                if runtime['inference']['calls'] != args.episodes * args.steps:
                    raise RuntimeError('Missing online risk scoring calls')
                if args.require_online_risk_updates and runtime['risk_updates'] < 1:
                    raise RuntimeError('No natural-outcome risk optimizer updates yet; extend diagnostic run')
                command(variant + '-test', ['timeout', '20m', sys.executable, '-B', 'test_sac.py',
                                            '--run_num', '0', '--timestep', str(args.episodes * args.steps),
                                            '--max_e_steps', '100', '--dvc', 'cuda', '--seed', '2',
                                            '--test_logs_path', str(run / (variant + '-test.pkl'))], cwd=work)
                write_json(run / (variant + '-runtime.json'), runtime)
                report['variants'][variant] = {
                    'decision_steps': runtime['decision_steps'], 'sac_updates': runtime['sac_updates'],
                    'risk_updates': runtime['risk_updates'],
                    'natural_risk_learning_observed': runtime['risk_updates'] > 0,
                    'inference_calls': runtime['inference']['calls']}
            report['passed'] = True
        if source_manifest() != before:
            report['passed'] = False
            raise RuntimeError('Sources changed during P3 verification')
    except Exception as error:
        report['error'] = '{}: {}'.format(type(error).__name__, error)
        print(report['error'], file=sys.stderr)
    finally:
        write_json(run / 'summary.json', report)
        print('Evidence: {}'.format(run), flush=True)
    return 0 if (report['passed'] or report.get('local_checks_passed')) and 'error' not in report else 1


if __name__ == '__main__':
    sys.exit(main())
