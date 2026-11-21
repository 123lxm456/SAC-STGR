"""P4 acceptance: real labels, isolation, natural-rate audit and calibration.

Exit 0 = target P4 gate passed (or auxiliary --local checks passed).
Exit 2 = engineering passed, natural-event evidence insufficient.
Exit 1 = failed checks or execution. Never starts P5; never connects SSH.
"""

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess
import shutil
import sys
import xml.etree.ElementTree as ET

from accept_p012 import ROOT, collect_environment, source_manifest, write_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--local', action='store_true')
    parser.add_argument('--steps', type=int, default=1000)
    parser.add_argument('--replicates', type=int, default=1)
    args = parser.parse_args()
    if args.steps < 100 or not 1 <= args.replicates <= 50:
        parser.error('steps >= 100, replicates in [1,50] required')
    sys.path.insert(0, str(ROOT / 'src/sac_rwb'))
    from algos.p4_data import create_scenarios, audit_datasets, read_dataset, write_reliability
    from algos.horizon_risk import PROTOCOL
    gate = json.loads((ROOT / 'plan/review/p0123-target-confirmation-20261121.json').read_text(encoding='utf-8'))
    if not gate.get('passed') or not gate.get('p4_engineering_ready'):
        raise RuntimeError('Accepted current P0-P3 engineering baseline is required')
    parent = ROOT / 'plan/p4/runs'
    # Create one level at a time: Windows UNC may report a missing ancestor as
    # PermissionError rather than FileNotFoundError used by pathlib recursion.
    parent.parent.mkdir(exist_ok=True)
    parent.mkdir(exist_ok=True)
    run = parent / datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
    run.mkdir()
    before = source_manifest()
    write_json(run / 'source-manifest.json', before)
    report = {'mode': 'local' if args.local else 'target', 'passed': False,
              'engineering_passed': False, 'data_ready': False, 'commands': [],
              'variants': {}, 'p4_started': True, 'p5_started': False}

    def command(name, argv, cwd=ROOT):
        print('Running {} ...'.format(name), flush=True)
        with (run / (name + '.log')).open('w', encoding='utf-8') as stream:
            try:
                code = subprocess.run([str(x) for x in argv], cwd=str(cwd),
                                      stdout=stream, stderr=subprocess.STDOUT, timeout=3600).returncode
            except subprocess.TimeoutExpired:
                code = 124
        (run / (name + '.exitcode')).write_text(str(code) + '\n', encoding='utf-8')
        report['commands'].append({'name': name, 'argv': [str(x) for x in argv], 'exitcode': code})
        if code:
            raise RuntimeError('{} failed; see log'.format(name))

    try:
        if not args.local:
            environment = collect_environment(True)
            write_json(run / 'environment.json', environment)
            if environment.get('target_errors'):
                raise RuntimeError('; '.join(environment['target_errors']))
        locked = json.loads((ROOT / 'plan/p4/protocol.json').read_text(encoding='utf-8'))
        if locked != PROTOCOL:
            raise RuntimeError('Locked P4 protocol disagrees with collector implementation')
        unit = [sys.executable, '-B', ROOT / 'scripts/run_unit_checks.py', '--output', run / 'unit-results.json']
        if not args.local:
            unit = ['timeout', '20m'] + unit + ['--require-cuda']
        command('unit', unit)
        units = json.loads((run / 'unit-results.json').read_text(encoding='utf-8'))
        if not units.get('passed') or units.get('tests_run', 0) < 1 or units.get('errors') or units.get('failures'):
            raise RuntimeError('Unit result is missing, incomplete or failed')
        if not args.local and (units.get('skipped') or not units.get('require_cuda')):
            raise RuntimeError('Target tests require CUDA and zero skips')
        command('static', [sys.executable, '-B', ROOT / 'scripts/verify_plan.py', '--report', run / 'static.json'])
        scenarios = create_scenarios(ROOT / 'src/sac_rwb/real_data', run / 'scenarios')
        write_json(run / 'protocol.json', scenarios['protocol'])
        if args.local:
            report['local_checks_passed'] = True
        else:
            baseline = ROOT / gate['target_evidence']
            for variant in ('b0', 'm1'):
                work = run / (variant + '-work')
                shutil.copytree(ROOT / 'src/sac_rwb', work, ignore=shutil.ignore_patterns(
                    '__pycache__', 'model', 'risk_model', 'Train_logs', 'tensorboard_logs', 'sumolog', 'rawdump.xml'))
                config = work / 'real_data/osm.sumocfg'
                tree = ET.parse(str(config))
                for output in tree.getroot().findall('output'):
                    for dump in output.findall('netstate-dump'):
                        output.remove(dump)
                tree.write(str(config), encoding='utf-8', xml_declaration=True)
                online_data = run / (variant + '-online-training')
                command(variant + '-online-training', [
                    'timeout', '20m', sys.executable, '-B', 'train_sac.py', '--risk_variant', variant,
                    '--risk_supervision', 'horizon', '--risk_dataset_dir', online_data,
                    '--epochs', '1', '--max_e_steps', '100', '--save_interval', '100',
                    '--dvc', 'cuda', '--seed', '100'], cwd=work)
                runtime = json.loads((work / 'Train_logs/unsignalizedIntersections/risk_runtime_0.json').read_text(encoding='utf-8'))
                if runtime['risk_supervision'] != 'fixed_future_horizon_p4-h10-v1' or runtime['decision_steps'] != 100:
                    raise RuntimeError('Incorrect online P4 mode or budget')
                if runtime['inference']['calls'] != 100 or runtime['sac_updates'] < 1:
                    raise RuntimeError('Online P4 scoring/SAC path did not run')
                _, online_rows, _ = read_dataset(online_data)
                if not online_rows:
                    raise RuntimeError('Online P4 did not preserve executed decisions')
                write_json(run / (variant + '-online-runtime.json'), runtime)
                old = baseline / (variant + '-work')
                actor = old / 'model/SAC_model_0/unsignalizedIntersections_actor_100.pth'
                weights = list((old / 'risk_model').glob('risk_{}_*step100.pth'.format(variant)))
                if not actor.is_file() or len(weights) != 1:
                    raise RuntimeError('Missing unambiguous frozen P3 checkpoints for ' + variant)
                directories = []
                for split in ('train', 'validation', 'test'):
                    for replicate in range(args.replicates):
                        output = run / variant / ('{}-{}'.format(split, replicate))
                        command('{}-{}-{}'.format(variant, split, replicate), [
                            'timeout', '50m', sys.executable, '-B', ROOT / 'scripts/collect_p4.py',
                            '--scenarios', run / 'scenarios/scenarios.json', '--split', split,
                            '--variant', variant, '--replicate', str(replicate), '--actor', actor,
                            '--risk-model', weights[0], '--steps', str(args.steps), '--output', output], cwd=work)
                        status = json.loads((output / 'collection.json').read_text(encoding='utf-8'))
                        if not status['completed'] or status['decision_steps'] != args.steps:
                            raise RuntimeError('Incomplete collection: ' + str(output))
                        directories.append(output)
                audit = audit_datasets(directories)
                write_json(run / (variant + '-audit.json'), audit)
                write_reliability(audit, run / (variant + '-reliability.svg'))
                report['variants'][variant] = {'isolation_passed': audit['isolation_passed'],
                                               'data_ready': audit['data_ready'],
                                               'online_sac_updates': runtime['sac_updates'],
                                               'online_risk_updates': runtime['risk_updates'],
                                               'physical_collision_vehicle_events': audit['physical_collision_vehicle_events']}
            report['engineering_passed'] = True
            report['data_ready'] = all(x['data_ready'] for x in report['variants'].values())
            report['passed'] = report['data_ready']
            report['status'] = 'passed' if report['passed'] else 'needs_more_natural_events'
        if source_manifest() != before:
            raise RuntimeError('Sources changed during P4 verification')
    except Exception as error:
        report.update(passed=False, engineering_passed=False, error='{}: {}'.format(type(error).__name__, error))
        print(report['error'], file=sys.stderr)
    finally:
        write_json(run / 'summary.json', report)
        print('Evidence: {}'.format(run), flush=True)
    if report.get('error'):
        return 1
    if args.local:
        return 0 if report.get('local_checks_passed') else 1
    return 0 if report['passed'] else (2 if report['engineering_passed'] else 1)


if __name__ == '__main__':
    sys.exit(main())
