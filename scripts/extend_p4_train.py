"""Extend only P4 train data with unused seeds; freeze existing evaluation data.

No model training, no P5, no SSH. Reuses collect_p4/audit_datasets. Exit codes:
0=data ready, 2=natural training support still insufficient, 1=execution error.
"""

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess
import sys

from accept_p012 import ROOT, collect_environment, source_manifest, write_json


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def plan_extension(base, replicates=3, steps=2000, previous=None):
    if not 1 <= replicates <= 49 or steps < 100:
        raise ValueError('replicates must be in [1,49], steps >= 100')
    base = Path(base).resolve()
    summary, units = read(base / 'summary.json'), read(base / 'unit-results.json')
    if summary.get('mode') != 'target' or not summary.get('engineering_passed'):
        raise ValueError('A completed target P4 engineering run is required')
    if not units.get('passed') or units.get('tests_run', 0) < 1 or not units.get('require_cuda') or any(
            units.get(key) for key in ('skipped', 'errors', 'failures')):
        raise ValueError('Base engineering tests must pass on CUDA without skips')
    inherited = {}
    if previous is not None:
        prior = read(Path(previous) / 'summary.json')
        if not prior.get('engineering_passed') or Path(prior['base_run']).resolve() != base:
            raise ValueError('Previous extension must be successful engineering work for this base')
        inherited = prior['training_collections']
    scenario = read(base / 'scenarios/scenarios.json')['scenarios']['train']
    training, evaluation, checkpoints, used = {}, {}, {}, set()
    for variant in ('b0', 'm1'):
        original = base / variant / 'train-0'
        original_meta = read(original / 'manifest.json')['metadata']
        if scenario['seed'] != original_meta['seed']:
            raise ValueError('Original training seed disagrees with the scenario manifest')
        directories = inherited.get(variant, [str(original)])
        if str(original) not in directories:
            raise ValueError('Original training collection must be retained')
        seeds = set()
        for directory in directories:
            manifest = read(Path(directory) / 'manifest.json')
            meta = manifest['metadata']
            if manifest['protocol'] != read(base / 'protocol.json'):
                raise ValueError('Mixed protocols in training data')
            if meta['split'] != 'train' or not meta.get('frozen_risk_model'):
                raise ValueError('Only frozen train collections can be extended')
            for key in ('scenario_id', 'policy_sha256', 'risk_sha256', 'allowed_route_hashes'):
                if meta[key] != original_meta[key]:
                    raise ValueError('Training scene/checkpoints changed: ' + key)
            if meta['seed'] in seeds:
                raise ValueError('Repeated training seed')
            seeds.add(meta['seed'])
        used.update(seeds)
        training[variant] = directories
        evaluation[variant] = [str(base / variant / split) for split in ('validation-0', 'test-0')]
        command = next(c['argv'] for c in summary['commands'] if c['name'] == variant + '-train-0')
        checkpoints[variant] = {name: command[command.index(flag) + 1] for name, flag in (
            ('actor', '--actor'), ('risk_model', '--risk-model'))}
    available = [index for index in range(1, 50) if scenario['seed'] + index not in used]
    if len(available) < replicates:
        raise ValueError('Not enough unused training seeds in the locked protocol')
    return {'base_run': str(base), 'replicates': available[:replicates], 'steps': steps,
            'seeds': [scenario['seed'] + index for index in available[:replicates]],
            'training_collections': training, 'evaluation_collections': evaluation,
            'checkpoints': checkpoints, 'p5_started': False}


def fingerprint(paths):
    result = {}
    for path in paths:
        path = Path(path)
        hasher = hashlib.sha256()
        with path.open('rb') as stream:
            for block in iter(lambda: stream.read(1024 * 1024), b''):
                hasher.update(block)
        result[str(path)] = hasher.hexdigest()
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--base-run', type=Path, default=ROOT / 'plan/p4/runs/20261121T115438612730Z')
    parser.add_argument('--previous-extension', type=Path)
    parser.add_argument('--replicates', type=int, default=3)
    parser.add_argument('--steps', type=int, default=2000)
    parser.add_argument('--plan-only', action='store_true')
    args = parser.parse_args()
    plan = plan_extension(args.base_run, args.replicates, args.steps, args.previous_extension)
    if args.plan_only:
        print(json.dumps(plan, indent=2))
        return 0
    sys.path.insert(0, str(ROOT / 'src/sac_rwb'))
    from algos.horizon_risk import PROTOCOL
    from algos.p4_data import audit_datasets, sha_file, write_reliability
    parent = ROOT / 'plan/p4/runs'
    run = parent / (datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ') + '-train-extension')
    run.mkdir()
    before = source_manifest()
    write_json(run / 'source-manifest.json', before)
    write_json(run / 'plan.json', plan)
    report = dict(plan, mode='target', passed=False, engineering_passed=False,
                  data_ready=False, variants={}, commands=[], p5_started=False)
    base = Path(plan['base_run'])
    try:
        environment = collect_environment(True)
        write_json(run / 'environment.json', environment)
        if environment.get('target_errors'):
            raise ValueError('; '.join(environment['target_errors']))
        if read(base / 'protocol.json') != PROTOCOL or read(ROOT / 'plan/p4/protocol.json') != PROTOCOL:
            raise ValueError('Locked protocol changed')
        for entry in read(base / 'source-manifest.json'):
            data = (ROOT / entry['path']).read_bytes()
            if entry['sha256'] not in (hashlib.sha256(data).hexdigest(),
                                      hashlib.sha256(data.replace(b'\r\n', b'\n')).hexdigest()):
                raise ValueError('Existing engineering source changed: ' + entry['path'])
        eval_paths = [Path(directory) / name for dirs in plan['evaluation_collections'].values()
                      for directory in dirs for name in ('manifest.json', 'counts.json', 'samples.jsonl', 'events.jsonl')]
        frozen = fingerprint(eval_paths)  # Server raw streams are required; never re-collect evaluation.
        write_json(run / 'frozen-evaluation-sha256.json', frozen)
        for variant in ('b0', 'm1'):
            meta = read(base / variant / 'train-0/manifest.json')['metadata']
            checkpoints = plan['checkpoints'][variant]
            if sha_file(checkpoints['actor']) != meta['policy_sha256'] or sha_file(checkpoints['risk_model']) != meta['risk_sha256']:
                raise ValueError('Frozen checkpoints changed')
            for index in plan['replicates']:
                name = '{}-train-{}'.format(variant, index)
                output = run / variant / ('train-' + str(index))
                work = run / (name + '-work')
                work.mkdir()
                subprocess.run(['git', 'check-ignore', '--no-index', '--quiet', str(output / 'samples.jsonl')],
                               cwd=str(ROOT), check=True)
                argv = ['timeout', '50m', sys.executable, '-B', str(ROOT / 'scripts/collect_p4.py'),
                        '--scenarios', str(base / 'scenarios/scenarios.json'), '--split', 'train',
                        '--variant', variant, '--replicate', str(index), '--actor', checkpoints['actor'],
                        '--risk-model', checkpoints['risk_model'], '--steps', str(args.steps), '--output', str(output)]
                print('Collecting {} ({} decisions) ...'.format(name, args.steps), flush=True)
                with (run / (name + '.log')).open('w', encoding='utf-8') as stream:
                    code = subprocess.run(argv, cwd=str(work), stdout=stream, stderr=subprocess.STDOUT).returncode
                (run / (name + '.exitcode')).write_text(str(code) + '\n', encoding='utf-8')
                report['commands'].append({'name': name, 'argv': argv, 'exitcode': code})
                if code:
                    raise RuntimeError('Collection failed: ' + name)
                status = read(output / 'collection.json')
                if not status['completed'] or status['decision_steps'] != args.steps or status['sac_updates'] or status['risk_updates']:
                    raise ValueError('Collection was incomplete or unexpectedly trained a model')
                report['training_collections'][variant].append(str(output))
            audit = audit_datasets(report['training_collections'][variant] + plan['evaluation_collections'][variant])
            original_audit = read(base / (variant + '-audit.json'))
            for split in ('validation', 'test'):
                if audit['splits'][split] != original_audit['splits'][split]:
                    raise ValueError('Frozen evaluation audit changed: ' + split)
            write_json(run / (variant + '-audit.json'), audit)
            write_reliability(audit, run / (variant + '-reliability.svg'))
            metrics = audit['splits']['train']['raw_metrics']
            report['variants'][variant] = {'data_ready': audit['data_ready'],
                                           'positive_windows': metrics['positive_windows'],
                                           'negative_windows': metrics['negative_windows']}
        if fingerprint(eval_paths) != frozen:
            raise ValueError('Existing evaluation files were modified')
        if source_manifest() != before:
            raise ValueError('Sources changed during collection')
        report['engineering_passed'] = True
        report['evaluation_unchanged'] = True
        report['data_ready'] = all(v['data_ready'] for v in report['variants'].values())
        report['passed'] = report['data_ready']
        report['status'] = 'passed' if report['passed'] else 'needs_more_natural_train_events'
    except Exception as error:
        report.update(passed=False, engineering_passed=False, error='{}: {}'.format(type(error).__name__, error))
        print(report['error'], file=sys.stderr)
    finally:
        write_json(run / 'summary.json', report)
        print('Evidence: {}'.format(run), flush=True)
    return 0 if report['passed'] else (2 if report['engineering_passed'] else 1)


if __name__ == '__main__':
    sys.exit(main())
