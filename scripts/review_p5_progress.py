"""Read-only review of partial P5 snapshots, never full research acceptance.

Reuses P4 prediction/calibration and P5 physical metrics. Raw online/P4 JSONL
may remain on server; missing execution proof is pending, not completion.
"""

import argparse
import hashlib
import json
from pathlib import Path
import sys
import tarfile
import zipfile
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src/sac_rwb'))
from algos.p4_data import sha_file, score_metrics, fit_calibration, apply_calibration
from algos.p5_metrics import audit_physical
from accept_p5 import protocol_hash, read, smoke_protocol


def require(value, message):
    if not value:
        raise ValueError(message)


def equivalent(left, right):
    if isinstance(left, bool) or isinstance(right, bool):
        return type(left) is type(right) and left == right
    if isinstance(left, int) and isinstance(right, int):
        return left == right
    if isinstance(left, dict) and isinstance(right, dict):
        return left.keys() == right.keys() and all(equivalent(left[k], right[k]) for k in left)
    if isinstance(left, list) and isinstance(right, list):
        return len(left) == len(right) and all(equivalent(a, b) for a, b in zip(left, right))
    if isinstance(left, (int, float)) and isinstance(right, (int, float)):
        return bool(np.isclose(left, right, rtol=1e-7, atol=1e-9))
    return left == right


def physical_digest(path, expected):
    data = Path(path).read_bytes()
    if hashlib.sha256(data).hexdigest() == expected:
        return 'original-bytes'
    require(hashlib.sha256(data.replace(b'\r\n', b'\n')).hexdigest() == expected,
            'Physical log differs beyond Git CRLF conversion')
    return 'Git-CRLF-copy; original LF hash matched'


def review(run):
    run = Path(run).resolve()
    plan = read(run / 'plan.json')
    protocol = plan['protocol']
    require(protocol_hash(protocol) == plan['protocol_sha256'], 'Protocol hash mismatch')
    locked = read(ROOT / 'plan/p5/protocol.json')
    require(protocol == (locked if protocol['research_run'] else smoke_protocol(locked)), 'Protocol differs from locked budget')
    # Stat on a deeply missing Windows UNC path may raise WinError 65 rather
    # than return False. Enumerate existing results, without creating folders.
    present_results = set(run.rglob('result.json'))
    units, env = read(run / 'unit-results.json'), read(run / 'environment.json')
    require(units['passed'] and units['require_cuda'] and units['tests_run'] > 0 and
            not any(units.get(k) for k in ('skipped', 'errors', 'failures')), 'Target regression incomplete')
    require(env.get('target_required') and env.get('cuda_available') and not env.get('target_errors') and
            env.get('gpu') == 'NVIDIA GeForce RTX 4090', 'Target environment not accepted')
    require(read(run / 'static.json')['passed'], 'Protection/static failure')
    for name in ('unit', 'static'):
        require(int((run / (name + '.exitcode')).read_text()) == 0, 'Preflight exitcode failure')
    require(sha_file(run / 'source.tar.gz') == read(run / 'source-snapshot.json')['sha256'], 'Source archive hash mismatch')
    runtime_deltas = []
    with tarfile.open(str(run / 'source.tar.gz'), 'r:gz') as archive:
        for entry in read(run / 'source-manifest.json'):
            saved = archive.extractfile(entry['path']).read()
            require(hashlib.sha256(saved).hexdigest() == entry['sha256'], 'Captured source hash mismatch')
            current = ROOT / entry['path']
            if entry['path'].startswith('src/') and (not current.is_file() or
                    saved.replace(b'\r\n', b'\n') != current.read_bytes().replace(b'\r\n', b'\n')):
                runtime_deltas.append(entry['path'])
    require(not runtime_deltas, 'Runtime source differs from captured target source: ' + str(runtime_deltas))
    commands = {row['name']: row for row in read(run / 'commands.json')}

    def execution(name):
        path = run / (name + '.exitcode')
        if name not in commands or not path.is_file():
            return False
        require(commands[name]['exitcode'] == int(path.read_text()) == 0 and
                (run / (name + '.log')).is_file(), 'Failed command: ' + name)
        return True

    online, offline, closed, pending, actors, weights = [], [], [], [], {}, {}
    line_endings = []
    for seed in protocol['training_seeds']:
        for variant in protocol['policy_variants']:
            folder = run / 'online' / variant / str(seed)
            if folder / 'result.json' not in present_results:
                continue
            result = read(folder / 'result.json')
            require(result['completed'] and result['initialized_from_scratch'] and result['seed'] == seed and
                    result['variant'] == variant and result['decision_steps'] == protocol['online']['decision_steps'] and
                    result['sac_updates'] > 0 and result['risk_supervision'] == 'p4-h10-v1', 'Invalid training result')
            for key, name in (('actor_sha256', 'actor.pth'), ('initial_actor_sha256', 'actor-initial.pth'),
                              ('risk_sha256', 'risk.pth')):
                require(sha_file(folder / name) == result[key], 'Checkpoint hash mismatch')
                with zipfile.ZipFile(str(folder / name)) as archive:
                    require(archive.testzip() is None, 'Checkpoint CRC mismatch')
            require(result['actor_sha256'] != result['initial_actor_sha256'], 'Actor did not change')
            require(actors.setdefault(seed, result['initial_actor_sha256']) == result['initial_actor_sha256'],
                    'Initial actors not paired')
            if variant == 's0':
                require(result['risk_updates'] == result['risk_update_attempts'] == 0, 'S0 optimized risk')
            require(result['risk_updates'] <= result['risk_update_attempts'] <=
                    protocol['online']['decision_steps'] // protocol['online']['update_every'] *
                    protocol['online']['risk_updates_per_trigger'], 'Risk update budget mismatch')
            counts = read(folder / 'online-data/counts.json')
            require(counts['windows'] == counts['executed_decisions'] == result['inference']['vehicles_scored'] ==
                    sum(counts.get(k, 0) for k in ('positive', 'negative', 'censored')), 'Online counters mismatch')
            require(result['inference']['calls'] == result['decision_steps'], 'Scoring calls mismatch')
            name = 'train-{}-{}'.format(variant, seed)
            confirmed = execution(name)
            if not confirmed:
                pending.append(name)
            weights[variant, seed] = result
            online.append(dict(variant=variant, seed=seed, decision_steps=result['decision_steps'],
                               sac_updates=result['sac_updates'], risk_updates=result['risk_updates'],
                               risk_update_attempts=result['risk_update_attempts'],
                               positive_windows=counts.get('positive', 0), windows=counts['windows'],
                               execution_confirmed=confirmed,
                               raw_online_streams_present=all((folder / 'online-data' / p).is_file() for p in ('samples.jsonl', 'events.jsonl'))))
        for variant in protocol['risk_variants']:
            folder = run / 'offline' / variant / str(seed)
            if folder / 'result.json' not in present_results:
                continue
            value = read(folder / 'result.json')
            require(value['updates'] == protocol['offline']['updates'] and
                    value['common_corpus_sha256'] == plan['common_corpus_sha256'] and
                    sha_file(folder / 'risk.pth') == value['checkpoint_sha256'], 'Offline budget/checkpoint mismatch')
            with np.load(str(folder / 'validation-predictions.npz')) as data:
                calibration = fit_calibration(data['labels'], data['scores'])
            require(equivalent(calibration, value['calibration']), 'Validation calibration mismatch')
            for split in ('validation', 'test'):
                with np.load(str(folder / (split + '-predictions.npz'))) as data:
                    require(equivalent(score_metrics(data['labels'], data['scores']), value['splits'][split]['raw']) and
                            equivalent(score_metrics(data['labels'], apply_calibration(data['scores'], calibration)),
                                       value['splits'][split]['calibrated']), 'Offline metric mismatch')
            require(execution('offline-{}-{}'.format(variant, seed)), 'Offline execution proof missing')
            offline.append(dict(variant=variant, seed=seed, updates=value['updates']))
        for variant in protocol['policy_variants']:
            for density in protocol['evaluation']['densities']:
                for evaluation_seed in protocol['evaluation']['seeds']:
                    folder = run / 'closed-loop' / variant / str(seed) / density / str(evaluation_seed)
                    if folder / 'result.json' not in present_results:
                        continue
                    value = read(folder / 'result.json')
                    model = weights[variant, seed]
                    require(value['sac_updates'] == value['risk_updates'] == 0 and
                            value['actor_sha256'] == model['actor_sha256'] and value['risk_sha256'] == model['risk_sha256'],
                            'Frozen evaluation mismatch')
                    mode = physical_digest(folder / 'physical.jsonl', value['physical_sha256'])
                    if mode != 'original-bytes':
                        line_endings.append(dict(path=(folder / 'physical.jsonl').relative_to(run).as_posix(), mode=mode))
                    metrics = audit_physical(folder / 'physical.jsonl')
                    require(equivalent(metrics, value['metrics']) and
                            metrics['simulation_steps'] == protocol['evaluation']['simulation_steps'], 'Physical metrics mismatch')
                    require(execution('eval-{}-{}-{}-{}'.format(variant, seed, density, evaluation_seed)), 'Evaluation exitcode missing')
                    closed.append(dict(variant=variant, seed=seed, density=density, evaluation_seed=evaluation_seed))
    return dict(run=run.name, snapshot_checks_passed=True, p5_accepted=False,
                target_tests=units['tests_run'], online=online, offline=offline, closed_loop=closed,
                expected=dict(online=len(protocol['training_seeds']) * len(protocol['policy_variants']),
                              offline=len(protocol['training_seeds']) * len(protocol['risk_variants']),
                              closed_loop=len(protocol['training_seeds']) * len(protocol['policy_variants']) *
                              len(protocol['evaluation']['densities']) * len(protocol['evaluation']['seeds'])),
                pending_command_proofs=pending, final_summary_present=(run / 'summary.json').is_file(),
                system_references_present=(run / 'system-references.json').is_file(),
                local_line_ending_copies=line_endings,
                scope='Partial local snapshot; no server liveness assertion, no raw P4/online stream re-audit; use audit_p5 on server for full acceptance')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('runs', nargs='+', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    reports = [review(path) for path in args.runs]
    with args.output.open('x', encoding='utf-8') as stream:
        json.dump(reports, stream, ensure_ascii=False, indent=2)
    print(json.dumps([dict(run=r['run'], checked=True, online=len(r['online']), offline=len(r['offline']),
                           closed_loop=len(r['closed_loop']), pending=r['pending_command_proofs']) for r in reports]))
