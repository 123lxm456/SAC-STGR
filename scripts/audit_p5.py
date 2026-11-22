"""Fail-closed P5 evidence audit and training-seed paired aggregation."""

import argparse
import hashlib
import json
from pathlib import Path
import sys
import tarfile
import numpy as np
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src/sac_rwb'))
from algos.p5_experiment import paired_summary, corpus_fingerprint
from algos.p4_data import sha_file, score_metrics, fit_calibration, apply_calibration
from algos.p5_metrics import audit_physical
from accept_p5 import protocol_hash, validate_protocol, read, common_corpus


def require(condition, message):
    if not condition:
        raise ValueError(message)


def within(run, path):
    result = (run / path).resolve()
    result.relative_to(run.resolve())
    return result


def system_references(run, plan):
    """Import original-method evidence, not a reimplementation claim."""
    path = run / 'system-references.json'
    if not path.is_file():
        return {'passed': False, 'missing': plan['protocol']['system_references_required']}
    data = read(path)
    require(data['protocol_sha256'] == plan['protocol_sha256'], 'System-reference protocol mismatch')
    protocol = plan['protocol']
    expected = {(a, s, d, e) for a in protocol['system_references_required']
                for s in protocol['training_seeds'] for d in protocol['evaluation']['densities']
                for e in protocol['evaluation']['seeds']}
    seen = set()
    for method in protocol['system_references_required']:
        record = data['methods'][method]
        for field in ('source_archive', 'method_review'):
            artifact = within(run, record[field]['path'])
            require(artifact.is_file() and sha_file(artifact) == record[field]['sha256'],
                    'Missing original-method source/review artifact')
        require(isinstance(record.get('interface'), str) and bool(record['interface'].strip()) and
                record.get('comparison_scope') == 'system_reference_only',
                'Reference observation/action differences must be documented as system-only comparison')
        require(record['training_decisions'] == protocol['online']['decision_steps'] or method == 'fcfs',
                'Unequal system-reference learning budget')
    for row in data['evaluations']:
        key = (row['method'], row['training_seed'], row['density'], row['evaluation_seed'])
        require(key in expected and key not in seen, 'Unexpected/duplicate system-reference evaluation')
        seen.add(key)
        require(row['scenario_id'] == plan['scenarios']['test-' + row['density']]['scenario_id'],
                'Different system-reference traffic flow')
        physical = within(run, row['physical_path'])
        require(sha_file(physical) == row['physical_sha256'], 'Changed system-reference physical log')
        metrics = audit_physical(physical)
        require(metrics == row['metrics'] and metrics['simulation_steps'] == protocol['evaluation']['simulation_steps'],
                'Invalid system-reference metrics/horizon')
    require(seen == expected, 'Incomplete system-reference matrix')
    return {'passed': True, 'evaluations': len(seen),
            'scope': 'Artifact integrity and common exposure verified; original-method identity additionally requires the archived method review'}


def audit(run, require_full=True):
    run = Path(run).resolve()
    plan = read(run / 'plan.json')
    protocol = plan['protocol']
    require(protocol_hash(protocol) == plan['protocol_sha256'], 'Changed protocol')
    environment = read(run / 'environment.json')
    require(environment.get('target_required') and not environment.get('target_errors') and
            environment.get('cuda_available') and environment.get('gpu') == 'NVIDIA GeForce RTX 4090',
            'Missing locked target environment evidence')
    source_entries = read(run / 'source-manifest.json')
    require(sha_file(run / 'source.tar.gz') == read(run / 'source-snapshot.json')['sha256'], 'Source snapshot changed')
    with tarfile.open(str(run / 'source.tar.gz'), 'r:gz') as archive:
        for entry in source_entries:
            require(hashlib.sha256(archive.extractfile(entry['path']).read()).hexdigest() == entry['sha256'],
                    'Captured source does not match execution manifest')
    units = read(run / 'unit-results.json')
    require(units.get('passed') and units.get('require_cuda') and units.get('tests_run', 0) > 0 and
            not any(units.get(k) for k in ('skipped', 'errors', 'failures')), 'Incomplete target regression')
    require(read(run / 'static.json')['passed'], 'Failed protection/static check')
    commands = read(run / 'commands.json')
    expected_commands = {'unit', 'static'}
    for seed in protocol['training_seeds']:
        expected_commands.update('train-{}-{}'.format(v, seed) for v in protocol['policy_variants'])
        expected_commands.update('offline-{}-{}'.format(v, seed) for v in protocol['risk_variants'])
        expected_commands.update('eval-{}-{}-{}-{}'.format(v, seed, d, e)
                                 for v in protocol['policy_variants'] for d in protocol['evaluation']['densities']
                                 for e in protocol['evaluation']['seeds'])
    require({c['name'] for c in commands} == expected_commands and len(commands) == len(expected_commands),
            'Incomplete/duplicate command matrix')
    for command in commands:
        name = command['name']
        require(command['exitcode'] == 0 and int((run / (name + '.exitcode')).read_text()) == 0 and
                (run / (name + '.log')).is_file(), 'Failed/missing worker execution proof')
    require(corpus_fingerprint(plan['common_corpus']) == plan['common_corpus_sha256'], 'Common corpus changed')
    if require_full:
        validate_protocol(protocol)
        require(protocol == read(ROOT / 'plan/p5/protocol.json'), 'Research budget differs from locked protocol')
        expected_paths, strict_common_audit = common_corpus()
        require([str(p) for p in expected_paths] == plan['common_corpus'] and
                strict_common_audit == read(run / 'common-corpus-audit.json'),
                'Common corpus is not the accepted P4 B0 data')
    online, offline, closed = {}, {}, {}
    missing_learning = []
    expected_labels = {'validation': [], 'test': []}
    for directory in plan['common_corpus']:
        directory = Path(directory)
        split = read(directory / 'manifest.json')['metadata']['split']
        if split in expected_labels:
            with (directory / 'samples.jsonl').open(encoding='utf-8') as stream:
                for line in stream:
                    row = json.loads(line)
                    if row['label'] is not None:
                        expected_labels[split].append(row['label'])
    label_fingerprints = {split: hashlib.sha256(np.asarray(values, np.float32).tobytes()).hexdigest()
                          for split, values in expected_labels.items()}
    for seed in protocol['training_seeds']:
        initial_actors = set()
        for variant in protocol['policy_variants']:
            folder = run / 'online' / variant / str(seed)
            result = read(folder / 'result.json')
            require(result['completed'] and result['initialized_from_scratch'] and result['seed'] == seed and
                    result['variant'] == variant and result['decision_steps'] == protocol['online']['decision_steps'] and
                    result['sac_updates'] > 0 and result['risk_supervision'] == 'p4-h10-v1',
                    'Incomplete online training: variant={}, seed={}, steps={}/{}, SAC updates={}, completed={}'.format(
                        variant, seed, result.get('decision_steps'), protocol['online']['decision_steps'],
                        result.get('sac_updates'), result.get('completed')))
            for key, filename in (('actor_sha256', 'actor.pth'), ('risk_sha256', 'risk.pth'),
                                  ('initial_actor_sha256', 'actor-initial.pth')):
                require(sha_file(folder / filename) == result[key], 'Changed online checkpoint')
            initial_actors.add(result['initial_actor_sha256'])
            if variant == 's0':
                require(result['risk_updates'] == result['risk_update_attempts'] == 0, 'S0 learned a risk penalty')
            elif result['risk_updates'] < 1:
                missing_learning.append([variant, seed])
            require(result['risk_updates'] <= result['risk_update_attempts'] <=
                    protocol['online']['decision_steps'] // protocol['online']['update_every'] *
                    protocol['online']['risk_updates_per_trigger'], 'Risk update counter exceeds locked budget')
            online[variant, seed] = result
            for density in protocol['evaluation']['densities']:
                for evaluation_seed in protocol['evaluation']['seeds']:
                    location = run / 'closed-loop' / variant / str(seed) / density / str(evaluation_seed)
                    value = read(location / 'result.json')
                    require(value['completed'] and value['variant'] == variant and value['evaluation_seed'] == evaluation_seed
                            and value['density'] == density and value['scenario_id'] == plan['scenarios']['test-' + density]['scenario_id']
                            and value['sac_updates'] == value['risk_updates'] == 0, 'Invalid/fitting-during-test evaluation')
                    require(value['actor_sha256'] == result['actor_sha256'] and value['risk_sha256'] == result['risk_sha256'],
                            'Evaluation used different weights')
                    physical = location / 'physical.jsonl'
                    require(sha_file(physical) == value['physical_sha256'], 'Physical log hash changed')
                    require(audit_physical(physical) == value['metrics'] and
                            value['metrics']['simulation_steps'] == protocol['evaluation']['simulation_steps'], 'Physical audit failed')
                    closed[variant, seed, density, evaluation_seed] = value['metrics']
        require(len(initial_actors) == 1, 'SAC initial actors are not paired across variants')
        for variant in protocol['risk_variants']:
            folder = run / 'offline' / variant / str(seed)
            result = read(folder / 'result.json')
            require(result['variant'] == variant and result['seed'] == seed and
                    result['updates'] == protocol['offline']['updates'] and
                    result['batch_size'] == protocol['offline']['batch_size'] and
                    result['common_corpus_sha256'] == plan['common_corpus_sha256'], 'Unequal offline dataset/budget')
            require(sha_file(folder / 'risk.pth') == result['checkpoint_sha256'], 'Changed offline checkpoint')
            with np.load(str(folder / 'validation-predictions.npz')) as validation:
                calibration = fit_calibration(validation['labels'], validation['scores'])
            require(calibration == result['calibration'], 'Calibration is not validation-only')
            for split in ('validation', 'test'):
                with np.load(str(folder / (split + '-predictions.npz'))) as saved:
                    labels, scores = saved['labels'], saved['scores']
                fingerprint = hashlib.sha256(labels.tobytes()).hexdigest()
                require(label_fingerprints[split] == fingerprint,
                        'Offline models evaluated on different labels/order')
                require(score_metrics(labels, scores) == result['splits'][split]['raw'] and
                        score_metrics(labels, apply_calibration(scores, calibration)) == result['splits'][split]['calibrated'],
                        'Offline metrics disagree with saved predictions')
            offline[variant, seed] = result
    if require_full:
        seed = protocol['training_seeds'][0]
        counts = {v: offline[v, seed]['engineering']['parameter_count'] for v in ('m1', 'c1')}
        require(abs(counts['c1'] / counts['m1'] - 1) <= protocol['c1_max_parameter_relative_difference'],
                'C1 capacity is not matched to M1')
    aggregate = {'offline': {}, 'closed_loop': {}, 'paired_m1_minus_comparator': {}}
    if len(protocol['training_seeds']) >= 3:
        for variant in protocol['risk_variants']:
            aggregate['offline'][variant] = {metric: paired_summary([
                offline[variant, seed]['splits']['test']['raw'][metric] for seed in protocol['training_seeds']])
                for metric in ('auprc_average_precision', 'auroc', 'brier', 'ece', 'false_negative_rate')}
        for variant in protocol['policy_variants']:
            aggregate['closed_loop'][variant] = {}
            for density in protocol['evaluation']['densities']:
                aggregate['closed_loop'][variant][density] = {metric: paired_summary([
                    np.mean([closed[variant, seed, density, e][metric] for e in protocol['evaluation']['seeds']])
                    for seed in protocol['training_seeds']]) for metric in
                    ('collision_events_per_1000_departed', 'completion_fraction', 'aql', 'awt', 'mean_speed')}
        for comparator in protocol['policy_variants']:
            if comparator == 'm1':
                continue
            differences = {}
            if comparator in protocol['risk_variants']:
                differences['offline'] = {metric: paired_summary([
                    offline['m1', seed]['splits']['test']['raw'][metric] - offline[comparator, seed]['splits']['test']['raw'][metric]
                    for seed in protocol['training_seeds']]) for metric in ('auprc_average_precision', 'brier')}
            differences['closed_loop'] = {density: {metric: paired_summary([
                np.mean([closed['m1', seed, density, e][metric] - closed[comparator, seed, density, e][metric]
                         for e in protocol['evaluation']['seeds']]) for seed in protocol['training_seeds']])
                for metric in ('collision_events_per_1000_departed', 'aql', 'awt', 'completion_fraction')}
                for density in protocol['evaluation']['densities']}
            aggregate['paired_m1_minus_comparator'][comparator] = differences
    references = system_references(run, plan) if require_full else {'passed': False, 'scope': 'smoke'}
    passed = bool(require_full and protocol['research_run'] and not missing_learning and references['passed'])
    return dict(passed=passed, status='passed' if passed else ('smoke_passed_not_research' if not require_full else 'needs_research_evidence'),
                primary_matrix_integrity_passed=True, missing_natural_risk_updates=missing_learning,
                system_references=references, aggregate=aggregate,
                scope='Fixed-budget final checkpoints, common offline corpus, paired training seeds; effect direction does not control acceptance')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('run', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    from algos.p5_experiment import write_json
    verdict = audit(args.run)
    write_json(args.output, verdict)
    print(json.dumps({'passed': verdict['passed'], 'status': verdict['status']}))
    sys.exit(0 if verdict['passed'] else 2)
