"""P4 scenario separation, immutable datasets and independent score audits."""

import hashlib
import json
from pathlib import Path
import xml.etree.ElementTree as ET

import numpy as np

from algos.horizon_risk import PROTOCOL, route_hash


SPLITS = ('train', 'validation', 'test')
SEEDS = {'train': 100, 'validation': 200, 'test': 300}


def sha_file(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def create_scenarios(source, output):
    """Partition *complete route edge sequences*, never adjacent windows.

    Retain original departures and demand within each subset. Shared road
    infrastructure is allowed; demand scenes, routes and seeds are disjoint.
    Each split receives a controllable flow, avoiding right-turn-only scenes.
    """
    source, output = Path(source), Path(output)
    output.mkdir(parents=True, exist_ok=False)
    root = ET.parse(str(source / 'newroute1220_start0.xml')).getroot()
    named = {x.get('id'): tuple(x.get('edges').split()) for x in root.findall('route')}
    anchors = {'topstraight': 'train', 'topleft': 'validation', 'rightleft': 'test'}
    if not set(anchors).issubset(named):
        raise ValueError('Scenario flow anchors are missing from source routes')
    assigned = {route_hash(named[name]): split for name, split in anchors.items()}

    def edges_for(element):
        if element.get('route'):
            return named[element.get('route')]
        inline = element.find('route')
        if inline is None or not inline.get('edges'):
            raise ValueError('Every demand element must have an explicit route')
        return tuple(inline.get('edges').split())

    groups = {split: [] for split in SPLITS}
    hashes = {split: set() for split in SPLITS}
    for element in root:
        if element.tag not in ('vehicle', 'flow'):
            if element.tag not in ('route', 'vType'):
                raise ValueError('Unsupported demand element: ' + element.tag)
            continue
        key = route_hash(edges_for(element))
        split = assigned.setdefault(key, SPLITS[int(key, 16) % 3])
        groups[split].append(element)
        hashes[split].add(key)
    result = {'protocol': PROTOCOL, 'source_route_sha256': sha_file(source / 'newroute1220_start0.xml'),
              'separation': 'route edge sequence, demand scene, seed and physical vehicle; shared road network',
              'scenarios': {}}
    for split in SPLITS:
        directory = output / split
        directory.mkdir()
        scene = ET.Element('routes')
        scene.extend(root.findall('vType'))
        used = {e.get('route') for e in groups[split] if e.get('route')}
        scene.extend(e for e in root.findall('route') if e.get('id') in used)
        scene.extend(groups[split])
        route_path = directory / 'routes.xml'
        ET.ElementTree(scene).write(str(route_path), encoding='utf-8', xml_declaration=True)
        config = ET.parse(str(source / 'osm.sumocfg'))
        config.find('./input/net-file').set('value', str((source / 'CSeditClean_1.net_threelegs.xml').resolve()))
        config.find('./input/route-files').set('value', str(route_path.resolve()))
        for node in list(config.getroot()):
            if node.tag in ('output', 'gui_only'):
                config.getroot().remove(node)
        config.write(str(directory / 'osm.sumocfg'), encoding='utf-8', xml_declaration=True)
        result['scenarios'][split] = {
            'split': split, 'seed': SEEDS[split], 'scenario_id': sha_file(route_path),
            'route_hashes': sorted(hashes[split]), 'demand_elements': len(groups[split]),
            'config': str((directory / 'osm.sumocfg').resolve()),
            'network': str((source / 'CSeditClean_1.net_threelegs.xml').resolve()),
        }
    (output / 'scenarios.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
    return result


def _read_jsonl(path):
    # Avoid holding a complete JSONL string and split-line copies alongside the
    # decoded tensors. Large real datasets remain on the experiment server.
    with path.open(encoding='utf-8') as stream:
        return [json.loads(line) for line in stream]


def read_dataset(directory):
    directory = Path(directory)
    manifest = json.loads((directory / 'manifest.json').read_text(encoding='utf-8'))
    if manifest['protocol'] != PROTOCOL:
        raise ValueError('Mixed or modified label protocols')
    rows = _read_jsonl(directory / 'samples.jsonl')
    events = _read_jsonl(directory / 'events.jsonl')
    steps = {}
    by_episode = {}
    committed = set()
    for event in events:
        if type(event['episode']) is not int or event['episode'] < 0 or type(event['step']) is not int or event['step'] < 1:
            raise ValueError('Simulation episode/step must be nonnegative integer indices')
        key = (event['episode'], event['step'])
        if key in steps:
            raise ValueError('Duplicate simulation event step')
        prior = by_episode.get(event['episode'])
        if prior is not None and event['step'] != prior + 1:
            raise ValueError('Unobserved simulation gap')
        steps[key] = event
        by_episode[event['episode']] = event['step']
        commands = event.get('commands', {})
        if not isinstance(commands, dict):
            raise ValueError('Physical command log must map vehicle IDs to actions')
        for vehicle, action in commands.items():
            if not isinstance(vehicle, str) or not vehicle or type(action) not in (int, float) or not np.isfinite(action) or not -3 <= action <= 3:
                raise ValueError('Physical command must be finite and bounded in [-3,3]')
            committed.add((event['episode'], vehicle, event['step'] - 1))
    seen = set()
    for row in rows:
        if type(row['episode']) is not int or row['episode'] < 0 or type(row['decision_step']) is not int or row['decision_step'] < 0:
            raise ValueError('Decision episode/step must be nonnegative integer indices')
        key = (row['episode'], row['vehicle_id'], row['decision_step'])
        if key in seen:
            raise ValueError('Duplicate decision window')
        seen.add(key)
        states, actions, mask = (np.asarray(row[name]) for name in ('state_seq', 'action_seq', 'valid_mask'))
        if states.shape != (10, 98) or actions.shape != (10, 1) or mask.shape != (10,):
            raise ValueError('Invalid stored tensor shapes')
        if mask.dtype != np.bool_ or not mask[-1] or np.any(mask[:-1] & ~mask[1:]):
            raise ValueError('Invalid left padding mask')
        if not np.isfinite(states).all() or not np.isfinite(actions).all() or np.abs(actions).max() > 3:
            raise ValueError('Invalid recorded state/action values')
        if np.any(states[~mask]) or np.any(actions[~mask]):
            raise ValueError('Padding must match online zero padding')
        if row['executed'] is not True or row['horizon_steps'] != PROTOCOL['horizon_steps']:
            raise ValueError('Unexecuted or inconsistent window')
        if row['label'] is not None and (type(row['label']) is not int or row['label'] not in (0, 1)):
            raise ValueError('Window label must be integer 0/1 or censored null')
        if not np.isfinite(row['score']) or not 0 <= row['score'] <= 1:
            raise ValueError('Invalid recorded score')
        episode, start, vehicle = row['episode'], row['decision_step'], row['vehicle_id']
        final = by_episode.get(episode)
        if final is None or (episode, start + 1) not in steps:
            raise ValueError('Decision lacks execution event')
        sent = steps[(episode, start + 1)].get('commands', {}).get(vehicle)
        if sent is None or not np.isclose(sent, actions[-1, 0], atol=1e-6, rtol=0) or row.get('executed_action') != sent:
            raise ValueError('Window action disagrees with the actual sent-command log')
        expected = None
        for step in range(start + 1, min(start + PROTOCOL['horizon_steps'], final) + 1):
            event = steps[(episode, step)]
            if vehicle in event['collided']:
                expected = (1, 'collision', step)
            elif vehicle in event['arrived']:
                expected = (0, 'arrival', step)
            elif vehicle in event['absent']:
                expected = (None, 'missing_vehicle', step)
            elif step == start + PROTOCOL['horizon_steps']:
                expected = (0, 'horizon_observed', step)
            if expected is not None:
                break
        if expected is None:
            if row['label'] is not None or row['end_step'] != final or row['reason'] not in (
                    'episode_reset', 'collection_end', 'collection_interrupted'):
                raise ValueError('Incomplete future must be explicitly censored')
        elif (row['label'], row['reason'], row['end_step']) != expected:
            raise ValueError('Stored label disagrees with raw physical events')
    counts = json.loads((directory / 'counts.json').read_text(encoding='utf-8'))
    if counts.get('executed_decisions', 0) != len(rows) or counts.get('windows', 0) != len(rows):
        raise ValueError('Unaccounted decisions or incomplete dataset closure')
    if any(type(value) is not int or value < 0 for value in counts.values()):
        raise ValueError('Dataset counters must be nonnegative integers')
    observed = {'positive': sum(row['label'] == 1 for row in rows),
                'negative': sum(row['label'] == 0 for row in rows),
                'censored': sum(row['label'] is None for row in rows)}
    if any(counts.get(name, 0) != value for name, value in observed.items()):
        raise ValueError('Dataset class counters disagree with stored labels')
    if seen != committed:
        raise ValueError('Committed physical commands and stored windows must correspond one-to-one')
    if not len(committed) <= counts.get('commands_sent', 0) <= len(committed) + counts.get('unexecuted_discarded', 0):
        raise ValueError('Sent-command counter disagrees with committed/discarded decisions')
    return manifest, rows, events


def score_metrics(labels, probabilities, threshold=0.5):
    y, p = np.asarray(labels, dtype=float), np.asarray(probabilities, dtype=float)
    if y.shape != p.shape or y.ndim != 1 or not len(y) or not np.isin(y, (0, 1)).all():
        raise ValueError('Nonempty binary labels with corresponding scores required')
    if not np.isfinite(p).all() or np.any((p < 0) | (p > 1)):
        raise ValueError('Probabilities must be finite and bounded')
    pos, neg = int(y.sum()), int(len(y) - y.sum())
    # Group ties: metrics must not depend on order inside tied scores.
    order = np.argsort(-p, kind='mergesort')
    yy, pp = y[order], p[order]
    ends = np.r_[np.where(pp[:-1] != pp[1:])[0], len(p) - 1]
    tp = np.cumsum(yy)[ends]
    fp = (ends + 1) - tp
    ap = float(np.sum(np.diff(np.r_[0, tp]) / pos * tp / (ends + 1))) if pos else None
    if pos and neg:
        tpr, fpr = np.r_[0, tp / pos], np.r_[0, fp / neg]
        auc = float(np.sum(np.diff(fpr) * (tpr[:-1] + tpr[1:]) / 2))
    else:
        auc = None
    bins, ece = [], 0.0
    indices = np.minimum((p * PROTOCOL['calibration_bins']).astype(int), PROTOCOL['calibration_bins'] - 1)
    for index in range(PROTOCOL['calibration_bins']):
        select = indices == index
        count = int(select.sum())
        confidence, rate = (float(p[select].mean()), float(y[select].mean())) if count else (None, None)
        if count:
            ece += count / len(y) * abs(confidence - rate)
        bins.append({'bin': index, 'count': count, 'mean_score': confidence, 'event_rate': rate})
    return {'samples': len(y), 'positive_windows': pos, 'negative_windows': neg,
            'prevalence': pos / len(y), 'auprc_average_precision': ap, 'auroc': auc,
            'brier': float(np.mean((p - y) ** 2)), 'ece': ece,
            'false_negatives': int(((y == 1) & (p < threshold)).sum()),
            'positive_denominator': pos, 'threshold': threshold,
            'false_negative_rate': float(((y == 1) & (p < threshold)).sum() / pos) if pos else None,
            'reliability_bins': bins,
            'undefined_reason': None if pos and neg else 'Both classes required for discrimination'}


def apply_calibration(scores, calibration):
    logits = np.log(np.clip(scores, 1e-7, 1 - 1e-7) / (1 - np.clip(scores, 1e-7, 1 - 1e-7)))
    z = np.clip(calibration['slope'] * logits + calibration['intercept'], -40, 40)
    return 1 / (1 + np.exp(-z))


def fit_calibration(validation_labels, validation_scores):
    """Validation-only regularized Platt fit; never fit on test labels."""
    y, p = np.asarray(validation_labels, float), np.asarray(validation_scores, float)
    score_metrics(y, p)
    if len(set(y)) != 2:
        raise ValueError('Calibration requires natural positive and negative validation outcomes')
    x = np.column_stack((np.log(np.clip(p, 1e-7, 1 - 1e-7) / (1 - np.clip(p, 1e-7, 1 - 1e-7))), np.ones(len(p))))
    beta = np.array([1.0, 0.0])
    penalty = 1e-4
    def objective(b):
        z = x.dot(b)
        return float(np.mean(np.logaddexp(0, z) - y * z) + penalty * np.sum(b * b) / 2)
    for _ in range(100):
        q = 1 / (1 + np.exp(-np.clip(x.dot(beta), -40, 40)))
        gradient = x.T.dot(q - y) / len(y) + penalty * beta
        hessian = x.T.dot((q * (1 - q))[:, None] * x) / len(y) + penalty * np.eye(2)
        delta = np.linalg.solve(hessian, gradient)
        scale = 1.0
        while scale > 1e-8 and objective(beta - scale * delta) > objective(beta):
            scale *= 0.5
        beta -= scale * delta
        if np.max(np.abs(scale * delta)) < 1e-8:
            break
    if not np.isfinite(beta).all():
        raise ValueError('Calibration failed to converge to finite parameters')
    return {'method': 'validation-only regularized Platt', 'slope': float(beta[0]),
            'intercept': float(beta[1]), 'regularization': penalty, 'fit_samples': len(y)}


def write_reliability(report, path):
    """Dependency-free SVG diagnostic from the audited natural-rate bins."""
    svg = ['<svg xmlns="http://www.w3.org/2000/svg" width="640" height="500" viewBox="0 0 640 500">',
           '<rect width="640" height="500" fill="white"/>',
           '<path d="M60 30 V430 H460 M60 430 L460 30" fill="none" stroke="#999"/>',
           '<text x="160" y="478" font-size="14">Mean predicted score</text>',
           '<text x="60" y="20" font-size="14">Observed event fraction (window denominator)</text>']
    for tick in range(6):
        value = tick / 5
        svg.append('<text x="{}" y="450" font-size="12">{:.1f}</text>'.format(55 + value * 400, value))
        svg.append('<text x="25" y="{}" font-size="12">{:.1f}</text>'.format(435 - value * 400, value))
    series = []
    for split in ('validation', 'test'):
        raw = report['splits'][split]['raw_metrics']
        if raw is not None:
            series.append((split + ' raw', raw))
        calibrated = report['calibrated_metrics'].get(split)
        if calibrated is not None:
            series.append((split + ' calibrated', calibrated))
    for index, (name, metrics) in enumerate(series):
        color = ('#0072B2', '#009E73', '#D55E00', '#CC79A7')[index]
        points = [(60 + x['mean_score'] * 400, 430 - x['event_rate'] * 400)
                  for x in metrics['reliability_bins'] if x['count']]
        svg.append('<polyline points="{}" fill="none" stroke="{}"/>'.format(
            ' '.join('{:.2f},{:.2f}'.format(x, y) for x, y in points), color))
        svg.extend('<circle cx="{:.2f}" cy="{:.2f}" r="4" fill="{}"/>'.format(x, y, color) for x, y in points)
        svg.append('<text x="475" y="{}" fill="{}" font-size="12">{}</text>'.format(60 + 25 * index, color, name))
    svg.append('<text x="60" y="495" font-size="11">P4 diagnostic only; validation fits calibration; correlated windows; no P5 claim.</text></svg>')
    Path(path).write_text('\n'.join(svg), encoding='utf-8')


def audit_datasets(directories):
    datasets = [read_dataset(path) for path in directories]
    memberships = {key: {} for key in ('vehicle', 'route', 'scenario', 'seed')}
    partitions = {split: [] for split in SPLITS}
    model_hashes, policy_hashes = set(), set()
    unique_events = set()
    run_ids = set()
    collection_ids = set()
    split_reports = {}
    for manifest, rows, events in datasets:
        meta = manifest['metadata']
        if meta['run_id'] in run_ids:
            raise ValueError('Duplicate collection run would count windows twice')
        run_ids.add(meta['run_id'])
        split = meta['split']
        if not meta.get('frozen_risk_model'):
            raise ValueError('Independent calibration requires a frozen risk model')
        model_hashes.add(meta.get('risk_sha256'))
        policy_hashes.add(meta.get('policy_sha256'))
        identity = (meta['seed'], meta['scenario_id'], meta.get('policy_sha256'), meta.get('risk_sha256'))
        if identity in collection_ids:
            raise ValueError('Repeated deterministic seed/scenario/checkpoints would duplicate trajectories')
        collection_ids.add(identity)
        if not rows:
            raise ValueError('Empty partition collection: ' + split)
        partitions[split].extend(rows)
        for row in rows:
            row['_audit_run_id'] = meta['run_id']
            if row['route_hash'] not in meta['allowed_route_hashes']:
                raise ValueError('Observed route not in locked demand scene')
        groups = {'vehicle': {row['vehicle_id'] for row in rows},
                  'route': set(meta['allowed_route_hashes']),
                  'scenario': {meta['scenario_id']}, 'seed': {meta['seed']}}
        for kind, values in groups.items():
            for value in values:
                previous = memberships[kind].setdefault(value, split)
                if previous != split:
                    raise ValueError('Cross-split {} leakage: {}'.format(kind, value))
        for event in events:
            for vehicle in event['collided']:
                unique_events.add((meta['run_id'], event['episode'], event['step'], vehicle))
    if len(model_hashes) != 1 or None in model_hashes or len(policy_hashes) != 1 or None in policy_hashes:
        raise ValueError('All independent splits must share frozen risk and behavior checkpoint hashes')
    for split, rows in partitions.items():
        if not rows:
            raise ValueError('Missing split: ' + split)
        labeled = [row for row in rows if row['label'] is not None]
        actions = np.array([row['executed_action'] for row in rows])
        split_reports[split] = {'windows': len(rows), 'censored': len(rows) - len(labeled),
                                'vehicles': len({(row['_audit_run_id'], row['episode'], row['vehicle_id']) for row in rows}),
                                'positive_vehicles': len({(row['_audit_run_id'], row['episode'], row['vehicle_id']) for row in labeled if row['label'] == 1}),
                                'positive_collision_vehicle_events': len({
                                    (row['_audit_run_id'], row['episode'], row['vehicle_id'], row['end_step'])
                                    for row in labeled if row['label'] == 1}),
                                'action': {'count': len(actions), 'mean': float(actions.mean()),
                                           'min': float(actions.min()), 'max': float(actions.max()),
                                           'quantiles': np.quantile(actions, [0, .25, .5, .75, 1]).tolist(),
                                           'histogram_edges': np.linspace(-3, 3, 13).tolist(),
                                           'histogram_counts': np.histogram(actions, np.linspace(-3, 3, 13))[0].tolist()},
                                'raw_metrics': score_metrics([x['label'] for x in labeled], [x['score'] for x in labeled]) if labeled else None}
    ready = all(split_reports[split]['raw_metrics'] is not None and
                split_reports[split]['raw_metrics']['positive_windows'] >= PROTOCOL['minimum_evaluation_positives'] and
                split_reports[split]['raw_metrics']['negative_windows'] >= PROTOCOL['minimum_evaluation_negatives'] and
                split_reports[split]['positive_vehicles'] >= PROTOCOL['minimum_evaluation_collision_vehicles']
                for split in ('validation', 'test'))
    train_metrics = split_reports['train']['raw_metrics']
    training_ready = (train_metrics is not None and
                      train_metrics['positive_windows'] >= PROTOCOL['minimum_training_positives'] and
                      train_metrics['negative_windows'] >= PROTOCOL['minimum_training_negatives'])
    calibration, calibrated = None, {}
    val = [row for row in partitions['validation'] if row['label'] is not None]
    if ready:
        calibration = fit_calibration([x['label'] for x in val], [x['score'] for x in val])
        for split in ('validation', 'test'):
            rows = [row for row in partitions[split] if row['label'] is not None]
            calibrated[split] = score_metrics([x['label'] for x in rows], apply_calibration([x['score'] for x in rows], calibration))
    return {'protocol': PROTOCOL, 'isolation_passed': True, 'splits': split_reports,
            'physical_collision_vehicle_events': len(unique_events),
            'collision_event_scope': 'all raw SUMO vehicle/step events; split positive events count only scored vehicles',
            'calibration': calibration, 'calibrated_metrics': calibrated,
            'training_data_ready': training_ready,
            'calibration_data_ready': ready,
            'data_ready': ready and training_ready,
            'insufficiency': None if ready and training_ready else 'Train needs >=64 positive/negative windows; validation/test each need >=20 positive/negative windows and >=5 positive vehicles',
            'scope': 'P4 data and calibration diagnostic only; correlated windows are not independent events; no P5 comparison',
            'p5_started': False}
