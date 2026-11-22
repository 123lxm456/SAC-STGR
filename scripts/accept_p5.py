"""Immutable P5 runner. No SSH. --local and --smoke never grant acceptance.

Exit 0: local auxiliary checks or full target acceptance. Exit 2: smoke or
missing research evidence. Exit 1: failed execution/integrity check.
"""

import argparse
import copy
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import tarfile
import xml.etree.ElementTree as ET
from accept_p012 import ROOT, collect_environment, source_manifest, write_json


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def protocol_hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def smoke_protocol(locked):
    protocol = copy.deepcopy(locked)
    protocol.update(research_run=False, training_seeds=[11], risk_variants=['b0', 'm1'],
                    policy_variants=['b0', 'm1', 's0'])
    # A sparse P4 train scene produced only 160 transitions in 100 decisions.
    # Keep SAC batch=256 and the real-update gate; provide sufficient exposure.
    protocol['online'].update(decision_steps=1000, episode_decisions=1000)
    protocol['offline']['updates'] = 2
    protocol['evaluation'].update(simulation_steps=100, seeds=[3000], densities={'nominal': 1.0})
    return protocol


def validate_protocol(protocol):
    from algos.p5_models import RISK_VARIANTS, VARIANTS
    if tuple(protocol['risk_variants']) != RISK_VARIANTS or tuple(protocol['policy_variants']) != VARIANTS:
        raise ValueError('Missing or reordered P5 comparison groups')
    seeds = protocol['training_seeds']
    if len(seeds) < 3 or len(set(seeds)) != len(seeds):
        raise ValueError('At least three distinct paired training seeds required')
    if set(seeds) & set(protocol['evaluation']['seeds']):
        raise ValueError('Training/test seeds must be disjoint')
    if len(set(protocol['evaluation']['seeds'])) != len(protocol['evaluation']['seeds']):
        raise ValueError('Repeated evaluation seed')
    if protocol['label_protocol'] != 'p4-h10-v1' or not protocol['research_run']:
        raise ValueError('Locked P4 labels and research protocol required')
    if protocol['offline']['updates'] < 1 or protocol['offline']['batch_size'] != 128:
        raise ValueError('Invalid offline optimization budget')
    if any(type(v) is not int or v < 1 for v in protocol['online'].values()):
        raise ValueError('Invalid online optimization budget')
    if set(protocol['evaluation']['densities']) != {'low', 'nominal', 'high'}:
        raise ValueError('All robustness density strata required')


def scenarios_at(run, protocol):
    from algos.p4_data import create_scenarios, sha_file
    base = create_scenarios(ROOT / 'src/sac_rwb/real_data', run / 'scenarios')
    result = dict(base['scenarios'])
    for name, multiplier in protocol['evaluation']['densities'].items():
        origin = run / 'scenarios/test'
        destination = run / ('scenarios/test-' + name)
        destination.mkdir()
        route = ET.parse(str(origin / 'routes.xml'))
        for flow in route.getroot().findall('flow'):
            if 'vehsPerHour' not in flow.attrib:
                raise ValueError('Density scaling requires explicit vehsPerHour flows')
            flow.set('vehsPerHour', str(float(flow.get('vehsPerHour')) * multiplier))
        route.write(str(destination / 'routes.xml'), encoding='utf-8', xml_declaration=True)
        config = ET.parse(str(origin / 'osm.sumocfg'))
        config.find('./input/route-files').set('value', str((destination / 'routes.xml').resolve()))
        config.write(str(destination / 'osm.sumocfg'), encoding='utf-8', xml_declaration=True)
        result['test-' + name] = dict(base['scenarios']['test'], density=name,
                                      scenario_id=sha_file(destination / 'routes.xml'),
                                      config=str((destination / 'osm.sumocfg').resolve()),
                                      density_multiplier=multiplier,
                                      density_scope='flow demand only; explicit background vehicles unchanged')
    return result


def common_corpus():
    from algos.p4_data import audit_datasets, sha_file
    summary = read(ROOT / 'plan/p4/runs/20261121T150247686103Z-train-extension/summary.json')
    origin = summary['base_run'].split('/plan/p4/runs/', 1)[0] + '/'
    paths = []
    for saved in summary['training_collections']['b0'] + summary['evaluation_collections']['b0']:
        if not saved.startswith(origin):
            raise ValueError('Unrecognized historical corpus root')
        path = (ROOT / saved[len(origin):]).resolve()
        path.relative_to(ROOT.resolve())
        paths.append(path)
    fingerprints = read(ROOT / 'plan/p4/runs/20261121T150247686103Z-train-extension/frozen-evaluation-sha256.json')
    checked = 0
    for saved, expected_hash in fingerprints.items():
        if '/b0/' not in saved:
            continue
        if not saved.startswith(origin):
            raise ValueError('Frozen evaluation outside accepted project root')
        path = (ROOT / saved[len(origin):]).resolve()
        path.relative_to(ROOT.resolve())
        hashes = [sha_file(path)]
        if path.suffix == '.json':
            hashes.append(hashlib.sha256(path.read_bytes().replace(b'\r\n', b'\n')).hexdigest())
        if expected_hash not in hashes:
            raise ValueError('Frozen P4 B0 evaluation changed: ' + str(path))
        checked += 1
    if checked != 8:
        raise ValueError('Incomplete frozen B0 evaluation fingerprint inventory')
    audit = audit_datasets(paths)
    expected = read(ROOT / 'plan/p4/runs/20261121T150247686103Z-train-extension/b0-audit.json')
    if audit != expected or not audit['data_ready']:
        raise ValueError('Strict common-corpus audit differs from accepted P4 data')
    return paths, audit


def execute(run, name, argv, timeout_seconds, commands, cwd=ROOT):
    with (run / (name + '.log')).open('x', encoding='utf-8') as stream:
        process = subprocess.Popen([str(v) for v in argv], cwd=str(cwd), stdout=stream,
                                   stderr=subprocess.STDOUT, start_new_session=os.name != 'nt')
        def stop():
            if os.name != 'nt':
                os.killpg(process.pid, signal.SIGKILL)
            else:
                process.kill()
            process.wait()
        try:
            code = process.wait(timeout=timeout_seconds)
        except subprocess.TimeoutExpired:
            stop()
            code = 124
        except BaseException:
            stop()
            raise
    (run / (name + '.exitcode')).write_text(str(code) + '\n', encoding='utf-8')
    commands.append(dict(name=name, argv=[str(v) for v in argv], exitcode=code))
    write_json(run / 'commands.json', commands)
    if code:
        raise RuntimeError('{} failed ({})'.format(name, code))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group()
    group.add_argument('--local', action='store_true')
    group.add_argument('--smoke', action='store_true')
    args = parser.parse_args()
    sys.path.insert(0, str(ROOT / 'src/sac_rwb'))
    from algos.p5_experiment import corpus_fingerprint
    protocol = read(ROOT / 'plan/p5/protocol.json')
    validate_protocol(protocol)
    gate = read(ROOT / 'plan/review/p5-entry-confirmation-20261122.json')
    if not gate.get('passed') or not gate.get('p5_ready'):
        raise ValueError('Accepted P0-P4 entry gate required')
    parent = ROOT / 'plan/p5/runs'
    parent.mkdir(exist_ok=True)
    run = parent / datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
    run.mkdir()
    before = source_manifest()
    write_json(run / 'source-manifest.json', before)
    archive = run / 'source.tar.gz'
    with tarfile.open(str(archive), 'w:gz') as stream:
        for entry in before:
            stream.add(str(ROOT / entry['path']), arcname=entry['path'], recursive=False)
        stream.add(str(ROOT / 'plan/p5/protocol.json'), arcname='plan/p5/protocol.json')
    write_json(run / 'source-snapshot.json', {'sha256': hashlib.sha256(archive.read_bytes()).hexdigest()})
    report = dict(mode='local' if args.local else ('smoke' if args.smoke else 'research'),
                  passed=False, p5_started=not args.local, commands=[], status='running')
    print('Evidence: {}'.format(run), flush=True)
    try:
        if not args.local:
            env = collect_environment(True)
            write_json(run / 'environment.json', env)
            if env.get('target_errors'):
                raise ValueError('Target environment differs from P0 lock')
        unit = [sys.executable, '-B', ROOT / 'scripts/run_unit_checks.py', '--output', run / 'unit-results.json']
        if not args.local:
            unit.append('--require-cuda')
        execute(run, 'unit', unit, 1800, report['commands'])
        execute(run, 'static', [sys.executable, '-B', ROOT / 'scripts/verify_plan.py', '--report', run / 'static.json'],
                1800, report['commands'])
        scenes = scenarios_at(run, protocol)
        if args.local:
            write_json(run / 'protocol.json', protocol)
            report.update(local_checks_passed=True, status='auxiliary_passed_not_p5_acceptance')
        else:
            paths, audit = common_corpus()
            write_json(run / 'common-corpus-audit.json', audit)
            fingerprint = corpus_fingerprint(paths)
            if args.smoke:
                protocol = smoke_protocol(protocol)
            plan = dict(protocol=protocol, protocol_sha256=protocol_hash(protocol), scenarios=scenes,
                        common_corpus=[str(path) for path in paths], common_corpus_sha256=fingerprint)
            write_json(run / 'plan.json', plan)
            worker = ROOT / 'scripts/p5_worker.py'
            for seed in protocol['training_seeds']:
                for variant in protocol['policy_variants']:
                    destination = run / 'online' / variant / str(seed)
                    argv = [sys.executable, '-B', worker, 'train', '--plan', run / 'plan.json',
                            '--variant', variant, '--seed', seed, '--output', destination]
                    work = run / ('work-' + variant + '-' + str(seed))
                    work.mkdir()
                    print('Training {} seed {}'.format(variant, seed), flush=True)
                    execute(run, 'train-{}-{}'.format(variant, seed), argv, 172800, report['commands'], work)
                for variant in protocol['risk_variants']:
                    execute(run, 'offline-{}-{}'.format(variant, seed), [sys.executable, '-B', worker,
                            'offline', '--plan', run / 'plan.json', '--variant', variant, '--seed', seed,
                            '--output', run / 'offline' / variant / str(seed)], 86400, report['commands'])
                for variant in protocol['policy_variants']:
                    for density in protocol['evaluation']['densities']:
                        for evaluation_seed in protocol['evaluation']['seeds']:
                            destination = run / 'closed-loop' / variant / str(seed) / density / str(evaluation_seed)
                            work = run / ('eval-work-{}-{}-{}-{}'.format(variant, seed, density, evaluation_seed))
                            work.mkdir()
                            execute(run, 'eval-{}-{}-{}-{}'.format(variant, seed, density, evaluation_seed),
                                    [sys.executable, '-B', worker, 'evaluate', '--plan', run / 'plan.json',
                                     '--variant', variant, '--seed', evaluation_seed, '--density', density,
                                     '--model-dir', run / 'online' / variant / str(seed), '--output', destination],
                                    14400, report['commands'], work)
            if corpus_fingerprint(paths) != fingerprint:
                raise ValueError('Frozen common corpus changed during experiment')
            from audit_p5 import audit
            verdict = audit(run, require_full=not args.smoke)
            write_json(run / 'audit.json', verdict)
            report.update(passed=verdict['passed'], status=verdict['status'],
                          engineering_passed=True, research_matrix_completed=not args.smoke)
        if source_manifest() != before:
            raise ValueError('Active sources changed during execution')
    except Exception as error:
        report.update(passed=False, status='failed', error='{}: {}'.format(type(error).__name__, error))
        print(report['error'], file=sys.stderr)
    write_json(run / 'summary.json', report)
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)
    return 1 if report.get('error') else (0 if args.local or report['passed'] else 2)


if __name__ == '__main__':
    sys.exit(main())
