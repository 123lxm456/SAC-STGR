"""Synthetic SUMO substitutes exercise P5 orchestration, not traffic results."""

import json
import hashlib
from pathlib import Path
import sys
import tempfile
import tarfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
sys.path.insert(0, str(ROOT / 'src/sac_rwb'))
import p5_worker
from sac.utils import Actor
from algos.p5_metrics import audit_physical
from algos.p5_experiment import fit_offline, corpus_fingerprint, make_directory
from accept_p5 import protocol_hash
from audit_p5 import audit


class FakeEnv:
    """Eight stationary synthetic cars, no collision. Includes reset/idle steps."""
    def __init__(self, config):
        self.config = config
        self.max_acc, self.min_acc = 3., -3.
        self.vehicles = {str(i): SimpleNamespace(route=['synthetic']) for i in range(8)}
        self.env_step = 0
        self._max_episode_steps = config['max_episode_steps']
        self.closed = False
        self.junction_list, self.keywords_order = ['j'], ['flow']
        self.sumo_interface = SimpleNamespace(tc=SimpleNamespace(
            simulation=SimpleNamespace(getCollisions=lambda: []),
            vehicle=SimpleNamespace(getIDList=lambda: list(self.vehicles), getSpeed=lambda v: 1.)))

    def reset(self):
        self.env_step = 0
        return self.step()[0], {}

    def step(self, actions=None, scores=None):
        actions = actions or {}
        collector = self.config.get('risk_event_collector')
        if collector is not None:
            for vehicle, action in actions.items():
                collector.confirm_action(vehicle, action, self.env_step)
        self.env_step += 1
        if collector is not None:
            collector.observe(self.env_step, [], [], list(self.vehicles))
        observer = self.config.get('physical_observer')
        if observer:
            observer.step(self, SimpleNamespace(departed_vehicles_ids=list(self.vehicles) if self.env_step == 1 else [],
                                                arrived_vehicles_ids=[], colliding_vehicles_ids=[]))
        states = {v: np.zeros(98, np.float32) for v in self.vehicles}
        return states, {v: 0. for v in actions}, {v: False for v in actions}, {}, {}, states

    def get_queue_len(self, *args):
        return 8

    def get_avg_wait_time(self, *args):
        return 1.

    def convert_virtual_id_to_real_id(self, vehicle):
        return vehicle

    def risk_history_context(self, vehicle):
        return ('synthetic', 'j')

    def close(self):
        self.closed = True


class FakeAgent:
    def __init__(self, **kwargs):
        self.actor = Actor(98, 1, (256, 256))
        self.q_critic = torch.nn.Linear(98, 1)
        self.replay_buffer = SimpleNamespace(size=0)
        def add(*args):
            self.replay_buffer.size += 1
        self.replay_buffer.add = add

    def select_action(self, states, deterministic):
        return np.zeros((len(states), 1), np.float32)

    def train(self):
        return {'updates': 5}


class WorkerTests(unittest.TestCase):
    def test_training_and_fixed_horizon_evaluation_reuse_p4_runtime_without_sumo(self):
        with tempfile.TemporaryDirectory(dir=ROOT) as temporary:
            root = Path(temporary)
            protocol = json.loads((ROOT / 'plan/p5/protocol.json').read_text())
            protocol['online'].update(decision_steps=60, episode_decisions=60)
            protocol['evaluation']['simulation_steps'] = 30
            protocol.update(research_run=False, training_seeds=[11], risk_variants=['b0', 'm1'],
                            policy_variants=['b0', 'm1', 's0'])
            protocol['offline'].update(updates=2, batch_size=4)
            protocol['evaluation'].update(seeds=[3000], densities={'nominal': 1.})
            scene = dict(scenario_id='synthetic', route_hashes=['synthetic'], config='synthetic', network='synthetic', density='nominal')
            initial = set()
            with patch.object(p5_worker, 'Env', FakeEnv), patch.object(p5_worker, 'SAC_countinuous', FakeAgent):
                for variant in ('b0', 'm1', 's0'):
                    model = root / 'online' / variant / '11'
                    make_directory(model, new=True)
                    p5_worker.train(protocol, scene, variant, 11, model, 'cpu')
                    result = json.loads((model / 'result.json').read_text())
                    self.assertEqual(result['sac_updates'], 5)
                    self.assertEqual(result['decision_steps'], 60)
                    initial.add(result['initial_actor_sha256'])
                    evaluation = root / 'closed-loop' / variant / '11' / 'nominal' / '3000'
                    make_directory(evaluation, new=True)
                    p5_worker.evaluate(protocol, scene, variant, 3000, model, evaluation, 'cpu')
                    measured = json.loads((evaluation / 'result.json').read_text())
                    self.assertEqual(measured['metrics']['simulation_steps'], 30)
                    self.assertEqual(measured['metrics']['departed'], 8)
                    self.assertEqual(measured['risk_updates'], 0)
                    self.assertEqual(audit_physical(evaluation / 'physical.jsonl'), measured['metrics'])
            self.assertEqual(len(initial), 1)
            # A complete *synthetic* smoke matrix may pass engineering audit,
            # but must never become full research acceptance.
            rng = np.random.RandomState(5)
            batch = (rng.randn(4, 10, 98).astype(np.float32), np.zeros((4, 10, 1), np.float32),
                     np.ones((4, 10), np.bool_), np.array([0, 1, 0, 1], np.float32))
            corpus = {split: batch for split in ('train', 'validation', 'test')}
            directories = []
            for split in corpus:
                directory = root / ('synthetic-corpus-' + split)
                directory.mkdir()
                (directory / 'manifest.json').write_text(json.dumps({'metadata': {'split': split}}))
                (directory / 'counts.json').write_text('{}')
                (directory / 'samples.jsonl').write_text(''.join(json.dumps({'label': int(y)}) + '\n' for y in batch[3]))
                (directory / 'events.jsonl').write_text('')
                directories.append(str(directory))
            fingerprint = corpus_fingerprint(directories)
            with patch('algos.p5_experiment.benchmark', return_value={'synthetic_fixture': True}):
                for variant in ('b0', 'm1'):
                    fit_offline(variant, 11, corpus, 2, 4, 'cpu', root / 'offline' / variant / '11', corpus_id=fingerprint)
            plan = dict(protocol=protocol, protocol_sha256=protocol_hash(protocol),
                        common_corpus=directories, common_corpus_sha256=fingerprint,
                        scenarios={'test-nominal': scene})
            (root / 'plan.json').write_text(json.dumps(plan))
            (root / 'environment.json').write_text(json.dumps(dict(target_required=True, cuda_available=True,
                                                                  target_errors=[], gpu='NVIDIA GeForce RTX 4090')))
            (root / 'unit-results.json').write_text(json.dumps(dict(passed=True, require_cuda=True, tests_run=1,
                                                                   skipped=[], errors=[], failures=[])))
            (root / 'static.json').write_text(json.dumps({'passed': True}))
            source = root / 'synthetic-source.txt'
            source.write_text('Synthetic fixture, not target execution evidence')
            (root / 'source-manifest.json').write_text(json.dumps([dict(path=source.name,
                sha256=hashlib.sha256(source.read_bytes()).hexdigest())]))
            with tarfile.open(str(root / 'source.tar.gz'), 'w:gz') as archive:
                archive.add(str(source), arcname=source.name)
            (root / 'source-snapshot.json').write_text(json.dumps(dict(
                sha256=hashlib.sha256((root / 'source.tar.gz').read_bytes()).hexdigest())))
            names = ['unit', 'static'] + ['train-{}-11'.format(v) for v in protocol['policy_variants']]
            names += ['offline-{}-11'.format(v) for v in protocol['risk_variants']]
            names += ['eval-{}-11-nominal-3000'.format(v) for v in protocol['policy_variants']]
            for name in names:
                (root / (name + '.log')).write_text('Synthetic fixture')
                (root / (name + '.exitcode')).write_text('0')
            (root / 'commands.json').write_text(json.dumps([dict(name=n, exitcode=0) for n in names]))
            verdict = audit(root, require_full=False)
            self.assertFalse(verdict['passed'])
            self.assertTrue(verdict['primary_matrix_integrity_passed'])
            self.assertEqual(verdict['status'], 'smoke_passed_not_research')
            with self.assertRaises(ValueError):
                audit(root, require_full=True)
            (root / 'train-b0-11.exitcode').write_text('1')
            with self.assertRaisesRegex(ValueError, 'execution proof'):
                audit(root, require_full=False)


if __name__ == '__main__':
    unittest.main()
