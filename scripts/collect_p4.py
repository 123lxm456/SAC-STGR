"""Collect real P4 outcomes using frozen P3 actor/risk weights; never train P5."""

import argparse
import json
from pathlib import Path
import random
import shutil
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src/sac_rwb'))

import numpy as np
import torch
from Env import Env
from algos.horizon_risk import HorizonRiskBuffer
from algos.online_risk import create_risk_predictor, OnlineRiskRuntime
from algos.p4_data import sha_file
from core.runtime import wait_for_observation
from sac.utils import Actor


def collect(args):
    scenario = json.loads(Path(args.scenarios).read_text(encoding='utf-8'))['scenarios'][args.split]
    seed = scenario['seed'] + args.replicate
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    actor = Actor(98, 1, (256, 256)).to(args.device).eval()
    actor.load_state_dict(torch.load(args.actor, map_location=args.device))
    predictor = create_risk_predictor(args.variant).to(args.device)
    predictor.load_model(args.risk_model, map_location=args.device)
    metadata = dict(scenario, seed=seed, run_id=str(Path(args.output).resolve()),
                    allowed_route_hashes=scenario['route_hashes'],
                    behavior='frozen P3 SAC actor, deterministic actions; no online updates',
                    variant=args.variant, frozen_risk_model=True,
                    policy_sha256=sha_file(args.actor), risk_sha256=sha_file(args.risk_model),
                    weight_scope='P3 engineering smoke weights, not an effectiveness result')
    buffer = HorizonRiskBuffer(args.output, metadata)
    predictor.buffer = buffer
    runtime = OnlineRiskRuntime(predictor, args.device)
    env, completed, failure = None, False, None
    try:
        env = Env({
            'junction_list': ['229', '499', '332', '334'], 'spawn_rl_prob': {},
            'probablity_RL': 1.0, 'cfg': scenario['config'], 'render': False,
            'map_xml': scenario['network'], 'max_episode_steps': args.steps,
            'conflict_mechanism': 'flexible',
            'traffic_light_program': {'disable_state': 'G', 'disable_light_start': 0},
            'use_safety_predictor': True, 'sumo_seed': seed,
            'risk_predictor': predictor, 'risk_event_collector': buffer,
        })
        buffer.start_episode(0)
        state, _ = env.reset()
        for _ in range(args.steps):
            state = wait_for_observation(env, state)
            with torch.no_grad():
                tensor = torch.as_tensor(np.asarray(list(state.values())), dtype=torch.float32, device=args.device)
                normalized, _ = actor(tensor, deterministic=True, with_logprob=False)
            actions = dict(zip(state, normalized.cpu().numpy().reshape(-1) * env.max_acc))
            scores = runtime.score(env, state, actions)
            state, _, dones, _, _, _ = env.step(actions, scores)
            for key, done in dones.items():
                if done:
                    state.pop(key, None)
        completed = True
    except Exception as error:
        failure = '{}: {}'.format(type(error).__name__, error)
        raise
    finally:
        try:
            if env is not None:
                env.close()
        except Exception as error:
            completed = False
            failure = '{}: {}'.format(type(error).__name__, error)
            raise
        finally:
            buffer.close('collection_end' if completed else 'collection_interrupted')
            if Path('sumolog').is_file():
                shutil.copy2('sumolog', Path(args.output) / 'sumolog')
            (Path(args.output) / 'collection.json').write_text(json.dumps({
                'completed': completed, 'error': failure, 'decision_steps': runtime.calls,
                'inference': runtime.summary(), 'sac_updates': 0, 'risk_updates': 0,
                'p5_started': False}, indent=2), encoding='utf-8')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--scenarios', required=True)
    parser.add_argument('--split', choices=('train', 'validation', 'test'), required=True)
    parser.add_argument('--variant', choices=('b0', 'm1'), required=True)
    parser.add_argument('--replicate', type=int, default=0)
    parser.add_argument('--actor', required=True)
    parser.add_argument('--risk-model', required=True)
    parser.add_argument('--steps', type=int, default=1000)
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    if args.steps < 10 or not 0 <= args.replicate < 50:
        parser.error('steps >= 10 and replicate in [0,49] required')
    collect(args)
