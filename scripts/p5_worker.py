"""Isolated P5 worker. Invoked by accept_p5.py; no SSH or historical writes."""

import argparse
import json
from pathlib import Path
import sys
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src/sac_rwb'))
from algos.p5_experiment import seed_all, write_json, fit_offline, load_corpus, balanced_update, corpus_fingerprint, make_directory
from algos.p5_models import create_p5_predictor
from algos.online_risk import OnlineRiskRuntime
from algos.horizon_risk import HorizonRiskBuffer
from algos.p4_data import sha_file
from algos.p5_metrics import PhysicalObserver, audit_physical
from core.runtime import wait_for_observation
from sac.SAC import SAC_countinuous
from sac.utils import Actor
from Env import Env


def env_config(scene, seed, predictor, collector=None, observer=None, horizon=1000):
    return dict(junction_list=['229', '499', '332', '334'], spawn_rl_prob={},
                probablity_RL=1.0, cfg=scene['config'], map_xml=scene['network'], render=False,
                max_episode_steps=horizon, max_reset_wait_steps=min(1000, horizon),
                conflict_mechanism='flexible',
                traffic_light_program={'disable_state': 'G', 'disable_light_start': 0},
                use_safety_predictor=True, sumo_seed=seed, risk_predictor=predictor,
                risk_event_collector=collector, physical_observer=observer)


def train(protocol, scene, variant, seed, output, device):
    seed_all(seed)
    predictor = create_p5_predictor(variant).to(device)
    collector = HorizonRiskBuffer(output / 'online-data', dict(
        split='train', scenario_id=scene['scenario_id'], seed=seed, run_id=str(output),
        behavior='online shared SAC; from scratch', frozen_risk_model=False,
        allowed_route_hashes=scene['route_hashes']), max_size=10000)
    predictor.buffer = collector
    runtime = OnlineRiskRuntime(predictor, device)
    hp = protocol['online']
    # Different risk networks consume different numbers of random draws.
    # Reset before constructing SAC to pair the initial actor across groups.
    seed_all(seed)
    agent = SAC_countinuous(state_dim=98, action_dim=1, net_width=256, dvc=torch.device(device),
                           gamma=.99, a_lr=3e-4, c_lr=3e-4, batch_size=256,
                           alpha=.12, adaptive_alpha=True)
    actor_initial = output / 'actor-initial.pth'
    torch.save(agent.actor.state_dict(), str(actor_initial))
    env = None
    steps = sac_updates = risk_updates = risk_attempts = 0
    completed = False
    try:
        env = Env(env_config(scene, seed, predictor, collector, horizon=hp['episode_decisions']))
        episode = 0
        while steps < hp['decision_steps']:
            collector.start_episode(episode)
            state, _ = env.reset()
            for _ in range(min(hp['episode_decisions'], hp['decision_steps'] - steps)):
                state = wait_for_observation(env, state)
                normalized = agent.select_action(list(state.values()), deterministic=False).reshape(-1)
                actions = dict(zip(state, normalized * env.max_acc))
                normalized = dict(zip(state, normalized))
                scores = runtime.score(env, state, actions)
                next_state, rewards, dones, _, _, replay_next = env.step(actions, scores)
                for vehicle, reward in rewards.items():
                    agent.replay_buffer.add(state[vehicle], np.asarray(normalized[vehicle]), reward,
                                            replay_next[vehicle], bool(dones.get(vehicle, False)))
                for vehicle, done in dones.items():
                    if done:
                        next_state.pop(vehicle, None)
                state = next_state
                steps += 1
                if steps % hp['update_every'] == 0 and agent.replay_buffer.size >= 256:
                    sac_updates += agent.train()['updates']
                    if variant != 's0':
                        for _ in range(hp['risk_updates_per_trigger']):
                            risk_attempts += 1
                            s, a, mask, y = collector.sample(128)
                            if s is not None:
                                balanced_update(predictor, (s, a, mask, y), device)
                                risk_updates += 1
            episode += 1
        torch.save(agent.actor.state_dict(), str(output / 'actor.pth'))
        torch.save(agent.q_critic.state_dict(), str(output / 'critic.pth'))
        torch.save(predictor.state_dict(), str(output / 'risk.pth'))
        completed = True
    finally:
        try:
            if env is not None:
                env.close()
        finally:
            collector.close('collection_end' if completed else 'collection_interrupted')
    report = dict(completed=completed, variant=variant, seed=seed, decision_steps=steps,
                  sac_updates=sac_updates, risk_updates=risk_updates, risk_update_attempts=risk_attempts,
                  safe_pool=len(collector.active_buffer), collision_pool=len(collector.risk_active_buffer),
                  initial_actor_sha256=sha_file(actor_initial),
                  actor_sha256=sha_file(output / 'actor.pth'), risk_sha256=sha_file(output / 'risk.pth'),
                  risk_supervision='p4-h10-v1', inference=runtime.summary(),
                  network_config=predictor.network_config, initialized_from_scratch=True)
    write_json(output / 'result.json', report)


def evaluate(protocol, scene, variant, seed, model_dir, output, device):
    seed_all(seed)
    actor_path, risk_path = model_dir / 'actor.pth', model_dir / 'risk.pth'
    actor = Actor(98, 1, (256, 256)).to(device).eval()
    actor.load_state_dict(torch.load(str(actor_path), map_location=device))
    predictor = create_p5_predictor(variant).to(device).eval()
    predictor.load_state_dict(torch.load(str(risk_path), map_location=device))
    runtime = OnlineRiskRuntime(predictor, device)
    observer = PhysicalObserver(output / 'physical.jsonl')
    env = None
    horizon = protocol['evaluation']['simulation_steps']
    try:
        env = Env(env_config(scene, seed, predictor, observer=observer, horizon=horizon))
        state, _ = env.reset()
        if env.env_step > horizon:
            raise ValueError('Reset exceeded fixed evaluation horizon')
        while env.env_step < horizon:
            if not state:
                state, _, _, _, _, _ = env.step()
                continue
            with torch.no_grad():
                tensor = torch.as_tensor(np.asarray(list(state.values())), dtype=torch.float32, device=device)
                actions, _ = actor(tensor, deterministic=True, with_logprob=False)
            actions = dict(zip(state, actions.cpu().numpy().reshape(-1) * env.max_acc))
            scores = runtime.score(env, state, actions)
            state, _, dones, _, _, _ = env.step(actions, scores)
            for vehicle, done in dones.items():
                if done:
                    state.pop(vehicle, None)
    finally:
        try:
            if env is not None:
                env.close()
        finally:
            observer.close()
    metrics = audit_physical(output / 'physical.jsonl')
    if metrics != observer.metrics.summary() or metrics['simulation_steps'] != horizon:
        raise ValueError('Physical audit/horizon mismatch')
    if sha_file(actor_path) != json.loads((model_dir / 'result.json').read_text())['actor_sha256']:
        raise ValueError('Actor changed after training')
    report = dict(completed=True, variant=variant, evaluation_seed=seed, density=scene['density'],
                  scenario_id=scene['scenario_id'], actor_sha256=sha_file(actor_path),
                  risk_sha256=sha_file(risk_path), sac_updates=0, risk_updates=0,
                  metrics=metrics, inference=runtime.summary(), physical_sha256=sha_file(output / 'physical.jsonl'))
    write_json(output / 'result.json', report)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=('train', 'offline', 'evaluate'))
    parser.add_argument('--plan', type=Path, required=True)
    parser.add_argument('--variant', required=True)
    parser.add_argument('--seed', type=int, required=True)
    parser.add_argument('--density', default='nominal')
    parser.add_argument('--model-dir', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--device', default='cuda')
    args = parser.parse_args()
    plan = json.loads(args.plan.read_text(encoding='utf-8'))
    protocol = plan['protocol']
    if args.mode == 'offline':
        if corpus_fingerprint(plan['common_corpus']) != plan['common_corpus_sha256']:
            raise ValueError('Common corpus changed after protocol preparation')
        corpus = load_corpus(plan['common_corpus'])
        result = fit_offline(args.variant, args.seed, corpus, protocol['offline']['updates'],
                             protocol['offline']['batch_size'], args.device, args.output,
                             corpus_id=plan['common_corpus_sha256'])
        if corpus_fingerprint(plan['common_corpus']) != plan['common_corpus_sha256']:
            raise ValueError('Common corpus changed during fitting')
        return
    make_directory(args.output, new=True)
    if args.mode == 'train':
        train(protocol, plan['scenarios']['train'], args.variant, args.seed, args.output, args.device)
    else:
        evaluate(protocol, plan['scenarios']['test-' + args.density], args.variant, args.seed,
                 args.model_dir, args.output, args.device)


if __name__ == '__main__':
    main()
