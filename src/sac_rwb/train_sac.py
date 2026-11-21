from sac.utils import str2bool
from datetime import datetime
from sac.SAC import SAC_countinuous
import os
import random
import argparse
import torch
from Env import Env
import numpy as np
import time
import json
from core.runtime import wait_for_observation, reserve_run
from torch.utils.tensorboard import SummaryWriter
from algos.online_risk import create_risk_predictor, OnlineRiskRuntime

'''Hyperparameter Setting'''
parser = argparse.ArgumentParser()
parser.add_argument('--dvc', type=str, default='cuda', help='running device: cuda or cpu')
parser.add_argument('--EnvIdex', type=int, default=0, help='PV1, Lch_Cv2, Humanv4, HCv4, BWv3, BWHv3')
parser.add_argument('--render', type=str2bool, default=False, help='Render or Not')
parser.add_argument('--Loadmodel', type=str2bool, default=False, help='Load pretrained model or Not')
parser.add_argument('--ModelIdex', type=int, default=100, help='which model to load')
parser.add_argument('--seed', type=int, default=0, help='random seed')
parser.add_argument('--Max_train_steps', type=int, default=int(5e5), help='Max training steps')
parser.add_argument('--save_interval', type=int, default=int(1e4), help='Model saving interval, in steps.')
parser.add_argument('--eval_interval', type=int, default=int(2.5e3), help='Model evaluating interval, in steps.')
parser.add_argument('--update_every', type=int, default=50, help='Training Fraquency, in stpes')
parser.add_argument('--gamma', type=float, default=0.99, help='Discounted Factor')
parser.add_argument('--net_width', type=int, default=256, help='Hidden net width, s_dim-400-300-a_dim')
parser.add_argument('--a_lr', type=float, default=3e-4, help='Learning rate of actor')
parser.add_argument('--c_lr', type=float, default=3e-4, help='Learning rate of critic')
parser.add_argument('--batch_size', type=int, default=256, help='batch_size of training')
parser.add_argument('--alpha', type=float, default=0.12, help='Entropy coefficient')
parser.add_argument('--adaptive_alpha', type=str2bool, default=True, help='Use adaptive_alpha or Not')
parser.add_argument('--risk_variant', choices=('b0', 'm1'), default='b0', help='B0 legacy time model or M1 P2 spatial-temporal model')
parser.add_argument('--d_model', type=int, default=128, help='B0 width; M1 uses the specified P2 width 64')
parser.add_argument('--nhead', type=int, default=4, help='B0 head count; M1 uses four heads')
parser.add_argument('--num_layers', type=int, default=2, help='B0 temporal depth; M1 uses the specified P2 architecture')
parser.add_argument('--buffer_size', type=int, default=10000, help='Buffer size for the risk prediction model')
parser.add_argument('--risk_model_path', type=str, default="risk_model/risk_pretrained.pth", help='path of risk prediction model')
parser.add_argument('--load_risk_model', type=int, default=0, help='Whether to load the pretrained risk prediction model (1 to load, 0 not to load)')
parser.add_argument('--epochs', type=int, default=500)
parser.add_argument('--max_e_steps', type=int, default=1000)
opt = parser.parse_args()
for name in ('epochs', 'max_e_steps', 'Max_train_steps', 'save_interval', 'update_every', 'batch_size'):
    if getattr(opt, name) < 1:
        parser.error('--{} must be positive'.format(name))
if opt.Loadmodel:
    parser.error('--Loadmodel is unsupported: SAC checkpoints lack optimizer/replay state for resumption')
opt.dvc = torch.device(opt.dvc) # from str to torch.device
print(opt)

def main():
    env_name = "unsignalizedIntersections"
    torch.manual_seed(opt.seed)
    torch.cuda.manual_seed(opt.seed)
    random.seed(opt.seed)
    np.random.seed(opt.seed)
    # 创建风险预测模型保存目录
    risk_model_dir = "risk_model/"
    if not os.path.exists(risk_model_dir):
        os.makedirs(risk_model_dir)
    # 创建风险预测模型
    risk_predictor = create_risk_predictor(
        opt.risk_variant, buffer_size=opt.buffer_size,
        b0_d_model=opt.d_model, b0_nhead=opt.nhead, b0_layers=opt.num_layers,
    ).to(opt.dvc)
    risk_runtime = OnlineRiskRuntime(risk_predictor, opt.dvc)
    if opt.load_risk_model:
        risk_model_path = opt.risk_model_path
        if os.path.exists(risk_model_path):
            risk_predictor.load_model(risk_model_path, map_location=opt.dvc)
        else:
            raise FileNotFoundError(f"Risk prediction model file not found: {risk_model_path}")
    else:
        print("Not loading pre trained risk prediction models, training will start from scratch")
    # Build Env
    env = Env({
        "junction_list": ['229', '499', '332', '334'],
        "spawn_rl_prob": {},
        "probablity_RL": 1.0,
        "cfg": 'real_data/osm.sumocfg',
        "render": opt.render,
        "map_xml": 'real_data/CSeditClean_1.net_threelegs.xml',
        "max_episode_steps": opt.max_e_steps,
        "conflict_mechanism": 'flexible',
        "traffic_light_program": {
            "disable_state": 'G',
            "disable_light_start": 0
        },
        "use_safety_predictor": True,
        "sumo_seed": opt.seed,
        "risk_predictor": risk_predictor
    })
    writer = None
    try:
        opt.state_dim = env.n_obs
        opt.action_dim = env.n_action
        opt.max_action = env.max_acc   #remark: action space【-max,max】
        # Seed Everything
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        print("Random Seed: {}".format(opt.seed))
        # Build DRL model
        if not os.path.exists('model'): os.mkdir('model')
        agent = SAC_countinuous(**vars(opt))
        ###### log ######
        log_dir = "Train_logs"
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)
        log_dir = log_dir + '/' + env_name + '/'
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)
        run_num, run_manifest = reserve_run(log_dir)
        run_manifest.write_text(json.dumps(vars(opt), default=str, indent=2), encoding='utf-8')
        print("current logging run number for " + env_name + " : ", run_num)
        print("run configuration : " + str(run_manifest))
        tensorboard_dir = "tensorboard_logs"
        if not os.path.exists(tensorboard_dir):
            os.makedirs(tensorboard_dir)
        tensorboard_dir = tensorboard_dir + '/' + env_name + '/'
        if not os.path.exists(tensorboard_dir):
            os.makedirs(tensorboard_dir)
    
        # 创建tensorboard writer
        timenow = str(datetime.now())[0:-10]
        timenow = ' ' + timenow[0:13] + '_' + timenow[-2::]
        writepath = tensorboard_dir + '/run{}'.format(run_num) + timenow
        writer = SummaryWriter(writepath)
        print("tensorboard logging at : " + tensorboard_dir + f'run_{run_num}')
        total_steps = 0
        sac_update_total = 0
        risk_update_total = 0
        cur_time = time.localtime()
        print("start time: %02d:%02d:%02d" % (cur_time.tm_hour, cur_time.tm_min, cur_time.tm_sec))
        for i_episode in range(opt.epochs):
            if total_steps >= opt.Max_train_steps:
                break
            risk_predictor.buffer.clear_pending()
            state, _ = env.reset()
            stime = time.time()
            current_ep_reward = 0
            current_ep_efficiency_reward = 0
            current_ep_safety_reward = 0
            current_reward_count = 0
            '''Interact & trian'''
            for t in range(min(opt.max_e_steps, opt.Max_train_steps - total_steps)):
                total_steps += 1
                state = wait_for_observation(env, state)
                s = list(state.values())
                a = agent.select_action(s, deterministic=False)  # a∈[-1,1]
                a = a.flatten()
                action = a * opt.max_action
                action_dict = dict(zip(state.keys(), action))
                a_dict = dict(zip(state.keys(), a))
                safe_reward_dict = risk_runtime.score(env, state, action_dict)
                next_state, reward, is_terminal, truncated, info, new_state = env.step(action_dict, safe_reward_dict)
                for i, veh_id in enumerate(reward.keys()):
                    agent.replay_buffer.add(state[veh_id], np.array(a_dict[veh_id]), reward[veh_id],
                                            new_state[veh_id], bool(is_terminal.get(veh_id, False)))
                for key, done in is_terminal.items():
                    if done:
                        next_state.pop(key)
                state = next_state.copy()
                # 统计各类奖励
                for veh_id, reward_val in reward.items():
                    current_ep_reward += reward_val
                    current_reward_count += 1
                    real_id = env.convert_virtual_id_to_real_id(veh_id)
                    if real_id in env.reward_record:
                        current_ep_efficiency_reward += env.reward_record[real_id]['efficiency']
                        current_ep_safety_reward += env.reward_record[real_id]['safety']
                '''train if it's time'''
                if total_steps % max(1, opt.update_every) == 0 and agent.replay_buffer.size >= opt.batch_size:
                    losses = agent.train()
                    sac_update_total += losses['updates']
                    writer.add_scalar('Loss/q_loss', losses['q_loss'], total_steps)
                    writer.add_scalar('Loss/a_loss', losses['a_loss'], total_steps)
                    if losses['alpha_loss'] is not None:
                        writer.add_scalar('Loss/alpha_loss', losses['alpha_loss'], total_steps)
                    writer.add_scalar('Updates/SAC', sac_update_total, total_steps)

                    if (env.use_safety_predictor and len(risk_predictor.buffer.active_buffer) >= 64
                            and len(risk_predictor.buffer.risk_active_buffer) >= 64):
                        loss = risk_predictor.train_model(batch_size=128, epochs=3, lr=1e-5, device=opt.dvc)
                        risk_update_total += risk_predictor.last_train_updates
                        if risk_predictor.last_train_updates:
                            writer.add_scalar('Loss/Risk_Model_Loss', loss, total_steps)
                        writer.add_scalar('Updates/Risk', risk_update_total, total_steps)
                        risk_predictor.eval()

                '''save model'''
                if total_steps % opt.save_interval == 0:
                    print("save model")
                    agent.save(env_name, total_steps, run_num)
                    run_num_pretrained = run_num
                    random_seed = opt.seed
                    risk_model_dir_path = risk_model_dir+"risk_{}_{}_{}_{}_{}_step{}.pth".format(opt.risk_variant, env_name, random_seed, i_episode, run_num_pretrained, total_steps)
                    risk_predictor.save_model(risk_model_dir_path)
            etime = time.time()
            print(f"\n{'=' * 100}\n"
                  f"Episode: {i_episode:<10} | Timestep: {total_steps:<10} | Average Reward: {current_ep_reward:<10.2f} | Episode Time: {round((etime - stime), 2):<10.2f}\n"
                  f"{'=' * 100}")
            print(f"Optimizer updates: SAC={sac_update_total}, risk={risk_update_total}; "
                  f"risk pools: safe={len(risk_predictor.buffer.active_buffer)}, "
                  f"collision={len(risk_predictor.buffer.risk_active_buffer)}")
            runtime_report = {
                'risk_variant': opt.risk_variant,
                'risk_parameter_count': sum(p.numel() for p in risk_predictor.parameters()),
                'risk_network_config': risk_predictor.network_config,
                'decision_steps': total_steps, 'episodes_completed': i_episode + 1,
                'sac_updates': sac_update_total, 'risk_updates': risk_update_total,
                'safe_pool': len(risk_predictor.buffer.active_buffer),
                'collision_pool': len(risk_predictor.buffer.risk_active_buffer),
                'seed': opt.seed, 'device': str(opt.dvc),
                'risk_supervision': 'legacy_terminal_outcomes; P4 horizon/censoring not implemented',
                'inference': risk_runtime.summary(),
            }
            runtime_path = os.path.join(log_dir, 'risk_runtime_{}.json'.format(run_num))
            with open(runtime_path, 'w', encoding='utf-8') as stream:
                json.dump(runtime_report, stream, ensure_ascii=False, indent=2)
            print('Risk runtime report: ' + runtime_path)
            # 记录奖励指标
            writer.add_scalar('Reward/Total', current_ep_reward, i_episode)
            writer.add_scalar('Reward/Efficiency', current_ep_efficiency_reward, i_episode)
            writer.add_scalar('Reward/Safety', current_ep_safety_reward, i_episode)
            # 计算平均奖励
            reward_count = max(1, current_reward_count)
            writer.add_scalar('Average_Reward/Avg_Total', current_ep_reward / reward_count, i_episode)
            writer.add_scalar('Average_Reward/Avg_Efficiency', current_ep_efficiency_reward / reward_count,
                              i_episode)
            writer.add_scalar('Average_Reward/Avg_Safety', current_ep_safety_reward / reward_count,
                              i_episode)
            # 记录环境数据
            env_data = env.monitor.get_data(env)
            writer.add_scalar('Env_Data/max_quene_length', env_data['max_quene_length'], total_steps)
            writer.add_scalar('Env_Data/avg_quene_length', env_data['avg_quene_length'], total_steps)
            writer.add_scalar('Env_Data/quene_wait', env_data['quene_wait'], total_steps)
            writer.add_scalar('Env_Data/conflict', env_data['conflict'], total_steps)
            writer.add_scalar('Env_Data/conflict_rate', env_data['conflict_rate'], total_steps)
            writer.add_scalar('Env_Data/avg_speed', env_data['avg_speed'], total_steps)

    finally:
        if writer is not None:
            writer.close()
        env.close()

if __name__ == '__main__':
    main()
