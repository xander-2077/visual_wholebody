from legged_gym import LEGGED_GYM_ROOT_DIR
import os

import isaacgym
from legged_gym.envs import *
from legged_gym.utils import  get_args, export_policy_as_jit, task_registry, Logger
from legged_gym.utils.helpers import get_load_path

import numpy as np
import torch
import time
import sys

np.set_printoptions(precision=3, suppress=True)

def play(args):
    log_pth = LEGGED_GYM_ROOT_DIR + "/logs/{}/".format(args.proj_name) + args.exptid
    env_cfg, train_cfg = task_registry.get_cfgs(name=args.task)
    # override some parameters for testing
    env_cfg.env.num_envs = 1
    # env_cfg.commands.ranges.lin_vel_x = [-1, 1]

    env_cfg.terrain.num_rows = 6
    env_cfg.terrain.num_cols = 3
    # env_cfg.env.episode_length_s = 10000
    env_cfg.domain_rand.push_robots = False
    # env_cfg.domain_rand.push_interval_s = 2
    env_cfg.domain_rand.randomize_base_mass = True #False
    env_cfg.domain_rand.randomize_base_com = False
    
    if args.flat_terrain:
        env_cfg.terrain.height = [0.0, 0.0]

    # prepare environment
    env, _ = task_registry.make_env(name=args.task, args=args, env_cfg=env_cfg)
    obs = env.get_observations()
    # load policy
    train_cfg.runner.resume = True
    ppo_runner, train_cfg, checkpoint, log_pth = task_registry.make_alg_runner(log_root = log_pth, env=env, name=args.task, args=args, train_cfg=train_cfg, return_log_dir=True)
    policy = ppo_runner.get_inference_policy(device=env.device, stochastic=args.stochastic)

    # export policy as a jit module (used to run it from C++)
    if EXPORT_POLICY:
        path = os.path.join(LEGGED_GYM_ROOT_DIR, 'logs', train_cfg.runner.experiment_name, 'exported', 'policies')
        export_policy_as_jit(ppo_runner.alg.actor_critic, path)
        print('Exported policy as jit script to: ', path)

    if SAVE_ACTOR_HIST_ENCODER:
        log_root = os.path.join(LEGGED_GYM_ROOT_DIR, 'logs', train_cfg.runner.experiment_name)
        model_file = get_load_path(log_root, load_run=args.load_run, checkpoint=args.checkpoint)
        model_name = model_file.split('/')[-1].split('.')[0]
        path = os.path.join(LEGGED_GYM_ROOT_DIR, 'logs', train_cfg.runner.experiment_name, train_cfg.runner.load_run, 'exported')
        os.makedirs(path, exist_ok=True)
        torch.save(ppo_runner.alg.actor_critic.actor.state_dict(), path + '/' + model_name + '_actor.pt')
        print('Saved actor to: ', path + '/' + model_name + '_actor.pt')
    
    if args.use_jit:
        path = os.path.join(log_pth, 'traced', args.exptid + "_" + str(args.checkpoint) + "_jit.pt")
        print("Loading jit for policy: ", path)
        policy = torch.jit.load(path, map_location=ppo_runner.device)
    
    logger = Logger(env.dt)
    robot_index = 0 # which robot is used for logging
    joint_index = 1 # which joint is used for logging
    stop_state_log = 100 # number of steps before plotting states
    stop_rew_log = env.max_episode_length + 1 # number of steps before print average episode rewards
    camera_position = np.array(env_cfg.viewer.pos, dtype=np.float64)
    camera_vel = np.array([1., 1., 0.])
    camera_direction = np.array(env_cfg.viewer.lookat) - np.array(env_cfg.viewer.pos)
    img_idx = 0

    mp4_writers = []
    if args.record_video:
        import imageio
        env.enable_viewer_sync = False
        for i in range(env.num_envs):
            video_name = args.exptid+ f'-{i}-' + str(checkpoint) +".mp4"
            run_name = log_pth.split("/")[-1]
            path = f"../../logs/videos/{run_name}"
            if not os.path.exists(path):
                os.makedirs(path)
            video_name = os.path.join(path, video_name)
            mp4_writer = imageio.get_writer(video_name, fps=25)
            mp4_writers.append(mp4_writer)

    if not args.record_video:
        traj_length = 1000*int(env.max_episode_length)
    else:
        traj_length = int(env.max_episode_length)

    # env.update_command_curriculum()
    env.reset()
    # env.curr_ee_goal_cart = torch.tensor([[0.5, 0.5, 0.5]], device=env.device)
    for i in range(traj_length):
        start_time = time.time()
        if args.use_jit:
            actions = policy(torch.cat((obs[:, :env.cfg.env.num_proprio], obs[:, env.cfg.env.num_proprio+env.cfg.env.num_priv:]), dim=1))
        else:
            actions = policy(obs.detach(), hist_encoding=True)
        # env.curr_ee_goal_cart = torch.tensor([[0.5, 0.5, 0.5]], device=env.device)
        obs, _, rews, arm_rews, dones, infos = env.step(actions.detach())
        if args.record_video:
            imgs = env.render_record(mode='rgb_array')
            if imgs is not None:
                for i in range(env.num_envs):
                    mp4_writers[i].append_data(imgs[i])
        
        stop_time = time.time()

        duration = stop_time - start_time
        time.sleep(max(0.02 - duration, 0))

    if args.record_video:
        for mp4_writer in mp4_writers:
            mp4_writer.close()

if __name__ == '__main__':
    EXPORT_POLICY = False
    SAVE_ACTOR_HIST_ENCODER = False
    RECORD_FRAMES = False
    MOVE_CAMERA = False
    args = get_args()
    play(args)
