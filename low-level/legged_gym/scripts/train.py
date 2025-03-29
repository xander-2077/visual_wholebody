import numpy as np
import os
from datetime import datetime
import isaacgym

from legged_gym import LEGGED_GYM_ROOT_DIR, LEGGED_GYM_ENVS_DIR
from legged_gym.envs import *
from legged_gym.utils import get_args, task_registry
import torch
import wandb

def train(args):
    log_pth = LEGGED_GYM_ROOT_DIR + "/logs/{}/".format(args.proj_name) + args.exptid
    try:
        os.makedirs(log_pth)
    except:
        pass
    if args.debug:
        mode = "disabled"
        args.rows = 6
        args.cols = 2
        args.num_envs = 128
    else:
        mode = "online"
    wandb.init(project=args.proj_name, name=args.exptid, mode=mode, dir=LEGGED_GYM_ENVS_DIR +"/logs")
    wandb.save(LEGGED_GYM_ENVS_DIR + "/manip_loco/b1z1_config.py", policy="now")
    wandb.save(LEGGED_GYM_ENVS_DIR + "/manip_loco/manip_loco.py", policy="now")

    env, env_cfg = task_registry.make_env(name=args.task, args=args)
    ppo_runner, train_cfg, _ = task_registry.make_alg_runner(log_root = log_pth, env=env, name=args.task, args=args)
    ppo_runner.learn(num_learning_iterations=train_cfg.runner.max_iterations, init_at_random_ep_len=True)

if __name__ == '__main__':
    args = get_args()
    train(args)
