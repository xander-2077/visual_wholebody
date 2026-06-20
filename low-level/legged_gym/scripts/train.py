import numpy as np
import os
import shutil
from datetime import datetime
import isaacgym

from legged_gym import LEGGED_GYM_ROOT_DIR, LEGGED_GYM_ENVS_DIR
from legged_gym.envs import *
from legged_gym.utils import get_args, task_registry
import torch

def copy_training_files(log_dir):
    for rel_path in [
        "manip_loco/b1z1_config.py",
        "manip_loco/manip_loco.py",
    ]:
        src = os.path.join(LEGGED_GYM_ENVS_DIR, rel_path)
        shutil.copy2(src, os.path.join(log_dir, os.path.basename(src)))

def train(args):
    log_name = "{}-{}".format(datetime.now().strftime("%y%m%d_%H%M%S"), args.exptid)
    log_pth = LEGGED_GYM_ROOT_DIR + "/logs/{}/".format(args.proj_name) + log_name
    try:
        os.makedirs(log_pth)
    except:
        pass
    copy_training_files(log_pth)
    if args.debug:
        mode = "disabled"
        args.rows = 6
        args.cols = 2
        args.num_envs = 128

    env, env_cfg = task_registry.make_env(name=args.task, args=args)
    ppo_runner, train_cfg, _ = task_registry.make_alg_runner(log_root = log_pth, env=env, name=args.task, args=args)
    ppo_runner.learn(num_learning_iterations=train_cfg.runner.max_iterations, init_at_random_ep_len=True)

if __name__ == '__main__':
    args = get_args()
    train(args)
