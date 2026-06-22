import numpy as np
import os
import shutil
from datetime import datetime
import isaacgym

from legged_gym import LEGGED_GYM_ROOT_DIR, LEGGED_GYM_ENVS_DIR
from legged_gym.envs import *
from legged_gym.utils import get_args, task_registry
from legged_gym.utils.helpers import get_load_path
import torch

def _checkpoint_from_model_path(path):
    model_name = os.path.basename(path)
    if model_name.startswith("model_") and model_name.endswith(".pt"):
        return model_name[len("model_"):-len(".pt")]
    return None

def _checkpoint_label(args):
    if getattr(args, "resume_path", ""):
        checkpoint = _checkpoint_from_model_path(args.resume_path)
        if checkpoint is not None:
            return checkpoint
    if args.checkpoint != -1:
        return str(args.checkpoint)
    if args.resumeid:
        try:
            resume_root = os.path.join(LEGGED_GYM_ROOT_DIR, "logs", args.proj_name, args.resumeid)
            checkpoint = _checkpoint_from_model_path(get_load_path(resume_root, checkpoint=-1))
            if checkpoint is not None:
                return checkpoint
        except Exception:
            pass
    return "latest"

def _resume_source_name(args):
    if getattr(args, "resume_path", ""):
        return os.path.basename(os.path.dirname(os.path.expanduser(args.resume_path)))
    if args.resumeid:
        return os.path.basename(os.path.normpath(args.resumeid))
    return "unknown"

def make_log_name(args):
    timestamp = datetime.now().strftime("%y%m%d_%H%M%S")
    is_resume = args.resume or bool(args.resumeid) or bool(getattr(args, "resume_path", ""))
    if not is_resume:
        return "{}-{}".format(timestamp, args.exptid)
    return "{}-resume-{}-ckpt_{}-{}".format(
        timestamp,
        _resume_source_name(args),
        _checkpoint_label(args),
        args.exptid,
    )

def copy_training_files(log_dir):
    for rel_path in [
        "manip_loco/b1z1_config.py",
        "manip_loco/manip_loco.py",
    ]:
        src = os.path.join(LEGGED_GYM_ENVS_DIR, rel_path)
        shutil.copy2(src, os.path.join(log_dir, os.path.basename(src)))

def train(args):
    log_name = make_log_name(args)
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
