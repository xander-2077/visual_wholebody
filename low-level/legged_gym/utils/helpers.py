import os
import copy
import torch
import numpy as np
import random
import sys
from isaacgym import gymapi
from isaacgym import gymutil

from legged_gym import LEGGED_GYM_ROOT_DIR, LEGGED_GYM_ENVS_DIR

def class_to_dict(obj) -> dict:
    if not  hasattr(obj,"__dict__"):
        return obj
    result = {}
    for key in dir(obj):
        if key.startswith("_"):
            continue
        element = []
        val = getattr(obj, key)
        if isinstance(val, list):
            for item in val:
                element.append(class_to_dict(item))
        else:
            element = class_to_dict(val)
        result[key] = element
    return result

def update_class_from_dict(obj, dict):
    for key, val in dict.items():
        attr = getattr(obj, key, None)
        if isinstance(attr, type):
            update_class_from_dict(attr, val)
        else:
            setattr(obj, key, val)
    return

def set_seed(seed):
    if seed == -1:
        seed = np.random.randint(0, 10000)
    print("Setting seed: {}".format(seed))
    
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def parse_sim_params(args, cfg):
    # code from Isaac Gym Preview 2
    # initialize sim params
    sim_params = gymapi.SimParams()

    # set some values from args
    if args.physics_engine == gymapi.SIM_FLEX:
        if args.device != "cpu":
            print("WARNING: Using Flex with GPU instead of PHYSX!")
    elif args.physics_engine == gymapi.SIM_PHYSX:
        sim_params.physx.use_gpu = args.use_gpu
        sim_params.physx.num_subscenes = args.subscenes
    sim_params.use_gpu_pipeline = args.use_gpu_pipeline

    # if sim options are provided in cfg, parse them and update/override above:
    if "sim" in cfg:
        gymutil.parse_sim_config(cfg["sim"], sim_params)

    # Override num_threads if passed on the command line
    if args.physics_engine == gymapi.SIM_PHYSX and args.num_threads > 0:
        sim_params.physx.num_threads = args.num_threads

    return sim_params

def get_load_path(root, checkpoint=-1, model_name_include="model"):
    if not os.path.isdir(root):  # use first 6 chars to mactch the run name
        model_name_cand = os.path.basename(root)
        model_parent = os.path.dirname(root)
        model_names = os.listdir(model_parent)
        model_names = [name for name in model_names if os.path.isdir(os.path.join(model_parent, name))]
        for name in model_names:
            if len(name) >= 6:
                if name[:6] == model_name_cand:
                    root = os.path.join(model_parent, name)
    if checkpoint==-1:
        models = [file for file in os.listdir(root) if model_name_include in file]
        models.sort(key=lambda m: '{0:0>15}'.format(m))
        model = models[-1]
    else:
        model = "model_{}.pt".format(checkpoint)

    load_path = os.path.join(root, model)
    return load_path

def update_cfg_from_args(env_cfg, cfg_train, args):
    # seed
    if env_cfg is not None:
        # num envs
        if args.num_envs is not None:
            env_cfg.env.num_envs = args.num_envs
        if args.stop_update_goal is not None:
            env_cfg.env.stop_update_goal = args.stop_update_goal
        if args.seed is not None:
            env_cfg.seed = args.seed
        if args.rows is not None:
            env_cfg.terrain.num_rows = args.rows
        if args.cols is not None:
            env_cfg.terrain.num_cols = args.cols
        if args.observe_gait_commands:
            env_cfg.env.observe_gait_commands = True
        if args.record_video:
            env_cfg.env.record_video = args.record_video
        if args.stand_by:
            env_cfg.env.stand_by = args.stand_by
        if getattr(args, "fix_base_command", False):
            env_cfg.env.fixed_base_command = list(args.base_command)
        if args.vel_obs:
            env_cfg.env.observe_velocities = args.vel_obs
        env_cfg.env.pitch_control = args.pitch_control
    if cfg_train is not None:
        if args.seed is not None:
            cfg_train.seed = args.seed
        # alg runner parameters
        if args.max_iterations is not None:
            cfg_train.runner.max_iterations = args.max_iterations
        if args.resume:
            cfg_train.runner.resume = args.resume
        if args.experiment_name is not None:
            cfg_train.runner.experiment_name = args.experiment_name
        if args.run_name is not None:
            cfg_train.runner.run_name = args.run_name
        if args.load_run is not None:
            cfg_train.runner.load_run = args.load_run
        if args.checkpoint is not None:
            cfg_train.runner.checkpoint = args.checkpoint

    return env_cfg, cfg_train

def _normalize_base_command_argv(argv):
    normalized = []
    i = 0
    while i < len(argv):
        if argv[i] == "--base_command" and i + 3 < len(argv):
            values = argv[i + 1:i + 4]
            try:
                [float(value) for value in values]
            except ValueError:
                normalized.append(argv[i])
                i += 1
                continue
            normalized.extend([argv[i], ",".join(values)])
            i += 4
            continue
        normalized.append(argv[i])
        i += 1
    return normalized

def _parse_base_command(value):
    if isinstance(value, (list, tuple)):
        values = value
    else:
        values = str(value).replace(",", " ").split()
    if len(values) != 3:
        raise ValueError("--base_command expects exactly 3 floats: lin_vel_x lin_vel_y ang_vel_yaw")
    return [float(value) for value in values]

def get_args(test=False):
    original_argv = sys.argv
    sys.argv = _normalize_base_command_argv(sys.argv)
    custom_parameters = [
        {"name": "--task", "type": str, "default": "widowGo1", "help": "Resume training or start testing from a checkpoint. Overrides config file if provided."},
        {"name": "--resume", "action": "store_true", "default": False,  "help": "Resume training from a checkpoint"},
        {"name": "--experiment_name", "type": str,  "help": "Name of the experiment to run or load. Overrides config file if provided."},
        {"name": "--run_name", "type": str,  "required": False,  "help": "Name of the run. Overrides config file if provided."},
        {"name": "--load_run", "type": str, "default": "", "help": "Name of the run to load when resume=True. If -1: will load the last run. Overrides config file if provided."},
        {"name": "--checkpoint", "type": int,"default": "-1",  "help": "Saved model checkpoint number. If -1: will load the last checkpoint. Overrides config file if provided."},
        {"name": "--resume_path", "type": str, "default": "", "help": "Checkpoint file path to resume from. Overrides --resumeid/--checkpoint when provided."},
        {"name": "--stop_update_goal", "action": "store_true", "help": "stop when update a new ee goal"},
        {"name": "--observe_gait_commands", "action": "store_true", "help": "if observe gait commands, ref to <walk these ways>"},
        
        {"name": "--exptid", "type": str,  "required": True if not test else False,  "help": "Experiment ID"},
        {"name": "--debug", "action": "store_true", "default": False, "help": "Disable wandb logging"},
        {"name": "--proj_name", "type": str,  "default": "b1z1-low", "help": "run folder name."},
        {"name": "--resumeid", "type": str, "help": "exptid"},

        {"name": "--headless", "action": "store_true", "default": True, "help": "Force display off at all times"},
        {"name": "--horovod", "action": "store_true", "default": False, "help": "Use horovod for multi-gpu training"},
        {"name": "--rl_device", "type": str, "default": "cuda:0", "help": 'Device used by the RL algorithm, (cpu, gpu, cuda:0, cuda:1 etc..)'},
        {"name": "--num_envs", "type": int, "help": "Number of environments to create. Overrides config file if provided."},
        {"name": "--seed", "type": int, "help": "Random seed. Overrides config file if provided."},
        {"name": "--max_iterations", "type": int, "help": "Maximum number of training iterations. Overrides config file if provided."},
        {"name": "--stochastic", "action": "store_true", "default": False, "help": "Use stochastic actions to play"},
        {"name": "--use_jit", "action": "store_true", "default": False,  "help": "Use jit to play"},
        {"name": "--record_video", "action": "store_true", "default": False,  "help": "Record video to play"},
        {"name": "--stand_by", "action": "store_true", "default": False,  "help": "Stand by to play"},
        {"name": "--fix_base_command", "action": "store_true", "default": False,  "help": "Fix base velocity command instead of sampling it"},
        {"name": "--base_command", "type": _parse_base_command, "default": [0.0, 0.0, 0.0], "help": "Fixed [lin_vel_x, lin_vel_y, ang_vel_yaw] command used with --fix_base_command"},
        {"name": "--flat_terrain", "action": "store_true", "default": False,  "help": "Flat the terrain"},
        {"name": "--pitch_control", "action": "store_true", "default": False,  "help": "Control Pitch"},
        {"name": "--vel_obs", "action": "store_true", "default": False,  "help": "Control Pitch"},
        
        {"name": "--rows", "type": int, "help": "num_rows."},
        {"name": "--cols", "type": int, "help": "num_cols"},
    ]
    # parse arguments
    try:
        args = gymutil.parse_arguments(
            description="RL Policy",
            custom_parameters=custom_parameters)
    finally:
        sys.argv = original_argv
    
    args.test = test

    # name allignment
    args.sim_device_id = args.compute_device_id
    args.sim_device = args.sim_device_type
    if args.sim_device=='cuda':
        args.sim_device += f":{args.sim_device_id}"
    return args

def export_policy_as_jit(actor_critic, path):
    if hasattr(actor_critic, 'memory_a'):
        # assumes LSTM: TODO add GRU
        exporter = PolicyExporterLSTM(actor_critic)
        exporter.export(path)
    else: 
        os.makedirs(path, exist_ok=True)
        path = os.path.join(path, 'policy_1.pt')
        model = copy.deepcopy(actor_critic.actor).to('cpu')
        traced_script_module = torch.jit.script(model)
        traced_script_module.save(path)


class PolicyExporterLSTM(torch.nn.Module):
    def __init__(self, actor_critic):
        super().__init__()
        self.actor = copy.deepcopy(actor_critic.actor)
        self.is_recurrent = actor_critic.is_recurrent
        self.memory = copy.deepcopy(actor_critic.memory_a.rnn)
        self.memory.cpu()
        self.register_buffer(f'hidden_state', torch.zeros(self.memory.num_layers, 1, self.memory.hidden_size))
        self.register_buffer(f'cell_state', torch.zeros(self.memory.num_layers, 1, self.memory.hidden_size))

    def forward(self, x):
        out, (h, c) = self.memory(x.unsqueeze(0), (self.hidden_state, self.cell_state))
        self.hidden_state[:] = h
        self.cell_state[:] = c
        return self.actor(out.squeeze(0))

    @torch.jit.export
    def reset_memory(self):
        self.hidden_state[:] = 0.
        self.cell_state[:] = 0.
 
    def export(self, path):
        os.makedirs(path, exist_ok=True)
        path = os.path.join(path, 'policy_lstm_1.pt')
        self.to('cpu')
        traced_script_module = torch.jit.script(self)
        traced_script_module.save(path)

    
