# Training a universal low-level policy

## Code structure
`legged_gym/envs` contains environment-related codes.

`legged_gym/scripts` contains train and test scripts.

## Train

The environment related code is `legged_gym/legged_gym/envs/manip_loco/manip_loco.py`, and the related config for b1z1 hardware is in `legged_gym/legged_gym/envs/b1z1/b1z1_config.py`.

```bash
cd legged_gym/scripts
python train.py --headless --exptid SOME_YOUR_DESCRIPTION --proj_name b1z1-low --task b1z1 --sim_device cuda:0 --rl_device cuda:0 --observe_gait_commands

# resume
python train.py \
    --headless \
    --task b1z1 \
    --proj_name b2z1-low \
    --resume \
    --resumeid 260621_090125-b2z1_test \
    --checkpoint 33600 \
    --exptid NEW_EXPT_NAME \
    --sim_device cuda:0 \
    --rl_device cuda:0 \
    --observe_gait_commands
```
- `--debug` disables wandb and set a small number of envs for faster execution.
- `--headless` disables rendering, typically used when you train model.
- `--proj_name` the folder containing all your logs and wandb project name. `manip-loco` is default.
- Logs are saved under `logs/<proj_name>/<YYMMDD_HHMMSS>-<exptid>`, for example `logs/b1z1-low/260620_110300-b1z1_test`.
- Resume logs are named automatically as `logs/<proj_name>/<YYMMDD_HHMMSS>-resume-<resumeid>-ckpt_<checkpoint>-<exptid>`. You can also resume directly from a checkpoint file with `--resume_path low-level/logs/b2z1-low/260621_090125-b2z1_test/model_33600.pt --exptid new_expt_name`.
- `--observe_gait_commands` is for tracking specific gait commands and learning the trotting behavior.

Check `legged_gym/legged_gym/utils/helpers.py` for all command line args.

## Play
Use `--load_run` to specify the trained run directory. It can be an absolute path, a path relative to the workspace root, or a run name under `logs/<proj_name>/`.

```bash
cd legged_gym/scripts
python play.py --load_run low-level/logs/b1z1-low/260620_110300-b1z1_test --task b1z1 --proj_name b1z1-low --checkpoint 64000 --observe_gait_commands
```
- `--exptid` is no longer required when `--load_run` is provided.
- You can also pass only the run folder name if it is under `logs/<proj_name>/`, for example `--load_run 260620_110300-b1z1_test --proj_name b1z1-low`.
- Use `--fix_base_command` to keep the base command fixed at `[0, 0, 0]` and inspect EEF tracking only. To use a non-zero fixed base command, pass `--base_command vx vy yaw`, for example `--fix_base_command --base_command 0.2 0.0 0.0`.

Use `--sim_device cpu --rl_device cpu` in case not enough GPU memory.

## Curriculum 配置

这里只保留当前代码会自动更新的 curriculum/schedule。配置里仅作为 `# curriculum:` 注释记录的候选值、未被消费的 schedule、当前关闭的 terrain curriculum 不列入本节。

`global_steps` 是 env policy step；当前 `runner.num_steps_per_env = 24`，所以 `20000 * 24` 个 `global_steps` 对应约 `20000` 个 PPO iteration。

| 模块 | 配置项 | 当前值 | 逻辑/含义 | 当前状态 |
| --- | --- | --- | --- | --- |
| EEF 轨迹时间 | `goal_ee.traj_time_init` -> `goal_ee.traj_time` | `[3, 6]` s -> `[1, 3]` s | `_resample_ee_goal_timings()` 按 progress 线性插值轨迹时间范围；早期目标慢，后期目标快 | 生效 |
| EEF 速度上限 | `goal_ee.max_ee_cart_goal_speed` | `0.25` -> `0.75` m/s | 同一个 progress 线性插值最大 EEF 笛卡尔目标速度；若路径长度要求更久，会把实际 `traj_time` 拉长到 `path_length / max_speed` | 生效 |
| EEF curriculum 进度 | `goal_ee.traj_time_curriculum_steps` | `20000 * 24` | `progress = clip(global_steps / traj_time_curriculum_steps, 0, 1)`，约 `20000` PPO iteration 拉满 | 生效 |
| Play 终态 curriculum | `scripts/play.py::use_final_ee_goal_curriculum()` | 使用 `traj_time` 和 `max_ee_cart_goal_speed[-1]` | play 时把 `traj_time_init` 改成最终 `traj_time`，并把速度范围固定为最终最大速度，避免从慢速初始 curriculum 开始 | 生效 |
| PPO value mixing | `algorithm.mixing_schedule` | `[1.0, 0, 3000]` | `value_mixing_ratio = clip((counter - 0) / 3000, 0, 1) * 1.0` | 生效 |
| PPO privileged regularization | `algorithm.priv_reg_coef_schedual` | `[0, 0.1, 3000, 7000]` | 第 `3000` 个 update 后开始，`7000` 个 update 内从 `0` 线性升到 `0.1` | 生效 |

EEF timing curriculum 会额外记录到 TensorBoard 的 `Curriculum/*` 标签下：`ee_traj_time_uniform_count`、`ee_traj_time_path_count`、`ee_traj_time_sample_count`、`ee_traj_time_uniform_frac`、`ee_traj_time_path_frac`。其中 `path_*` 表示因为路径长度和最大速度约束而拉长轨迹时间的采样比例。

## Suggestions
To choose a good low-level policy that can be further used for training the high-level policy, we suggest you deploy the low-level policy first, and see if it goes well before training a high-level policy.

## 训练指标

`Episode_metric/*` 是各 reward term 对应的原始误差或物理量，`Episode_rew/*` 是乘上 reward scale 之后的奖励贡献。判断行为质量时优先看 `Episode_metric/*`；判断各项 reward 权重是否平衡时看 `Episode_rew/*`。

重点关注的指标：

- `Episode_metric/metric_tracking_ee_world`：EEF 位置跟踪误差，越低越好。
- `Episode_rew/rew_tracking_ee_world`：缩放后的 EEF 跟踪奖励，越高越好。
- `Episode_metric/metric_tracking_ee_vel`：EEF local velocity 对平滑 target local velocity 的跟踪误差，越低越好。
- `Episode_rew/rew_tracking_ee_vel`：缩放后的 EEF 速度跟踪辅助奖励，权重应明显小于位置跟踪。
- `Episode_metric/metric_tracking_lin_vel_max`：前向速度跟踪得分，越接近 1 越好。
- `Episode_metric/metric_tracking_ang_vel`：yaw 角速度跟踪平方误差，越低越好。
- `Episode_metric/metric_collision`：trunk/thigh/calf 的碰撞计数，应接近 0。
- `Episode_metric/metric_base_height`：实际 base 高度，用来和 `base_height_target` 对比。
- `Episode_metric/metric_pitch`：超过配置阈值后的 pitch 误差，应接近 0。
- `Episode_metric/metric_roll`、`metric_lin_vel_z`、`metric_ang_vel_xy`：base 稳定性指标，越低越好。
- `Train/mean_episode_length` 和 `Train/dones`：整体稳定性；episode 越长、done 越少越好。

控制平滑性相关指标包括 `metric_torques`、`metric_dof_acc`、`metric_action_rate`、`metric_delta_torques`、`metric_feet_contact_forces`、`metric_feet_drag` 和 `metric_feet_jerk`。在跟踪效果变好的同时，这些指标不应明显发散或持续升高。

## B1Z1 Low-Level ABI

以 `ManipLoco.compute_observations()` / `step()` 为准：

当前 proprio 表：

| slice | dim | 内容 | scale | noise scale |
| --- | ---: | --- | --- | --- |
| `0:2` | 2 | `roll, pitch` | 原值 | `0` |
| `2:5` | 3 | base angular velocity，本体系 | `obs_scales.ang_vel = 1.0` | `0.2 * noise_level` |
| `5:17` | 12 | 腿部 `reindex(dof_pos - default_dof_pos)` | `obs_scales.dof_pos = 1.0` | `0.01 * noise_level` |
| `17:23` | 6 | 机械臂 `reindex(dof_pos - default_dof_pos)`，不含 gripper | `1.0` | `0` |
| `23:35` | 12 | 腿部 `reindex(dof_vel)` | `obs_scales.dof_vel = 0.05` | `1.5 * 0.05 * noise_level` |
| `35:41` | 6 | 机械臂 `reindex(dof_vel)`，不含 gripper | `0.05` | `0` |
| `41:53` | 12 | 上一帧腿部 action：`reindex(action_history[-1])[:12]` | 原值 | `0` |
| `53:57` | 4 | 足端接触：`reindex_feet(foot_contacts_from_sensor)` | 原值 | `0` |
| `57:60` | 3 | 底盘 command：`[lin_vel_x, lin_vel_y, ang_vel_yaw]` | `commands_scale = [1, 1, 1]` | `0` |
| `60:63` | 3 | 机械臂 command/target：`ee_goal_local_cart`，相对 arm base 的 base-frame EEF 目标位置 | 米，原值 | `0` |
| `63:66` | 3 | 机械臂 command/target orientation 占位：当前恒为 `0` | `0` | `0` |
| `66:71` | 5 | 仅 `--observe_gait_commands`：`gait_index(1) + clock_inputs(4)` | 原值，clock 为正弦相位 | `0` |

完整 obs 布局：

| 模式 | 当前 proprio | priv | history | env 返回 obs | JIT/高层 history 输入 |
| --- | ---: | ---: | ---: | ---: | ---: |
| 默认 | 66 | 18 | `10 * 66 = 660` | `66 + 18 + 660 = 744` | `66 + 660 = 726` |
| `--observe_gait_commands` | 71 | 18 | `10 * 71 = 710` | `71 + 18 + 710 = 799` | `71 + 710 = 781` |

`priv = mass_params(5) + friction(1) + leg_motor_strength_delta(12)`。训练时 `hist_encoding=False` 使用 `当前 proprio + priv`，`hist_encoding=True` 和部署/JIT 使用 `当前 proprio + history`，因此导出或高层调用时要去掉 priv 中段。

Command/target 输入：

| 项 | dim | 内容 | 范围/缩放 | 备注 |
| --- | ---: | --- | --- | --- |
| `obs[:, 57:60]` / `commands[:, :3]` | 3 | 底盘 command：`[lin_vel_x, lin_vel_y, ang_vel_yaw]` | `lin_vel_x in [-0.8, 0.8] m/s`，`lin_vel_y = 0`，`ang_vel_yaw in [-1, 1] rad/s` | 代码变量 `self.commands` 只包含这 3 维；前 `5000*24` global steps 只采正向 `lin_vel_x`；小 command 会整体置零 |
| `obs[:, 60:63]` / `ee_goal_local_cart` | 3 | 机械臂 EEF target position | 相对 arm base 的 base-frame 位置，单位 m | 这是 agent 输入中的机械臂 command/target，不在 `self.commands` 变量里 |
| `obs[:, 63:66]` | 3 | 机械臂 EEF target orientation 占位 | 当前恒为 `0` | 预留 ABI 位，当前未提供真实姿态目标 |

`ee_goal_local_vel` 只在训练环境内部由 target position 差分得到，并用于 `tracking_ee_vel` 辅助 reward；它不进入 actor obs，也不增加部署输入 ABI。

Action 输出：

| 项 | dim | 内容 | 范围/缩放 | 备注 |
| --- | ---: | --- | --- | --- |
| `actions[:, 0:12]` | 12 | 腿部目标关节偏移 | action scale `[0.4, 0.45, 0.45] * 4` | 经过 `_reindex_all()` 对齐仿真 DOF |
| `actions[:, 12:18]` | 6 | 机械臂 action head 输出 | 配置 scale `[2.1, 0.6, 0.6, 0, 0, 0]` | 当前 `step()` 先清零，机械臂实际由内部 IK 跟踪 EEF target |

噪声默认关闭：`noise.add_noise = False`。如果要打开噪声，先修正 `_get_noise_scale_vec()` 中旧的 height slice 覆盖逻辑；当前 `terrain.measure_heights=True`，但 `ManipLoco` obs 未拼 height measurements，直接启用会把 `48:235` 误设为 height noise。

对齐检查：obs 中 DOF、last action、feet contact 和 action 输出都必须沿用 `_reindex_all()` / `_reindex_feet()`，不要手工改关节顺序。
