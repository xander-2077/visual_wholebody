# Training a universal low-level policy

## Code structure
`legged_gym/envs` contains environment-related codes.

`legged_gym/scripts` contains train and test scripts.

## Train

The environment related code is `legged_gym/legged_gym/envs/manip_loco/manip_loco.py`, and the related config for b1z1 hardware is in `legged_gym/legged_gym/envs/b1z1/b1z1_config.py`.

```bash
cd legged_gym/scripts
python train.py --headless --exptid SOME_YOUR_DESCRIPTION --proj_name b1z1-low --task b1z1 --sim_device cuda:0 --rl_device cuda:0 --observe_gait_commands
```
- `--debug` disables wandb and set a small number of envs for faster execution.
- `--headless` disables rendering, typically used when you train model.
- `--proj_name` the folder containing all your logs and wandb project name. `manip-loco` is default.
- Logs are saved under `logs/<proj_name>/<YYMMDD_HHMMSS>-<exptid>`, for example `logs/b1z1-low/260620_110300-b1z1_test`.
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

Use `--sim_device cpu --rl_device cpu` in case not enough GPU memory.

## Suggestions
To choose a good low-level policy that can be further used for training the high-level policy, we suggest you deploy the low-level policy first, and see if it goes well before training a high-level policy.
