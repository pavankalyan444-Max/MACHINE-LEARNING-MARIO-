# Mario DQN

This project trains and evaluates a DQN agent in the Gym Super Mario Bros. environment.

## Setup (Windows)

Use Python 3.12 and create a virtual environment:

```powershell
py -3.12 -m venv mario_env
.\mario_env\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

The project uses the legacy Gym API exposed by `gym-super-mario-bros`; the pinned Gym, Mario, and NumPy versions keep that API compatible. For GPU acceleration, install the appropriate PyTorch build for your CUDA setup using the official PyTorch installation instructions.

## Run

Start training with `python train_parallel.py` or resume with `python train_resume.py`. The resume script lists checkpoints in `trained_models` and prompts for a number; press Enter to select the original default when it is available, or enter `q` to cancel. To skip the menu, use `--checkpoint mario_checkpoint_100000.pth` or provide a path. Missing checkpoints now raise an error instead of silently starting a new model.

The following options apply to both parallel training scripts:

```powershell
# Train seven games without display windows, on world 1 stage 2
python train_parallel.py --num-envs 7 --headless --level SuperMarioBros-1-2-v0

# Resume an existing model for 10,000 additional vector steps
python train_resume.py --checkpoint mario_checkpoint_100000.pth --steps 10000 --headless
```

By default the display stays enabled and the original `SuperMarioBros-v0` environment is used. Use a valid Gym environment ID such as `SuperMarioBros-1-2-v0` for a specific level. Each run uses one configured level; it does not switch levels automatically. To resume on the same level, pass the same `--level` option; newly saved checkpoints include their level and warn if it differs.

The agent is evaluated separately every 10,000 vector steps by default, using greedy actions for three episodes of up to 3,000 steps each. A vector step advances each training game once. Evaluation pauses training, does not add experiences to the replay buffer, and does not change epsilon. Adjust it with `--eval-every`, `--eval-episodes`, and `--eval-max-steps`; `--eval-every 0` disables evaluation. The best checkpoint **for each run** is saved in `trained_models`, ranked by completion rate, then average maximum distance, then average reward. Its menu entry includes evaluation results. Checkpoints from before this feature generally do not have evaluation results.

Training metrics and evaluation results are written to separate run directories under `runs`. To view the TensorBoard dashboard, open another terminal in the project folder:

```powershell
.\mario_env\Scripts\tensorboard.exe --logdir runs --reload_interval 2
```

Open the address printed by TensorBoard in your browser and enable auto-refresh in TensorBoard's settings. Training metrics are recorded on the first step, every 10 vector steps, or after five seconds while training is advancing, whichever comes first. The writer flushes at least every two seconds; browser refresh and backend reload can add a short delay. Restart training to use this logging behavior. Training charts include rolling episode reward, distance, length and completion rate, plus loss, epsilon, replay size and experiences per second. Episode metrics appear after an episode finishes; evaluation charts update only after each evaluation. The graph does not recreate metrics from training before TensorBoard logging was added. In headless mode, `train_resume.py` saves on Ctrl+C; `train_parallel.py` retains its existing periodic and normal-exit saves, so Ctrl+C can lose progress since the last checkpoint.

These options and the dashboard apply to `train_parallel.py` and `train_resume.py`; the older single-game `train.py` remains unchanged. `evaluate.py` also remains a separate evaluation script with its own configured checkpoint.

The parallel training scripts run each game in a separate CPU process, while the DQN uses the GPU when CUDA is available. Set `NUM_ENVS` near the top of the training script to choose the number of games (the default is 10). Try 7 to 10 first and compare the reported training speed; more games also use more memory and may not be faster. The game grid currently displays up to 10 games. On Windows, start these scripts from their normal command line entry points so worker processes can start correctly.

The replay buffer defaults to 10,000 transitions to avoid the high memory usage of retaining 100,000 full frame-stack transitions. Model checkpoints, the virtual environment, and Python cache files are local/generated data and should not be committed.
