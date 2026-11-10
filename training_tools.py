import argparse
import os
import re
import time
import uuid
from collections import deque
from pathlib import Path

import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter

from checkpoint_utils import MODEL_DIR
from environment import MarioEnvironment

METRICS_EVERY_STEPS = 10
METRICS_INTERVAL_SECONDS = 5
TENSORBOARD_FLUSH_SECONDS = 2


def parse_training_options(description, num_envs, steps, resume=False):
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--num-envs", type=int, default=num_envs)
    parser.add_argument("--steps", type=int, default=steps,
                        help="Vector steps to train (additional steps when resuming)")
    parser.add_argument("--headless", action="store_true")
    parser.add_argument("--level", default="SuperMarioBros-v0",
                        help="Gym environment ID, e.g. SuperMarioBros-1-2-v0")
    parser.add_argument("--eval-every", type=int, default=10000,
                        help="Evaluate every N vector steps; 0 disables evaluation")
    parser.add_argument("--eval-episodes", type=int, default=3)
    parser.add_argument("--eval-max-steps", type=int, default=3000)
    if resume:
        parser.add_argument("--checkpoint", help="Checkpoint path; skips the interactive menu")
    options = parser.parse_args()
    if options.num_envs < 1:
        parser.error("--num-envs must be at least 1")
    if options.steps < 1:
        parser.error("--steps must be at least 1")
    if options.eval_every < 0:
        parser.error("--eval-every cannot be negative")
    if options.eval_episodes < 1 or options.eval_max_steps < 1:
        parser.error("Evaluation episodes and max steps must be at least 1")
    return options


def choose_checkpoint(default):
    directory = Path(MODEL_DIR)
    checkpoints = sorted(directory.glob("*.pth"), key=lambda path: path.stat().st_mtime, reverse=True)
    if not checkpoints:
        raise FileNotFoundError(f"No checkpoints found in {directory}")
    print("\nAvailable checkpoints (newest first):")
    for index, path in enumerate(checkpoints, 1):
        modified = time.strftime("%Y-%m-%d %H:%M", time.localtime(path.stat().st_mtime))
        match = re.search(r"checkpoint_(?:interrupt_)?(\d+)", path.name)
        details = f"step {match.group(1)}" if match else "step not in filename"
        if path.name.startswith("best_"):
            checkpoint = torch.load(path, map_location="cpu", weights_only=True)
            details = f"step {checkpoint.get('step', 'unknown')}, evaluation {checkpoint.get('evaluation', 'unknown')}"
        print(f"  {index:2d}. {path.name}  [{modified}; {details}]")
    default_path = Path(default)
    default_index = next((i for i, path in enumerate(checkpoints, 1)
                          if path == default_path), None)
    while True:
        suffix = f" [default {default_index}]" if default_index else ""
        choice = input(f"Select a checkpoint number{suffix} (q to cancel): ").strip()
        if choice.lower() == "q":
            raise SystemExit("Checkpoint selection cancelled.")
        if not choice and default_index:
            return str(checkpoints[default_index - 1])
        if choice.isdigit() and 1 <= int(choice) <= len(checkpoints):
            return str(checkpoints[int(choice) - 1])
        print("Enter a listed number or q to cancel.")


def evaluate_policy(agent, level, episodes, max_steps):
    env = MarioEnvironment(level=level)
    rewards, distances, successes, lengths = [], [], [], []
    was_training = agent.policy_net.training
    agent.policy_net.eval()
    try:
        for _ in range(episodes):
            state, _ = env.reset()
            total_reward = 0.0
            max_distance = 0
            success = False
            for step in range(1, max_steps + 1):
                with torch.no_grad():
                    state_tensor = torch.as_tensor(
                        state, dtype=torch.float32, device=agent.device
                    ).unsqueeze(0) / 255.0
                    action = int(agent.policy_net(state_tensor).argmax(dim=1).item())
                state, reward, terminated, truncated, info = env.step(action)
                total_reward += float(reward)
                max_distance = max(max_distance, int(info.get("x_pos", 0)))
                success = bool(info.get("flag_get", False))
                if terminated or truncated or success:
                    break
            rewards.append(total_reward)
            distances.append(max_distance)
            successes.append(success)
            lengths.append(step)
    finally:
        agent.policy_net.train(was_training)
        env.close()
    return {
        "reward": float(np.mean(rewards)),
        "distance": float(np.mean(distances)),
        "completion_rate": float(np.mean(successes)),
        "length": float(np.mean(lengths)),
    }


class TrainingMonitor:
    def __init__(self, options, start_step=0):
        self.options = options
        self.run_id = time.strftime("%Y%m%d-%H%M%S") + "-" + uuid.uuid4().hex[:8]
        level_name = re.sub(r"[^A-Za-z0-9_-]", "_", options.level)
        self.log_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                    "runs", f"{level_name}-{self.run_id}")
        self.best_path = os.path.join(MODEL_DIR, f"best_{level_name}_{self.run_id}.pth")
        self.writer = SummaryWriter(
            log_dir=self.log_dir, flush_secs=TENSORBOARD_FLUSH_SECONDS
        )
        self.recent = deque(maxlen=100)
        self.best_score = None
        self.start_time = time.monotonic()
        self.last_metrics_time = None
        self.start_step = start_step
        self.last_eval_step = start_step
        print(f"TensorBoard log: {self.log_dir}")

    def close(self):
        self.writer.close()

    def record_episode(self, reward, distance, length, success):
        self.recent.append((reward, distance, length, float(success)))

    def update(self, agent, step, episode, loss):
        now = time.monotonic()
        if (self.last_metrics_time is None
                or step % METRICS_EVERY_STEPS == 0
                or now - self.last_metrics_time >= METRICS_INTERVAL_SECONDS):
            self.writer.add_scalar("train/epsilon", agent.epsilon, step)
            self.writer.add_scalar("train/episodes", episode, step)
            self.writer.add_scalar("train/replay_size", len(agent.memory), step)
            elapsed = max(now - self.start_time, 0.001)
            self.writer.add_scalar(
                "train/experiences_per_second",
                (step - self.start_step) * self.options.num_envs / elapsed, step
            )
            if loss is not None:
                self.writer.add_scalar("train/loss", loss, step)
            if self.recent:
                for name, value in zip(("reward", "distance", "length", "completion_rate"),
                                       np.mean(self.recent, axis=0)):
                    self.writer.add_scalar(f"train/episode_{name}", value, step)
            self.last_metrics_time = now
        if self.options.eval_every and step - self.last_eval_step >= self.options.eval_every:
            print(f"\nEvaluating at vector step {step}...")
            metrics = evaluate_policy(agent, self.options.level, self.options.eval_episodes,
                                      self.options.eval_max_steps)
            for name, value in metrics.items():
                self.writer.add_scalar(f"evaluation/{name}", value, step)
            self.writer.flush()
            print(f"Evaluation: {metrics}")
            score = (metrics["completion_rate"], metrics["distance"], metrics["reward"])
            if self.best_score is None or score > self.best_score:
                os.makedirs(MODEL_DIR, exist_ok=True)
                torch.save({
                    "policy_net": agent.policy_net.state_dict(),
                    "target_net": agent.target_net.state_dict(),
                    "optimizer": agent.optimizer.state_dict(),
                    "steps_done": agent.steps_done,
                    "epsilon": agent.epsilon,
                    "step": step,
                    "episode": episode,
                    "level": self.options.level,
                    "evaluation": metrics,
                }, self.best_path)
                self.best_score = score
                print(f"Best evaluation checkpoint: {self.best_path}")
            self.last_eval_step = step
