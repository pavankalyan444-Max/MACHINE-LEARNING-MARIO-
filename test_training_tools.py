import tempfile
import time
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import torch
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

from training_tools import TrainingMonitor, choose_checkpoint, evaluate_policy


class EvaluationEnvironment:
    created = []

    def __init__(self, level):
        self.level = level
        self.closed = False
        self.created.append(self)

    def reset(self):
        self.steps = 0
        return np.zeros((4, 2, 2), dtype=np.uint8), {}

    def step(self, action):
        self.steps += 1
        assert action == 1
        return (np.zeros((4, 2, 2), dtype=np.uint8), 2.0, False, False,
                {"x_pos": self.steps * 10, "flag_get": self.steps == 2})

    def close(self):
        self.closed = True


class Policy(torch.nn.Module):
    def forward(self, states):
        return torch.tensor([[0.0, 1.0]], device=states.device)


class TrainingToolsTests(unittest.TestCase):
    def test_metrics_use_step_and_time_intervals(self):
        options = SimpleNamespace(level="SuperMarioBros-v0", num_envs=2, eval_every=0)
        agent = SimpleNamespace(epsilon=0.2, memory=[])
        with patch("training_tools.SummaryWriter") as writer, \
                patch("training_tools.time.monotonic", side_effect=[100, 100, 104, 105, 106]):
            monitor = TrainingMonitor(options)
            monitor.update(agent, 1, 0, None)
            monitor.update(agent, 2, 0, None)
            monitor.update(agent, 3, 0, None)
            monitor.update(agent, 10, 0, None)
            monitor.close()
            writer.assert_called_once_with(log_dir=monitor.log_dir, flush_secs=2)
            epsilon_calls = [call for call in writer.return_value.add_scalar.call_args_list
                             if call.args[0] == "train/epsilon"]
            self.assertEqual([call.args[2] for call in epsilon_calls], [1, 3, 10])

    def test_metrics_reach_event_file_while_writer_is_open(self):
        options = SimpleNamespace(level="SuperMarioBros-v0", num_envs=2, eval_every=0)
        agent = SimpleNamespace(epsilon=0.2, memory=[])
        with tempfile.TemporaryDirectory() as directory, \
                patch("training_tools.__file__", str(Path(directory) / "training_tools.py")):
            monitor = TrainingMonitor(options)
            try:
                monitor.update(agent, 1, 0, None)
                accumulator = EventAccumulator(monitor.log_dir)
                deadline = time.monotonic() + 8
                while True:
                    accumulator.Reload()
                    if "train/epsilon" in accumulator.Tags()["scalars"]:
                        break
                    if time.monotonic() >= deadline:
                        self.fail("Training metrics were not flushed while the writer was open")
                    time.sleep(0.1)
                self.assertEqual(accumulator.Scalars("train/epsilon")[0].step, 1)
            finally:
                monitor.close()

    def test_evaluation_is_greedy_bounded_and_preserves_agent(self):
        agent = SimpleNamespace(policy_net=Policy(), device="cpu", epsilon=0.3, steps_done=123)
        agent.policy_net.train()
        with patch("training_tools.MarioEnvironment", EvaluationEnvironment):
            metrics = evaluate_policy(agent, "SuperMarioBros-1-2-v0", 2, 3)
            limited = evaluate_policy(agent, "SuperMarioBros-1-2-v0", 1, 1)
        self.assertEqual(metrics, {"reward": 4.0, "distance": 20.0,
                                   "completion_rate": 1.0, "length": 2.0})
        self.assertEqual(limited["completion_rate"], 0.0)
        self.assertEqual(limited["length"], 1.0)
        self.assertTrue(agent.policy_net.training)
        self.assertEqual((agent.epsilon, agent.steps_done), (0.3, 123))
        self.assertTrue(all(env.closed for env in EvaluationEnvironment.created))
        self.assertEqual(EvaluationEnvironment.created[-1].level, "SuperMarioBros-1-2-v0")

    def test_monitor_saves_only_improving_evaluations(self):
        options = SimpleNamespace(level="SuperMarioBros-v0", num_envs=2, eval_every=2,
                                  eval_episodes=1, eval_max_steps=10)
        agent = SimpleNamespace(
            policy_net=torch.nn.Linear(1, 1), target_net=torch.nn.Linear(1, 1),
            epsilon=0.2, steps_done=9, memory=[]
        )
        agent.optimizer = torch.optim.Adam(agent.policy_net.parameters())
        first = {"reward": 1.0, "distance": 20.0, "completion_rate": 0.0, "length": 10.0}
        worse = {"reward": 10.0, "distance": 10.0, "completion_rate": 0.0, "length": 10.0}
        better = {"reward": 0.0, "distance": 5.0, "completion_rate": 1.0, "length": 5.0}
        with tempfile.TemporaryDirectory() as directory, \
                patch("training_tools.MODEL_DIR", directory), \
                patch("training_tools.SummaryWriter") as writer, \
                patch("training_tools.evaluate_policy", side_effect=[first, worse, better]) as evaluate:
            monitor = TrainingMonitor(options, start_step=100)
            try:
                monitor.record_episode(4, 20, 10, True)
                monitor.update(agent, 101, 2, None)
                self.assertEqual(evaluate.call_count, 0)
                monitor.update(agent, 102, 2, None)
                self.assertEqual(torch.load(monitor.best_path, weights_only=True)["step"], 102)
                monitor.update(agent, 104, 3, 0.5)
                self.assertEqual(torch.load(monitor.best_path, weights_only=True)["step"], 102)
                monitor.update(agent, 106, 4, 0.5)
                saved = torch.load(monitor.best_path, weights_only=True)
                self.assertEqual(saved["step"], 106)
                self.assertEqual(saved["evaluation"], better)
                self.assertEqual(saved["level"], options.level)
                writer.return_value.add_scalar.assert_any_call("evaluation/completion_rate", 1.0, 106)
            finally:
                monitor.close()
            writer.return_value.close.assert_called_once()

    def test_checkpoint_menu_selects_and_displays_best_metrics(self):
        with tempfile.TemporaryDirectory() as directory:
            normal = Path(directory) / "mario_checkpoint_12.pth"
            normal.touch()
            best = Path(directory) / "best_test.pth"
            torch.save({"step": 34, "evaluation": {"completion_rate": 1.0}}, best)
            with patch("training_tools.MODEL_DIR", directory), \
                    patch("builtins.input", side_effect=["bad", ""]), \
                    patch("builtins.print") as output:
                self.assertEqual(choose_checkpoint(str(best)), str(best))
            self.assertIn("completion_rate", str(output.call_args_list))

    def test_checkpoint_menu_handles_empty_directory(self):
        with tempfile.TemporaryDirectory() as directory, patch("training_tools.MODEL_DIR", directory):
            with self.assertRaises(FileNotFoundError):
                choose_checkpoint("missing.pth")


if __name__ == "__main__":
    unittest.main()
