import os
import time
import unittest
from unittest.mock import patch

import numpy as np

from agent import DQNAgent
from brain_visualizer import BrainVisualizer
from checkpoint_utils import checkpoint_value
from parallel_env import ParallelMarioEnv


class FakeMarioEnvironment:

    def __init__(self, level="SuperMarioBros-v0"):
        self.reset_count = 0
        self.step_count = 0

    def get_num_actions(self):
        return 7

    def reset(self):
        self.reset_count += 1
        self.step_count = 0
        self.rgb_frame = np.full((2, 2, 3), self.reset_count, dtype=np.uint8)
        state = np.full((4, 2, 2), self.reset_count, dtype=np.uint8)
        return state, {"reset_count": self.reset_count}

    def get_rgb_frame(self):
        return self.rgb_frame

    def step(self, action):
        self.step_count += 1
        state = np.full((4, 2, 2), 10 + self.step_count, dtype=np.uint8)
        self.rgb_frame = np.full(
            (2, 2, 3),
            10 + self.step_count,
            dtype=np.uint8
        )
        done = self.step_count == 1
        return state, 1.0, done, False, {
            "step_count": self.step_count,
            "pid": os.getpid()
        }

    def close(self):
        pass


class SlowMarioEnvironment(FakeMarioEnvironment):
    def step(self, action):
        started = time.monotonic()
        time.sleep(0.3)
        state, reward, terminated, truncated, info = super().step(action)
        info["started"] = started
        info["finished"] = time.monotonic()
        return state, reward, terminated, truncated, info


class FailingMarioEnvironment(FakeMarioEnvironment):
    def step(self, action):
        raise ValueError("test worker failure")


class CoreBehaviorTests(unittest.TestCase):

    def test_single_action_uses_batch_policy(self):
        agent = object.__new__(DQNAgent)
        observed = {}

        def act_batch(states):
            observed["states"] = states
            return np.array([3]), np.zeros((1, 7))

        agent.act_batch = act_batch
        state = np.zeros((4, 84, 84), dtype=np.uint8)

        self.assertEqual(agent.act(state), 3)
        self.assertEqual(observed["states"].shape, (1, 4, 84, 84))

    def test_parallel_environment_autoresets_and_preserves_terminal_state(self):
        with patch("parallel_env.MarioEnvironment", FakeMarioEnvironment), \
                patch("parallel_env.cv2.namedWindow"), \
                patch("parallel_env.cv2.resizeWindow"), \
                patch("parallel_env.cv2.destroyWindow"):
            env = ParallelMarioEnv(num_envs=2)
            try:
                states, _ = env.reset()
                next_states, rewards, dones, infos, frames = env.step([1, 1])

                self.assertEqual(states.shape, (2, 4, 2, 2))
                self.assertEqual(rewards.tolist(), [1.0, 1.0])
                self.assertEqual(dones.tolist(), [True, True])
                self.assertEqual(
                    infos[0]["terminal_observation"][0, 0, 0],
                    11
                )
                self.assertEqual(next_states[0, 0, 0, 0], 2)
                self.assertEqual(frames[0][0, 0, 0], 2)
                self.assertEqual(infos[0]["reset_info"]["reset_count"], 2)
                self.assertNotEqual(infos[0]["pid"], os.getpid())
                self.assertNotEqual(infos[0]["pid"], infos[1]["pid"])
            finally:
                env.close()
            self.assertTrue(all(not process.is_alive() for process in env.processes))
            env.close()

    def test_parallel_environment_rejects_wrong_action_count(self):
        with patch("parallel_env.MarioEnvironment", FakeMarioEnvironment), \
                patch("parallel_env.cv2.namedWindow"), \
                patch("parallel_env.cv2.resizeWindow"), \
                patch("parallel_env.cv2.destroyWindow"):
            env = ParallelMarioEnv(num_envs=2)
            try:
                with self.assertRaises(ValueError):
                    env.step([1])
            finally:
                env.close()

    def test_headless_mode_skips_window_and_frames(self):
        with patch("parallel_env.MarioEnvironment", FakeMarioEnvironment), \
                patch("parallel_env.cv2.namedWindow") as window, \
                patch("parallel_env.cv2.destroyWindow") as destroy:
            env = ParallelMarioEnv(num_envs=2, headless=True)
            try:
                _, frames = env.reset()
                _, _, _, _, next_frames = env.step([1, 1])
                self.assertEqual(frames, [None, None])
                self.assertEqual(next_frames, [None, None])
            finally:
                env.close()
            window.assert_not_called()
            destroy.assert_not_called()

    def test_environment_steps_overlap(self):
        with patch("parallel_env.MarioEnvironment", SlowMarioEnvironment), \
                patch("parallel_env.cv2.namedWindow"), \
                patch("parallel_env.cv2.resizeWindow"), \
                patch("parallel_env.cv2.destroyWindow"):
            env = ParallelMarioEnv(num_envs=2)
            try:
                env.reset()
                _, _, _, infos, _ = env.step([1, 1])
                self.assertLess(
                    max(info["started"] for info in infos),
                    min(info["finished"] for info in infos)
                )
            finally:
                env.close()

    def test_worker_error_is_reported_and_workers_are_closed(self):
        with patch("parallel_env.MarioEnvironment", FailingMarioEnvironment), \
                patch("parallel_env.cv2.namedWindow"), \
                patch("parallel_env.cv2.resizeWindow"), \
                patch("parallel_env.cv2.destroyWindow"):
            env = ParallelMarioEnv(num_envs=2)
            env.reset()
            with self.assertRaisesRegex(RuntimeError, "test worker failure"):
                env.step([1, 1])
            self.assertTrue(all(not process.is_alive() for process in env.processes))

    def test_worker_exit_is_reported_and_workers_are_closed(self):
        with patch("parallel_env.MarioEnvironment", FakeMarioEnvironment), \
                patch("parallel_env.cv2.namedWindow"), \
                patch("parallel_env.cv2.resizeWindow"), \
                patch("parallel_env.cv2.destroyWindow"):
            env = ParallelMarioEnv(num_envs=2)
            env.reset()
            env.processes[0].terminate()
            env.processes[0].join()
            with self.assertRaisesRegex(RuntimeError, "worker 0 exited unexpectedly"):
                env.step([1, 1])
            self.assertTrue(all(not process.is_alive() for process in env.processes))

    def test_real_environments_work_with_agent_training(self):
        with patch("parallel_env.cv2.namedWindow"), \
                patch("parallel_env.cv2.resizeWindow"), \
                patch("parallel_env.cv2.destroyWindow"):
            env = ParallelMarioEnv(num_envs=7)
            try:
                agent = DQNAgent(num_actions=env.get_num_actions(), batch_size=2)
                states, frames = env.reset()
                actions, q_values = agent.act_batch(states)
                next_states, rewards, dones, infos, next_frames = env.step(actions)
                self.assertEqual(states.shape, (7, 4, 84, 84))
                self.assertEqual(next_states.shape, states.shape)
                self.assertEqual(q_values.shape, (7, env.get_num_actions()))
                self.assertEqual(len(frames), 7)
                self.assertEqual(len(next_frames), 7)
                self.assertEqual(len(infos), 7)
                for i in range(7):
                    agent.memory.push(
                        states[i], actions[i], rewards[i], next_states[i], dones[i]
                    )
                self.assertIsNotNone(agent.learn())
            finally:
                env.close()
            self.assertTrue(all(not process.is_alive() for process in env.processes))

    def test_visualizer_accepts_dimensions_and_closes(self):
        with patch("brain_visualizer.cv2.namedWindow"), \
                patch("brain_visualizer.cv2.resizeWindow"), \
                patch("brain_visualizer.cv2.destroyWindow") as destroy:
            visualizer = BrainVisualizer(width=320, height=240, num_envs=1)

            self.assertTrue(visualizer.running)
            self.assertEqual((visualizer.width, visualizer.height), (320, 240))

            visualizer.close()
            visualizer.close()

            destroy.assert_called_once_with(visualizer.window_name)
            self.assertFalse(visualizer.running)

    def test_visualizer_stops_on_escape(self):
        with patch("brain_visualizer.cv2.namedWindow"), \
                patch("brain_visualizer.cv2.resizeWindow"), \
                patch("brain_visualizer.cv2.imshow"), \
                patch("brain_visualizer.cv2.waitKey", return_value=27), \
                patch("brain_visualizer.cv2.destroyWindow") as destroy:
            visualizer = BrainVisualizer(num_envs=1)
            with patch.object(visualizer, "draw_text"), \
                    patch.object(visualizer, "draw_mario_grid"), \
                    patch.object(visualizer, "draw_brain"), \
                    patch.object(visualizer, "draw_hud"):
                visualizer.update(
                    frames=[],
                    q_values=np.zeros((1, 7)),
                    actions=np.zeros(1, dtype=np.int64),
                    epsilon=0.0,
                    rewards=np.zeros(1),
                    step=0
                )

            self.assertFalse(visualizer.running)
            visualizer.close()
            destroy.assert_called_once_with(visualizer.window_name)

    def test_checkpoint_key_compatibility(self):
        self.assertEqual(
            checkpoint_value({"policy_net": "new"}, "policy_net", "model_state_dict"),
            "new"
        )
        self.assertEqual(
            checkpoint_value(
                {"model_state_dict": "legacy"},
                "policy_net",
                "model_state_dict"
            ),
            "legacy"
        )
        with self.assertRaises(KeyError):
            checkpoint_value({}, "policy_net", "model_state_dict")


if __name__ == "__main__":
    unittest.main()
