import cv2
import numpy as np
from collections import deque

import gym
import gym_super_mario_bros

from nes_py.wrappers import JoypadSpace
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT


class MarioEnvironment:

    def __init__(self, level="SuperMarioBros-v0"):

        mario_env = gym_super_mario_bros.make(
            level
        )

        # Remove Gym TimeLimit because nes-py uses old API
        if hasattr(mario_env, "env"):
            mario_env = mario_env.env

        self.env = JoypadSpace(
            mario_env,
            SIMPLE_MOVEMENT
        )

        self.num_actions = self.env.action_space.n

        self.frame_stack = deque(maxlen=4)
        self.num_frames = 4

        # Keep original RGB observation for visualization
        self.latest_rgb_frame = None

    def reset(self):

        observation = self.env.reset()

        # Save ORIGINAL RGB frame
        self.latest_rgb_frame = observation.copy()

        processed_frame = self._process_frame(
            observation
        )

        self.frame_stack.clear()

        for _ in range(self.num_frames):
            self.frame_stack.append(
                processed_frame
            )

        state = np.stack(
            self.frame_stack,
            axis=0
        )

        return state, {}

    def step(self, action):

        observation, reward, done, info = self.env.step(
            int(action)
        )

        # Save ORIGINAL RGB frame
        self.latest_rgb_frame = observation.copy()

        processed_frame = self._process_frame(
            observation
        )

        self.frame_stack.append(
            processed_frame
        )

        state = np.stack(
            self.frame_stack,
            axis=0
        )

        terminated = bool(done)
        truncated = False

        return (
            state,
            reward,
            terminated,
            truncated,
            info
        )

    def _process_frame(self, frame):

        # CNN input remains grayscale
        gray = cv2.cvtColor(
            frame,
            cv2.COLOR_RGB2GRAY
        )

        resized = cv2.resize(
            gray,
            (84, 84),
            interpolation=cv2.INTER_AREA
        )

        return resized.astype(
            np.uint8
        )

    def get_rgb_frame(self):

        return self.latest_rgb_frame

    def render(self):

        return self.env.render()

    def close(self):

        self.env.close()

    def get_num_actions(self):

        return self.num_actions


if __name__ == "__main__":

    env = MarioEnvironment()

    print(
        "Number of actions:",
        env.get_num_actions()
    )

    state, info = env.reset()

    print(
        "CNN state:",
        state.shape,
        state.dtype
    )

    print(
        "RGB frame:",
        env.get_rgb_frame().shape,
        env.get_rgb_frame().dtype
    )

    env.close()