import multiprocessing as mp
import time
import traceback
from functools import partial

import cv2
import numpy as np

from environment import MarioEnvironment


def _environment_worker(connection, parent_connection, environment_factory, capture_frames):
    parent_connection.close()
    env = None
    try:
        env = environment_factory()
        connection.send(("ok", env.get_num_actions()))
        while True:
            command, value = connection.recv()
            if command == "close":
                break
            if command == "reset":
                state, _ = env.reset()
                result = (state, env.get_rgb_frame() if capture_frames else None)
            elif command == "step":
                state, reward, terminated, truncated, info = env.step(int(value))
                done = terminated or truncated
                frame = env.get_rgb_frame() if capture_frames else None
                if done:
                    info = dict(info)
                    info["terminal_observation"] = state
                    state, reset_info = env.reset()
                    info["reset_info"] = reset_info
                    frame = env.get_rgb_frame() if capture_frames else None
                result = (state, float(reward), bool(done), info, frame)
            else:
                raise ValueError(f"Unknown environment command: {command}")
            connection.send(("ok", result))
    except EOFError:
        pass
    except Exception:
        try:
            connection.send(("error", traceback.format_exc()))
        except (BrokenPipeError, EOFError, OSError):
            pass
    finally:
        if env is not None:
            env.close()
        connection.close()


class ParallelMarioEnv:

    def __init__(self, num_envs=10, level="SuperMarioBros-v0", headless=False):

        if num_envs <= 0:
            raise ValueError("num_envs must be greater than zero")

        self.num_envs = num_envs
        self.connections = []
        self.processes = []
        self.window_open = False
        self.closed = False

        print(f"Creating {num_envs} Mario environments...")

        self.window_name = "SUPER MARIO AI - 10 ENVIRONMENTS"
        self.window_width = 1280
        self.window_height = 900

        try:
            context = mp.get_context("spawn")
            for _ in range(num_envs):
                parent, child = context.Pipe()
                process = context.Process(
                    target=_environment_worker,
                    args=(child, parent, partial(MarioEnvironment, level=level), not headless),
                    daemon=True
                )
                try:
                    process.start()
                except BaseException:
                    parent.close()
                    child.close()
                    raise
                child.close()
                self.connections.append(parent)
                self.processes.append(process)

            action_counts = [self._receive(i) for i in range(num_envs)]
            self.num_actions = action_counts[0]
            if any(count != self.num_actions for count in action_counts):
                raise RuntimeError("Mario environments have different action counts")

            if not headless:
                cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
                self.window_open = True
                cv2.resizeWindow(
                    self.window_name,
                    self.window_width,
                    self.window_height
                )
        except BaseException:
            self.close()
            raise

        print("All environments created.")
        print("Actions:", self.num_actions)

    def _receive(self, index):
        try:
            status, result = self.connections[index].recv()
        except (EOFError, OSError) as exc:
            raise RuntimeError(
                f"Mario environment worker {index} exited unexpectedly"
            ) from exc
        if status != "ok":
            raise RuntimeError(f"Mario environment worker {index} failed:\n{result}")
        return result

    def _request_all(self, command, values):
        if self.closed:
            raise RuntimeError("Mario environments are closed")
        try:
            for index, (connection, value) in enumerate(zip(self.connections, values)):
                try:
                    connection.send((command, value))
                except (BrokenPipeError, EOFError, OSError) as exc:
                    raise RuntimeError(
                        f"Mario environment worker {index} exited unexpectedly"
                    ) from exc
            return [self._receive(i) for i in range(self.num_envs)]
        except BaseException:
            self.close()
            raise

    # ==========================================================
    # RESET
    # ==========================================================

    def reset(self):

        results = self._request_all("reset", [None] * self.num_envs)
        states, rgb_frames = zip(*results)
        return np.stack(states), list(rgb_frames)

    # ==========================================================
    # STEP
    # ==========================================================

    def step(self, actions):

        if len(actions) != self.num_envs:
            raise ValueError(
                f"Expected {self.num_envs} actions, received {len(actions)}"
            )

        results = self._request_all("step", actions)
        next_states, rewards, dones, infos, rgb_frames = zip(*results)

        return (
            np.stack(next_states),
            np.asarray(
                rewards,
                dtype=np.float32
            ),
            np.asarray(
                dones,
                dtype=np.bool_
            ),
            list(infos),
            list(rgb_frames)
        )

    # ==========================================================
    # RENDER 10 MARIO ENVIRONMENTS
    # ==========================================================

    def render_grid(
        self,
        frames,
        actions=None,
        rewards=None,
        step=0
    ):

        canvas = np.zeros(
            (
                self.window_height,
                self.window_width,
                3
            ),
            dtype=np.uint8
        )

        # ------------------------------------------------------
        # MAIN TITLE
        # ------------------------------------------------------

        cv2.putText(
            canvas,
            "SUPER MARIO AI",
            (20, 35),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (255, 255, 255),
            2,
            cv2.LINE_AA
        )

        cv2.putText(
            canvas,
            f"STEP: {step}",
            (1050, 35),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (180, 200, 220),
            1,
            cv2.LINE_AA
        )

        # ------------------------------------------------------
        # GRID
        # ------------------------------------------------------

        columns = 2
        rows = 5

        margin_x = 10
        margin_top = 48

        gap_x = 8
        gap_y = 8

        available_width = (
            self.window_width
            - 2 * margin_x
            - gap_x
        )

        available_height = (
            self.window_height
            - margin_top
            - 10
            - (rows - 1) * gap_y
        )

        panel_width = (
            available_width // columns
        )

        panel_height = (
            available_height // rows
        )

        # ------------------------------------------------------
        # DRAW EACH MARIO
        # ------------------------------------------------------

        for i in range(
            min(
                self.num_envs,
                len(frames)
            )
        ):

            row = i // columns
            col = i % columns

            x = (
                margin_x
                + col * (
                    panel_width
                    + gap_x
                )
            )

            y = (
                margin_top
                + row * (
                    panel_height
                    + gap_y
                )
            )

            self._draw_single_panel(
                canvas=canvas,
                frame=frames[i],
                x=x,
                y=y,
                width=panel_width,
                height=panel_height,
                index=i,
                action=(
                    actions[i]
                    if actions is not None
                    else None
                ),
                reward=(
                    rewards[i]
                    if rewards is not None
                    else None
                )
            )

        cv2.imshow(
            self.window_name,
            canvas
        )

        key = cv2.waitKey(1)

        if key == 27:

            cv2.destroyWindow(
                self.window_name
            )
            self.window_open = False

            raise KeyboardInterrupt

    # ==========================================================
    # DRAW ONE MARIO PANEL
    # ==========================================================

    def _draw_single_panel(
        self,
        canvas,
        frame,
        x,
        y,
        width,
        height,
        index,
        action=None,
        reward=None
    ):

        # ------------------------------------------------------
        # PANEL BORDER
        # ------------------------------------------------------

        cv2.rectangle(
            canvas,
            (x, y),
            (
                x + width - 1,
                y + height - 1
            ),
            (70, 85, 100),
            1,
            cv2.LINE_AA
        )

        header_height = 25

        # ------------------------------------------------------
        # HEADER
        # ------------------------------------------------------

        cv2.rectangle(
            canvas,
            (x + 1, y + 1),
            (
                x + width - 2,
                y + header_height
            ),
            (10, 15, 22),
            -1
        )

        cv2.putText(
            canvas,
            f"MARIO {index + 1}",
            (x + 8, y + 18),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.42,
            (245, 250, 255),
            1,
            cv2.LINE_AA
        )

        # ------------------------------------------------------
        # ACTION
        # ------------------------------------------------------

        if action is not None:

            action_names = [
                "NOOP",
                "RIGHT",
                "RIGHT + A",
                "RIGHT + B",
                "RIGHT + A + B",
                "A",
                "LEFT"
            ]

            action = int(action)

            if 0 <= action < len(action_names):

                action_text = action_names[action]

            else:

                action_text = str(action)

            cv2.putText(
                canvas,
                action_text,
                (
                    x + 105,
                    y + 18
                ),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.36,
                (80, 255, 120),
                1,
                cv2.LINE_AA
            )

        # ------------------------------------------------------
        # REWARD
        # ------------------------------------------------------

        if reward is not None:

            reward_text = f"R:{float(reward):.1f}"

            cv2.putText(
                canvas,
                reward_text,
                (
                    x + width - 70,
                    y + 18
                ),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.34,
                (220, 230, 240),
                1,
                cv2.LINE_AA
            )

        # ------------------------------------------------------
        # NO FRAME
        # ------------------------------------------------------

        if frame is None:
            return

        frame = np.asarray(frame)

        # ------------------------------------------------------
        # CONVERT FRAME TO BGR
        # ------------------------------------------------------

        if frame.ndim == 3:

            if frame.shape[2] == 3:

                frame = cv2.cvtColor(
                    frame,
                    cv2.COLOR_RGB2BGR
                )

            else:

                return

        elif frame.ndim == 2:

            frame = cv2.cvtColor(
                frame,
                cv2.COLOR_GRAY2BGR
            )

        else:

            return

        # ------------------------------------------------------
        # IMAGE AREA
        # ------------------------------------------------------

        image_x = x + 3
        image_y = y + header_height + 3

        image_width = max(
            1,
            width - 6
        )

        image_height = max(
            1,
            height - header_height - 6
        )

        # ------------------------------------------------------
        # RESIZE EXACTLY TO AVAILABLE AREA
        # ------------------------------------------------------

        resized = cv2.resize(
            frame,
            (
                image_width,
                image_height
            ),
            interpolation=cv2.INTER_NEAREST
        )

        # Safety check
        actual_height = resized.shape[0]
        actual_width = resized.shape[1]

        canvas[
            image_y:
            image_y + actual_height,
            image_x:
            image_x + actual_width
        ] = resized

    # ==========================================================
    # CLOSE
    # ==========================================================

    def close(self):

        if self.closed:
            return
        self.closed = True
        try:
            if self.window_open:
                cv2.destroyWindow(self.window_name)
                self.window_open = False
        finally:
            for connection in self.connections:
                try:
                    connection.send(("close", None))
                except (BrokenPipeError, EOFError, OSError):
                    pass
            deadline = time.monotonic() + 3
            for process in self.processes:
                process.join(max(0, deadline - time.monotonic()))
            for process in self.processes:
                if process.is_alive():
                    process.terminate()
                    process.join()
            for connection in self.connections:
                connection.close()

    # ==========================================================
    # NUMBER OF ACTIONS
    # ==========================================================

    def get_num_actions(self):

        return self.num_actions