import time
import numpy as np

from parallel_env import ParallelMarioEnv
from brain_visualizer import BrainVisualizer


NUM_ENVS = 10


def main():
    env = ParallelMarioEnv(num_envs=NUM_ENVS)
    try:
        run_test(env)
    finally:
        env.close()


def run_test(env):
    print("=" * 60)
    print("       MARIO AI DASHBOARD TEST")
    print("=" * 60)

    visualizer = BrainVisualizer(
        width=1280,
        height=760,
        num_envs=NUM_ENVS
    )

    states, rgb_frames = env.reset()

    step = 0

    print("\nDashboard started.")
    print("You should see ONE window.")
    print("Press ESC to stop.\n")

    while (
        visualizer.running
        and step < 1000
    ):

        # Make all Marios move RIGHT
        actions = np.array(
            [1] * NUM_ENVS,
            dtype=np.int64
        )

        (
            next_states,
            rewards,
            dones,
            infos,
            rgb_frames
        ) = env.step(
            actions
        )

        # Fake Q-values for visualization
        q_values = np.random.randn(
            NUM_ENVS,
            7
        ).astype(
            np.float32
        )

        visualizer.update(

            frames=rgb_frames,

            q_values=q_values,

            actions=actions,

            epsilon=0.85,

            rewards=rewards,

            step=step,

            loss=0.123456
        )

        if step % 50 == 0:

            print(
                f"Step: {step:4d} | "
                f"Average reward: "
                f"{np.mean(rewards):+.2f}"
            )

        states = next_states

        step += 1

        time.sleep(0.005)

    visualizer.close()

    print(
        "\nDashboard test finished."
    )


if __name__ == "__main__":

    main()