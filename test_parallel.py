import time

from parallel_env import ParallelMarioEnv


def main():
    env = ParallelMarioEnv(num_envs=10)
    try:
        run_test(env)
    finally:
        env.close()


def run_test(env):
    print("Starting parallel Mario test...")

    states, rgb_frames = env.reset()

    print(
        "Initial states:",
        states.shape,
        states.dtype
    )

    print("\nRunning 100 parallel steps...\n")

    for step in range(100):

        # For testing, make every Mario move RIGHT
        actions = [1] * 10

        next_states, rewards, dones, infos, rgb_frames = env.step(
            actions
        )

        if step % 10 == 0:

            print(
                f"Step: {step:3d} | "
                f"Rewards: {rewards} | "
                f"Dones: {dones}"
            )

        env.render_grid(
            frames=rgb_frames,
            actions=actions,
            rewards=rewards,
            step=step
        )

        time.sleep(0.01)

    print("\nParallel test finished.")


if __name__ == "__main__":
    main()