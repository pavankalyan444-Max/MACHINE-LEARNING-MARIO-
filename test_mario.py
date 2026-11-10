import time

from environment import MarioEnvironment


def main():
    print("Creating Mario environment...")

    env = MarioEnvironment()

    print("Number of actions:", env.get_num_actions())

    state, info = env.reset()

    print("Initial state shape:", state.shape)
    print("Initial state dtype:", state.dtype)

    print("\nStarting Mario...")
    print("Close the game window to stop.")

    for step in range(1000):
        # For now, always move RIGHT
        action = 1

        state, reward, terminated, truncated, info = env.step(action)

        env.render()

        if step % 30 == 0:
            print(
                f"Step: {step:4d} | "
                f"Reward: {reward:6.2f} | "
                f"State: {state.shape}"
            )

        if terminated or truncated:
            print("Episode ended. Resetting...")
            state, info = env.reset()

        time.sleep(0.01)

    env.close()


if __name__ == "__main__":
    main()