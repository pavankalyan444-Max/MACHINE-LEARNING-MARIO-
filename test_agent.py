import time
import torch

from environment import MarioEnvironment
from agent import DQNAgent


def main():

    print("Creating Mario environment...")

    env = MarioEnvironment()

    print(
        "Number of actions:",
        env.get_num_actions()
    )

    print("\nCreating DQN agent...")

    agent = DQNAgent(
        num_actions=env.get_num_actions()
    )

    print("Device:", agent.device)

    state, info = env.reset()

    print(
        "State shape:",
        state.shape
    )

    print("\nStarting AI-controlled Mario...\n")

    for step in range(1000):

        # AI chooses the action
        action = agent.act(state)

        # Execute AI action
        next_state, reward, terminated, truncated, info = env.step(
            action
        )

        # Store experience
        agent.memory.push(
            state,
            action,
            reward,
            next_state,
            terminated or truncated
        )

        state = next_state

        env.render()

        if step % 30 == 0:

            q_values = None

            # Get Q-values for visualization/debugging
            with torch.no_grad():

                state_tensor = torch.as_tensor(
                    state,
                    dtype=torch.float32,
                    device=agent.device
                )

                state_tensor = state_tensor / 255.0

                state_tensor = state_tensor.unsqueeze(0)

                q_values = agent.policy_net(
                    state_tensor
                )[0]

            print(
                f"Step: {step:4d} | "
                f"Action: {action} | "
                f"Reward: {reward:6.2f} | "
                f"Q-values: {q_values.cpu().numpy()}"
            )

        if terminated or truncated:

            print(
                "\nEpisode ended. Resetting...\n"
            )

            state, info = env.reset()

        time.sleep(0.01)

    env.close()


if __name__ == "__main__":
    main()