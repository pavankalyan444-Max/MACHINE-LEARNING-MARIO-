import time

import numpy as np
import torch

from environment import MarioEnvironment
from agent import DQNAgent
from brain_visualizer import BrainVisualizer


ACTION_NAMES = [
    "NOOP",
    "RIGHT",
    "RIGHT + A",
    "RIGHT + B",
    "RIGHT + A + B",
    "A",
    "LEFT",
]


def get_q_values(agent, state):

    state_tensor = torch.as_tensor(
        state,
        dtype=torch.float32,
        device=agent.device
    )

    state_tensor = state_tensor / 255.0

    state_tensor = state_tensor.unsqueeze(0)

    with torch.no_grad():

        q_values = agent.policy_net(
            state_tensor
        )

    return q_values[0].cpu().numpy()


def main():

    print("Starting Mario AI Brain...")

    # ---------------------------------------------------------
    # Environment
    # ---------------------------------------------------------

    env = MarioEnvironment()

    print(
        "Actions:",
        env.get_num_actions()
    )

    # ---------------------------------------------------------
    # DQN Agent
    # ---------------------------------------------------------

    agent = DQNAgent(
        num_actions=env.get_num_actions()
    )

    print(
        "Device:",
        agent.device
    )

    # ---------------------------------------------------------
    # Visualizer
    # ---------------------------------------------------------

    visualizer = BrainVisualizer()

    # ---------------------------------------------------------
    # Start Mario
    # ---------------------------------------------------------

    state, info = env.reset()

    step = 0

    print("\nAI is controlling Mario...")
    print("Close the brain window to stop.\n")

    while visualizer.running:

        # -----------------------------------------------------
        # Get REAL Q-values from YOUR neural network
        # -----------------------------------------------------

        q_values = get_q_values(
            agent,
            state
        )

        # -----------------------------------------------------
        # Choose action
        # -----------------------------------------------------

        action = agent.act(
            state
        )

        # -----------------------------------------------------
        # Execute action in Mario
        # -----------------------------------------------------

        next_state, reward, terminated, truncated, info = env.step(
            action
        )

        done = terminated or truncated

        # -----------------------------------------------------
        # Update neural network visualization
        # -----------------------------------------------------

        visualizer.update(

            frames=[env.get_rgb_frame()],

            q_values=np.expand_dims(
                q_values,
                axis=0
            ),

            actions=np.array([action]),

            epsilon=agent.epsilon,

            rewards=np.array([reward]),

            step=step,

            loss=None
        )

        # -----------------------------------------------------
        # Print information
        # -----------------------------------------------------

        if step % 30 == 0:

            best_action = int(
                np.argmax(q_values)
            )

            print(
                f"Step: {step:4d} | "
                f"Action: {action} "
                f"({ACTION_NAMES[action]}) | "
                f"Best Q: {ACTION_NAMES[best_action]} | "
                f"Reward: {reward:+.2f} | "
                f"Epsilon: {agent.epsilon:.4f}"
            )

        # -----------------------------------------------------
        # Move to next state
        # -----------------------------------------------------

        state = next_state

        step += 1

        # -----------------------------------------------------
        # Episode ended
        # -----------------------------------------------------

        if done:

            print(
                "\nEpisode ended → resetting Mario...\n"
            )

            state, info = env.reset()

        # Small delay so visualization is visible
        time.sleep(0.01)

    env.close()

    visualizer.close()

    print("Mario AI stopped.")


if __name__ == "__main__":

    main()