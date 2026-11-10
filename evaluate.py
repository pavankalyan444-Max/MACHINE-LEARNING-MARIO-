import time
import numpy as np
import torch

from parallel_env import ParallelMarioEnv
from agent import DQNAgent
from brain_visualizer import BrainVisualizer
from checkpoint_utils import checkpoint_value, model_path


# =========================================================
# SETTINGS
# =========================================================

NUM_ENVS = 1

CHECKPOINT = model_path("mario_checkpoint_100000.pth")

TOTAL_STEPS = 100000

VISUALIZE_EVERY = 2


ACTION_NAMES = [
    "NOOP",
    "RIGHT",
    "RIGHT + A",
    "RIGHT + B",
    "RIGHT + A + B",
    "A",
    "LEFT",
]


# =========================================================
# LOAD MODEL
# =========================================================

def load_checkpoint(agent):

    print(
        f"\nLoading: {CHECKPOINT}"
    )

    checkpoint = torch.load(
        CHECKPOINT,
        map_location=agent.device
    )

    agent.policy_net.load_state_dict(
        checkpoint_value(
            checkpoint,
            "policy_net",
            "model_state_dict"
        )
    )

    agent.target_net.load_state_dict(
        checkpoint_value(
            checkpoint,
            "target_net",
            "target_state_dict"
        )
    )

    agent.policy_net.eval()

    agent.epsilon = 0.0

    print("✓ Model loaded")
    print(
        "Training steps:",
        checkpoint.get("step", "unknown")
    )

    print(
        "Evaluation mode: GREEDY"
    )

    print(
        "Epsilon:",
        agent.epsilon
    )


# =========================================================
# GET Q VALUES
# =========================================================

def get_q_values(
    agent,
    state
):

    state_tensor = torch.as_tensor(
        state,
        dtype=torch.float32,
        device=agent.device
    )

    state_tensor = (
        state_tensor / 255.0
    )

    state_tensor = state_tensor.unsqueeze(0)

    with torch.no_grad():

        q_values = agent.policy_net(
            state_tensor
        )

    return q_values[0].cpu().numpy()


# =========================================================
# MAIN
# =========================================================

def main():
    env = ParallelMarioEnv(num_envs=NUM_ENVS)
    try:
        evaluate(env)
    finally:
        env.close()


def evaluate(env):

    print("=" * 70)

    print(
        "          MARIO AI — MODEL EVALUATION"
    )

    print("=" * 70)

    print()

    # -----------------------------------------------------
    # Environment
    # -----------------------------------------------------

    # -----------------------------------------------------
    # Agent
    # -----------------------------------------------------

    agent = DQNAgent(
        num_actions=env.get_num_actions()
    )

    print()

    print(
        "Device:",
        agent.device
    )

    if torch.cuda.is_available():

        print(
            "GPU:",
            torch.cuda.get_device_name(0)
        )

    # -----------------------------------------------------
    # Load trained model
    # -----------------------------------------------------

    load_checkpoint(
        agent
    )

    # -----------------------------------------------------
    # Visualizer
    # -----------------------------------------------------

    visualizer = BrainVisualizer(
        width=1280,
        height=760,
        num_envs=1
    )

    # -----------------------------------------------------
    # Reset
    # -----------------------------------------------------

    states, rgb_frames = env.reset()

    # -----------------------------------------------------
    # Statistics
    # -----------------------------------------------------

    episode_reward = 0.0

    episode_steps = 0

    episode_number = 1

    best_episode_reward = -float(
        "inf"
    )

    total_reward = 0.0

    deaths = 0

    start_time = time.time()

    global_step = 0

    print()

    print(
        "=========================================="
    )

    print(
        "       GREEDY EVALUATION STARTED"
    )

    print(
        "=========================================="
    )

    print()

    # =====================================================
    # LOOP
    # =====================================================

    while (

        global_step < TOTAL_STEPS

        and visualizer.running

    ):

        # -------------------------------------------------
        # Current state
        # -------------------------------------------------

        state = states[0]

        # -------------------------------------------------
        # Q-values
        # -------------------------------------------------

        q_values = get_q_values(
            agent,
            state
        )

        # -------------------------------------------------
        # Greedy action
        # -------------------------------------------------

        action = int(
            np.argmax(q_values)
        )

        # -------------------------------------------------
        # Step Mario
        # -------------------------------------------------

        (
            next_states,
            rewards,
            dones,
            infos,
            rgb_frames
        ) = env.step(
            [action]
        )

        reward = float(
            rewards[0]
        )

        done = bool(
            dones[0]
        )

        # -------------------------------------------------
        # Update statistics
        # -------------------------------------------------

        episode_reward += reward

        episode_steps += 1

        total_reward += reward

        # -------------------------------------------------
        # Visualizer
        # -------------------------------------------------

        if (

            global_step
            % VISUALIZE_EVERY == 0

        ):

            visualizer.update(

                frames=rgb_frames,

                q_values=np.expand_dims(
                    q_values,
                    axis=0
                ),

                actions=np.array(
                    [action]
                ),

                epsilon=0.0,

                rewards=np.array(
                    [reward]
                ),

                step=global_step,

                loss=None

            )

        # -------------------------------------------------
        # Progress
        # -------------------------------------------------

        if global_step % 100 == 0:

            elapsed = (
                time.time()
                - start_time
            )

            fps = (
                global_step
                / max(
                    elapsed,
                    0.001
                )
            )

            print(

                f"Step: "
                f"{global_step:6d} | "

                f"Action: "
                f"{ACTION_NAMES[action]:12s} | "

                f"Reward: "
                f"{reward:+6.2f} | "

                f"Episode Reward: "
                f"{episode_reward:8.2f} | "

                f"Episode Steps: "
                f"{episode_steps:5d} | "

                f"FPS: "
                f"{fps:6.1f}"

            )

        # -------------------------------------------------
        # Episode finished
        # -------------------------------------------------

        if done:

            print()

            print(
                "=========================================="
            )

            print(
                f"EPISODE {episode_number} FINISHED"
            )

            print(
                f"Reward: "
                f"{episode_reward:.2f}"
            )

            print(
                f"Length: "
                f"{episode_steps}"
            )

            # Best reward
            if (
                episode_reward
                > best_episode_reward
            ):

                best_episode_reward = (
                    episode_reward
                )

                print(
                    "🏆 NEW BEST REWARD!"
                )

            else:

                print(
                    f"Best so far: "
                    f"{best_episode_reward:.2f}"
                )

            print(
                "=========================================="
            )

            print()

            deaths += 1

            episode_number += 1

            episode_reward = 0.0

            episode_steps = 0

        states = next_states

        global_step += 1

    # =====================================================
    # RESULTS
    # =====================================================

    print()

    print("=" * 70)

    print(
        "             EVALUATION RESULTS"
    )

    print("=" * 70)

    print()

    print(
        f"Total steps:          {global_step}"
    )

    print(
        f"Episodes:             {episode_number - 1}"
    )

    print(
        f"Best episode reward:  "
        f"{best_episode_reward:.2f}"
    )

    print(
        f"Total reward:         "
        f"{total_reward:.2f}"
    )

    print(
        f"Episode endings:      "
        f"{deaths}"
    )

    print()

    print(
        "Model:",
        CHECKPOINT
    )

    print(
        "Exploration:",
        "OFF (epsilon = 0)"
    )

    print()

    print(
        "No training was performed."
    )

    print(
        "The trained checkpoint was only evaluated."
    )

    print()

    visualizer.close()


# =========================================================
# START
# =========================================================

if __name__ == "__main__":

    main()