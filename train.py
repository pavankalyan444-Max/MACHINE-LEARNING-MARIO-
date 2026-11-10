import os
import time

import torch

from environment import MarioEnvironment
from agent import DQNAgent
from checkpoint_utils import MODEL_DIR


# =============================================================
# CONFIGURATION
# =============================================================

NUM_EPISODES = 3

# Update target network every 10,000 environment steps
TARGET_UPDATE_FREQUENCY = 10000

# Train the neural network every 4 environment steps
#
# This is a major performance improvement.
#
# Instead of:
#
# step 1 -> train
# step 2 -> train
# step 3 -> train
# step 4 -> train
#
# We do:
#
# step 1
# step 2
# step 3
# step 4 -> train
#
LEARN_EVERY = 4

# Print progress every N steps
PRINT_EVERY = 100

# Save model every N episodes
SAVE_FREQUENCY = 100

MODEL_DIRECTORY = MODEL_DIR


# =============================================================
# RENDERING
# =============================================================

# True  = show Mario while training
# False = maximum training speed
#
# For our first test keep this TRUE so we can see what is
# happening.

RENDER_TRAINING = True

# Render every few environment steps.
#
# Rendering every single step can slow the emulator down.

RENDER_EVERY = 4


# =============================================================
# CREATE CHECKPOINT DIRECTORY
# =============================================================

os.makedirs(
    MODEL_DIRECTORY,
    exist_ok=True
)


# =============================================================
# CREATE ENVIRONMENT
# =============================================================

env = MarioEnvironment()


print("=" * 70)
print("                 MARIO DQN TRAINING")
print("=" * 70)

print(
    "Number of actions:",
    env.num_actions
)


# =============================================================
# CREATE AGENT
# =============================================================

agent = DQNAgent(
    num_actions=env.num_actions
)


print(
    "Device:",
    agent.device
)


# =============================================================
# TRAINING
# =============================================================

training_start_time = time.perf_counter()


for episode in range(
    1,
    NUM_EPISODES + 1
):

    # ---------------------------------------------------------
    # Reset environment
    # ---------------------------------------------------------

    state, info = env.reset()

    episode_reward = 0.0

    episode_loss = 0.0

    loss_count = 0

    terminated = False

    truncated = False

    episode_start_time = time.perf_counter()


    # ---------------------------------------------------------
    # Episode loop
    # ---------------------------------------------------------

    while not (
        terminated
        or truncated
    ):

        # =====================================================
        # CHOOSE ACTION
        # =====================================================

        action = agent.act(
            state
        )


        # =====================================================
        # PERFORM ACTION
        # =====================================================

        (
            next_state,
            reward,
            terminated,
            truncated,
            info
        ) = env.step(
            action
        )


        # =====================================================
        # RENDER
        # =====================================================

        if (
            RENDER_TRAINING
            and agent.steps_done % RENDER_EVERY == 0
        ):

            env.render()


        # =====================================================
        # STORE EXPERIENCE
        # =====================================================

        done = (
            terminated
            or truncated
        )

        agent.memory.push(
            state,
            action,
            reward,
            next_state,
            done
        )


        # =====================================================
        # MOVE TO NEXT STATE
        # =====================================================

        state = next_state


        # =====================================================
        # REWARD
        # =====================================================

        episode_reward += reward


        # =====================================================
        # LEARN
        # =====================================================

        loss = None

        if (
            agent.steps_done % LEARN_EVERY == 0
        ):

            loss = agent.learn()


        # =====================================================
        # STORE LOSS
        # =====================================================

        if loss is not None:

            episode_loss += loss

            loss_count += 1


        # =====================================================
        # TARGET NETWORK UPDATE
        # =====================================================

        if (
            agent.steps_done
            % TARGET_UPDATE_FREQUENCY
            == 0
        ):

            agent.update_target_network()

            print(
                "\nTarget network updated."
            )


        # =====================================================
        # LIVE PROGRESS
        # =====================================================

        if (
            agent.steps_done
            % PRINT_EVERY
            == 0
        ):

            elapsed = (
                time.perf_counter()
                - episode_start_time
            )

            steps_per_second = (
                agent.steps_done / elapsed
                if elapsed > 0
                else 0
            )

            current_loss = (
                loss
                if loss is not None
                else 0.0
            )

            print(
                f"\r"
                f"Episode: {episode:3d} | "
                f"Steps: {agent.steps_done:6d} | "
                f"Reward: {episode_reward:8.1f} | "
                f"Loss: {current_loss:.5f} | "
                f"Epsilon: {agent.epsilon:.4f} | "
                f"Speed: {steps_per_second:.1f} steps/s",
                end="",
                flush=True
            )


    # =========================================================
    # EPISODE FINISHED
    # =========================================================

    episode_time = (
        time.perf_counter()
        - episode_start_time
    )

    average_loss = (
        episode_loss / loss_count
        if loss_count > 0
        else 0.0
    )


    print()

    print(
        "-" * 70
    )

    print(
        f"Episode {episode} finished"
    )

    print(
        f"Reward       : {episode_reward:.2f}"
    )

    print(
        f"Average Loss : {average_loss:.5f}"
    )

    print(
        f"Epsilon      : {agent.epsilon:.4f}"
    )

    print(
        f"Total Steps  : {agent.steps_done}"
    )

    print(
        f"Episode Time : {episode_time:.2f} seconds"
    )

    print(
        "-" * 70
    )


    # =========================================================
    # SAVE CHECKPOINT
    # =========================================================

    if (
        episode % SAVE_FREQUENCY
        == 0
    ):

        checkpoint_path = os.path.join(
            MODEL_DIRECTORY,
            f"mario_dqn_episode_{episode}.pth"
        )

        torch.save(
            {
                "episode": episode,

                "policy_net":
                    agent.policy_net.state_dict(),

                "target_net":
                    agent.target_net.state_dict(),

                "optimizer":
                    agent.optimizer.state_dict(),

                "step":
                    agent.steps_done,

                "steps_done":
                    agent.steps_done,

                "epsilon":
                    agent.epsilon,
            },
            checkpoint_path
        )

        print(
            f"Checkpoint saved: "
            f"{checkpoint_path}"
        )


# =============================================================
# SAVE FINAL MODEL
# =============================================================

final_path = os.path.join(
    MODEL_DIRECTORY,
    "mario_dqn_final.pth"
)


torch.save(
    {
        "policy_net":
            agent.policy_net.state_dict(),

        "target_net":
            agent.target_net.state_dict(),

        "optimizer":
            agent.optimizer.state_dict(),

        "step":
            agent.steps_done,

        "episode":
            NUM_EPISODES,

        "steps_done":
            agent.steps_done,

        "epsilon":
            agent.epsilon,
    },
    final_path
)


# =============================================================
# TOTAL TRAINING TIME
# =============================================================

total_time = (
    time.perf_counter()
    - training_start_time
)


# =============================================================
# CLOSE ENVIRONMENT
# =============================================================

env.close()


print()
print("=" * 70)
print("                 TRAINING FINISHED")
print("=" * 70)

print(
    f"Total training time: "
    f"{total_time:.2f} seconds"
)

print(
    f"Final model: "
    f"{final_path}"
)