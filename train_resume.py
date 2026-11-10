import os
import time
from contextlib import ExitStack
import numpy as np
import torch

from parallel_env import ParallelMarioEnv
from agent import DQNAgent
from checkpoint_utils import checkpoint_value, model_path
from training_tools import TrainingMonitor, choose_checkpoint, parse_training_options


# ============================================================
# CONFIGURATION
# ============================================================

NUM_ENVS = 10

# Resume from checkpoint
CHECKPOINT = model_path("mario_checkpoint_100000.pth")

# Additional training steps
TRAINING_STEPS = 1_000_000

# DQN learning frequency
LEARN_EVERY = 4

# Target network update frequency
TARGET_UPDATE_EVERY = 10_000

# Save every 50k steps
SAVE_EVERY = 50_000

# Display Mario grid every N steps
DISPLAY_EVERY = 4


# ============================================================
# CHECKPOINT SAVE
# ============================================================

def save_checkpoint(
    agent,
    global_step,
    episode,
    filename,
    level
):

    filename = model_path(filename, create=True)

    checkpoint = {
        "policy_net": agent.policy_net.state_dict(),
        "target_net": agent.target_net.state_dict(),
        "optimizer": agent.optimizer.state_dict(),

        "steps_done": agent.steps_done,
        "epsilon": agent.epsilon,

        "step": global_step,
        "episode": episode,
        "level": level,
    }

    torch.save(
        checkpoint,
        filename
    )

    print()
    print("=" * 70)
    print("CHECKPOINT SAVED")
    print("=" * 70)

    print(
        "File       :",
        filename
    )

    print(
        "Step       :",
        global_step
    )

    print(
        "Experiences:",
        global_step * NUM_ENVS
    )

    print(
        "Episode    :",
        episode
    )

    print(
        "Epsilon    :",
        f"{agent.epsilon:.6f}"
    )

    print("=" * 70)
    print()


# ============================================================
# CHECKPOINT LOAD
# ============================================================

def load_checkpoint(
    agent,
    filename,
    level=None
):

    if not os.path.exists(filename):
        raise FileNotFoundError(f"Checkpoint not found: {filename}")

    print()
    print("=" * 70)
    print("LOADING CHECKPOINT")
    print("=" * 70)

    print(
        filename
    )

    checkpoint = torch.load(
        filename,
        map_location=agent.device
    )

    saved_level = checkpoint.get("level")
    if saved_level and level and saved_level != level:
        print(f"Warning: checkpoint level is {saved_level}; training on {level}.")

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

    agent.optimizer.load_state_dict(
        checkpoint_value(
            checkpoint,
            "optimizer",
            "optimizer_state_dict"
        )
    )

    agent.steps_done = checkpoint.get(
        "steps_done",
        0
    )

    agent.epsilon = checkpoint.get(
        "epsilon",
        agent.epsilon
    )

    global_step = checkpoint.get(
        "step",
        0
    )

    episode = checkpoint.get(
        "episode",
        0
    )

    print(
        "Loaded step   :",
        global_step
    )

    print(
        "Loaded episode:",
        episode
    )

    print(
        "Epsilon       :",
        f"{agent.epsilon:.6f}"
    )

    print("=" * 70)
    print()

    return global_step, episode


# ============================================================
# MAIN
# ============================================================

def main():
    global NUM_ENVS, TRAINING_STEPS
    options = parse_training_options(
        "Resume Mario DQN training", NUM_ENVS, TRAINING_STEPS, resume=True
    )
    NUM_ENVS = options.num_envs
    TRAINING_STEPS = options.steps
    checkpoint = model_path(options.checkpoint) if options.checkpoint else choose_checkpoint(CHECKPOINT)
    with ExitStack() as stack:
        env = ParallelMarioEnv(num_envs=NUM_ENVS, level=options.level,
                               headless=options.headless)
        stack.callback(env.close)
        monitor = TrainingMonitor(options)
        stack.callback(monitor.close)
        train(env, options, monitor, checkpoint)


def train(env, options, monitor, checkpoint):

    print()
    print("=" * 70)
    print("MARIO DOUBLE DQN TRAINING")
    print("=" * 70)
    print()

    # ========================================================
    # CREATE ENVIRONMENTS
    # ========================================================

    num_actions = env.get_num_actions()

    print()
    print(
        "Environments :",
        NUM_ENVS
    )

    print(
        "Actions      :",
        num_actions
    )

    print()

    # ========================================================
    # CREATE AGENT
    # ========================================================

    agent = DQNAgent(
        num_actions=num_actions,

        learning_rate=0.00025,

        gamma=0.99,

        epsilon_start=0.10,
        epsilon_end=0.05,
        epsilon_decay=500_000,

        buffer_size=10_000,

        batch_size=32
    )

    # ========================================================
    # LOAD CHECKPOINT
    # ========================================================

    global_step, episode = load_checkpoint(
        agent,
        checkpoint,
        options.level
    )

    start_step = global_step
    monitor.start_step = start_step
    monitor.last_eval_step = start_step

    target_step = (
        start_step
        + TRAINING_STEPS
    )

    print()
    print("=" * 70)
    print("TRAINING SETTINGS")
    print("=" * 70)

    print(
        "Starting step       :",
        start_step
    )

    print(
        "Target step         :",
        target_step
    )

    print(
        "Number of envs      :",
        NUM_ENVS
    )

    print(
        "Learning every      :",
        LEARN_EVERY
    )

    print(
        "Target update every :",
        TARGET_UPDATE_EVERY
    )

    print(
        "Save every           :",
        SAVE_EVERY
    )

    print(
        "Display every        :",
        DISPLAY_EVERY
    )

    print(
        "Device               :",
        agent.device
    )

    print(
        "Epsilon              :",
        agent.epsilon
    )

    print("=" * 70)
    print()

    # ========================================================
    # INITIAL RESET
    # ========================================================

    states, rgb_frames = env.reset()

    # ========================================================
    # EPISODE STATISTICS
    # ========================================================

    episode_rewards = np.zeros(
        NUM_ENVS,
        dtype=np.float32
    )

    episode_lengths = np.zeros(
        NUM_ENVS,
        dtype=np.int32
    )

    recent_rewards = []
    recent_x_positions = []
    recent_lengths = []

    # ========================================================
    # TIMING
    # ========================================================

    start_time = time.time()

    last_print_time = start_time

    last_save_step = global_step

    loss_value = None

    # ========================================================
    # TRAINING LOOP
    # ========================================================

    try:

        while global_step < target_step:

            # =================================================
            # ACTION SELECTION
            # =================================================

            actions, q_values = agent.act_batch(
                states
            )

            # =================================================
            # STEP MARIO ENVIRONMENTS
            # =================================================

            (
                next_states,
                rewards,
                dones,
                infos,
                rgb_frames
            ) = env.step(
                actions
            )

            # =================================================
            # STORE EXPERIENCES
            # =================================================

            for i in range(NUM_ENVS):

                agent.memory.push(
                    states[i],
                    int(actions[i]),
                    float(rewards[i]),
                    infos[i].get(
                        "terminal_observation",
                        next_states[i]
                    ),
                    bool(dones[i])
                )

            # =================================================
            # EPISODE STATISTICS
            # =================================================

            for i in range(NUM_ENVS):

                episode_rewards[i] += float(
                    rewards[i]
                )

                episode_lengths[i] += 1

                info = infos[i]

                # ------------------------------------------------
                # Mario position
                # ------------------------------------------------

                x_pos = int(
                    info.get(
                        "x_pos",
                        0
                    )
                )

                # ------------------------------------------------
                # Flag
                # ------------------------------------------------

                flag_get = bool(
                    info.get(
                        "flag_get",
                        False
                    )
                )

                # ------------------------------------------------
                # Life
                # ------------------------------------------------

                life = info.get(
                    "life",
                    None
                )

                # ------------------------------------------------
                # World
                # ------------------------------------------------

                world = info.get(
                    "world",
                    None
                )

                # ------------------------------------------------
                # Stage
                # ------------------------------------------------

                stage = info.get(
                    "stage",
                    None
                )

                # =================================================
                # EPISODE FINISHED
                # =================================================

                if dones[i]:

                    episode += 1
                    monitor.record_episode(
                        float(episode_rewards[i]), x_pos,
                        int(episode_lengths[i]), flag_get
                    )

                    final_reward = float(
                        episode_rewards[i]
                    )

                    final_length = int(
                        episode_lengths[i]
                    )

                    final_x = x_pos

                    # ------------------------------------------------
                    # Store statistics
                    # ------------------------------------------------

                    recent_rewards.append(
                        final_reward
                    )

                    recent_x_positions.append(
                        final_x
                    )

                    recent_lengths.append(
                        final_length
                    )

                    # Keep last 100 episodes

                    if len(recent_rewards) > 100:
                        recent_rewards.pop(0)

                    if len(recent_x_positions) > 100:
                        recent_x_positions.pop(0)

                    if len(recent_lengths) > 100:
                        recent_lengths.pop(0)

                    # ------------------------------------------------
                    # Print episode
                    # ------------------------------------------------

                    print()

                    print(
                        f"Episode {episode:5d} | "
                        f"Env {i} | "
                        f"Reward {final_reward:8.1f} | "
                        f"X {final_x:4d} | "
                        f"Length {final_length:6d} | "
                        f"Flag {flag_get} | "
                        f"Life {life} | "
                        f"World {world}-{stage}"
                    )

                    # ------------------------------------------------
                    # Reset episode counters
                    # ------------------------------------------------

                    episode_rewards[i] = 0.0

                    episode_lengths[i] = 0

            # =================================================
            # MOVE TO NEXT STATE
            # =================================================

            states = next_states

            # =================================================
            # GLOBAL STEP
            # =================================================

            global_step += 1

            # =================================================
            # LEARNING
            # =================================================

            if (
                global_step
                % LEARN_EVERY
                == 0
            ):

                loss = agent.learn()

                if loss is not None:

                    loss_value = loss

            monitor.update(agent, global_step, episode, loss_value)

            # =================================================
            # TARGET NETWORK
            # =================================================

            if (
                global_step
                % TARGET_UPDATE_EVERY
                == 0
            ):

                agent.update_target_network()

                print()

                print(
                    "Target network updated "
                    f"at step {global_step}"
                )

                print()

            # =================================================
            # SAVE CHECKPOINT
            # =================================================

            if (
                global_step
                - last_save_step
                >= SAVE_EVERY
            ):

                filename = (
                    f"mario_checkpoint_"
                    f"{global_step}.pth"
                )

                save_checkpoint(
                    agent,
                    global_step,
                    episode,
                    filename,
                    options.level
                )

                last_save_step = global_step

            # =================================================
            # DISPLAY 10 MARIO
            # =================================================

            if (
                not options.headless
                and global_step
                % DISPLAY_EVERY
                == 0
            ):

                env.render_grid(
                    frames=rgb_frames,
                    actions=actions,
                    rewards=rewards,
                    step=global_step
                )

            # =================================================
            # PROGRESS REPORT
            # =================================================

            current_time = time.time()

            if (
                current_time
                - last_print_time
                >= 10
            ):

                elapsed = (
                    current_time
                    - start_time
                )

                steps_since_start = (
                    global_step
                    - start_step
                )

                if elapsed > 0:

                    experiences_per_second = (
                        steps_since_start
                        * NUM_ENVS
                        / elapsed
                    )

                else:

                    experiences_per_second = 0.0

                # ------------------------------------------------
                # Average statistics
                # ------------------------------------------------

                if recent_rewards:

                    avg_reward = (
                        sum(recent_rewards)
                        / len(recent_rewards)
                    )

                    avg_x = (
                        sum(recent_x_positions)
                        / len(recent_x_positions)
                    )

                    avg_length = (
                        sum(recent_lengths)
                        / len(recent_lengths)
                    )

                else:

                    avg_reward = 0.0
                    avg_x = 0.0
                    avg_length = 0.0

                # ------------------------------------------------
                # Progress
                # ------------------------------------------------

                print(
                    f"Step {global_step:8d} | "
                    f"Experience "
                    f"{global_step * NUM_ENVS:9d} | "
                    f"FPS {experiences_per_second:7.1f} | "
                    f"AvgReward "
                    f"{avg_reward:8.1f} | "
                    f"AvgX "
                    f"{avg_x:6.1f} | "
                    f"AvgLen "
                    f"{avg_length:7.1f} | "
                    f"Loss "
                    f"{loss_value if loss_value is not None else 0:.4f} | "
                    f"Epsilon "
                    f"{agent.epsilon:.4f} | "
                    f"Replay "
                    f"{len(agent.memory):6d}"
                )

                last_print_time = current_time

    # ========================================================
    # CTRL + C
    # ========================================================

    except KeyboardInterrupt:

        print()
        print()
        print("=" * 70)
        print("TRAINING INTERRUPTED")
        print("=" * 70)

        interrupt_filename = (
            f"mario_checkpoint_interrupt_"
            f"{global_step}.pth"
        )

        save_checkpoint(
            agent,
            global_step,
            episode,
            interrupt_filename,
            options.level
        )

        print(
            "Progress saved successfully."
        )

    # ========================================================
    # CLEANUP
    # ========================================================

    finally:

        if global_step > start_step:

            final_filename = (
                f"mario_checkpoint_"
                f"{global_step}_final.pth"
            )

            save_checkpoint(
                agent,
                global_step,
                episode,
                final_filename,
                options.level
            )

        print()
        print("=" * 70)
        print("TRAINING STOPPED")
        print("=" * 70)

        print(
            "Final step :",
            global_step
        )

        print(
            "Episodes   :",
            episode
        )

        print(
            "Epsilon    :",
            f"{agent.epsilon:.6f}"
        )

        print(
            "Replay     :",
            len(agent.memory)
        )

        print("=" * 70)
        print()


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":

    main()