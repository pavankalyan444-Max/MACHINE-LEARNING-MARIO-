import time
from contextlib import ExitStack
import numpy as np
import torch

from parallel_env import ParallelMarioEnv
from agent import DQNAgent
from checkpoint_utils import model_path
from brain_visualizer import BrainVisualizer
from training_tools import TrainingMonitor, parse_training_options


# =========================================================
# SETTINGS
# =========================================================

NUM_ENVS = 10

TOTAL_STEPS = 10000

LEARN_EVERY = 4

TARGET_UPDATE_EVERY = 2000

SAVE_EVERY = 5000

VISUALIZE_EVERY = 4


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
# CHECKPOINT
# =========================================================

def save_checkpoint(
    agent,
    step,
    episode,
    level
):

    checkpoint = {

        "policy_net":
            agent.policy_net.state_dict(),

        "target_net":
            agent.target_net.state_dict(),

        "optimizer":
            agent.optimizer.state_dict(),

        "steps_done":
            agent.steps_done,

        "epsilon":
            agent.epsilon,

        "step":
            step,

        "episode":
            episode,

        "level":
            level,
    }

    filename = model_path(f"mario_checkpoint_{step}.pth", create=True)

    torch.save(
        checkpoint,
        filename
    )

    print(
        f"\nCheckpoint saved: {filename}\n"
    )


# =========================================================
# MAIN
# =========================================================

def main():
    global NUM_ENVS, TOTAL_STEPS
    options = parse_training_options("Train a new Mario DQN", NUM_ENVS, TOTAL_STEPS)
    NUM_ENVS = options.num_envs
    TOTAL_STEPS = options.steps
    with ExitStack() as stack:
        env = ParallelMarioEnv(num_envs=NUM_ENVS, level=options.level,
                               headless=options.headless)
        stack.callback(env.close)
        monitor = TrainingMonitor(options)
        stack.callback(monitor.close)
        train(env, options, monitor, stack)


def train(env, options, monitor, stack):

    print("=" * 70)

    print(
        f"       MARIO AI - {NUM_ENVS} ENVIRONMENT DQN TRAINING"
    )

    print("=" * 70)

    print()

    # -----------------------------------------------------
    # Environments
    # -----------------------------------------------------

    # -----------------------------------------------------
    # Agent
    # -----------------------------------------------------

    agent = DQNAgent(
        num_actions=env.get_num_actions()
    )

    # -----------------------------------------------------
    # GPU information
    # -----------------------------------------------------

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

        print(
            "CUDA:",
            torch.version.cuda
        )

    # -----------------------------------------------------
    # Visualizer
    # -----------------------------------------------------

    visualizer = None
    if not options.headless:
        visualizer = BrainVisualizer(
            width=1280,
            height=760,
            num_envs=NUM_ENVS
        )
        stack.callback(visualizer.close)

    # -----------------------------------------------------
    # Initial state
    # -----------------------------------------------------

    states, rgb_frames = env.reset()

    episode_rewards = np.zeros(
        NUM_ENVS,
        dtype=np.float32
    )

    episode_lengths = np.zeros(
        NUM_ENVS,
        dtype=np.int32
    )

    episode_count = 0

    global_step = 0

    last_loss = None

    start_time = time.time()

    print()

    print(
        "TRAINING STARTED"
    )

    print(
        "Press Ctrl+C to stop." if options.headless else "Press ESC to stop."
    )

    print()

    # =====================================================
    # TRAINING LOOP
    # =====================================================

    while (

        global_step < TOTAL_STEPS

        and (visualizer is None or visualizer.running)

    ):

        # -------------------------------------------------
        # 1. ONE GPU CALL FOR ALL 10 MARIOS
        # -------------------------------------------------

        actions, q_values = agent.act_batch(
            states
        )

        # -------------------------------------------------
        # 2. STEP ALL 10 ENVIRONMENTS
        # -------------------------------------------------

        (
            next_states,
            rewards,
            dones,
            infos,
            rgb_frames
        ) = env.step(
            actions
        )

        # -------------------------------------------------
        # 3. STORE 10 EXPERIENCES
        # -------------------------------------------------

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

            episode_rewards[i] += (
                rewards[i]
            )

            episode_lengths[i] += 1

        # -------------------------------------------------
        # 4. LEARN
        # -------------------------------------------------

        if (

            global_step % LEARN_EVERY == 0

            and len(agent.memory)
            >= agent.batch_size

        ):

            last_loss = agent.learn()

        # -------------------------------------------------
        # 5. TARGET NETWORK
        # -------------------------------------------------

        if (

            global_step > 0

            and global_step
            % TARGET_UPDATE_EVERY == 0

        ):

            agent.update_target_network()

            print(
                f"\nTarget network updated "
                f"at step {global_step}\n"
            )

        # -------------------------------------------------
        # 6. EPISODES
        # -------------------------------------------------

        for i in range(NUM_ENVS):

            if dones[i]:

                episode_count += 1
                monitor.record_episode(
                    float(episode_rewards[i]), int(infos[i].get("x_pos", 0)),
                    int(episode_lengths[i]), bool(infos[i].get("flag_get", False))
                )

                print(

                    f"Episode "
                    f"{episode_count:5d} | "

                    f"Env {i + 1:2d} | "

                    f"Reward "
                    f"{episode_rewards[i]:8.2f} | "

                    f"Length "
                    f"{episode_lengths[i]:5d} | "

                    f"Epsilon "
                    f"{agent.epsilon:.4f}"

                )

                episode_rewards[i] = 0

                episode_lengths[i] = 0

        # -------------------------------------------------
        # 7. VISUALIZATION
        # -------------------------------------------------

        if (

            visualizer is not None
            and global_step
            % VISUALIZE_EVERY == 0

        ):

            visualizer.update(

                frames=rgb_frames,

                q_values=q_values,

                actions=actions,

                epsilon=agent.epsilon,

                rewards=rewards,

                step=global_step,

                loss=last_loss

            )

        # -------------------------------------------------
        # 8. PROGRESS
        # -------------------------------------------------

        if global_step % 100 == 0:

            elapsed = (
                time.time()
                - start_time
            )

            fps = (
                global_step
                * NUM_ENVS
                / max(elapsed, 0.001)
            )

            best_action = int(
                np.argmax(
                    q_values[0]
                )
            )

            avg_reward = float(
                np.mean(rewards)
            )

            loss_text = (

                "--"

                if last_loss is None

                else f"{last_loss:.5f}"

            )

            print(

                f"Step: "
                f"{global_step:6d} | "

                f"Action: "
                f"{ACTION_NAMES[actions[0]]:12s} | "

                f"Best Q: "
                f"{ACTION_NAMES[best_action]:12s} | "

                f"Avg Reward: "
                f"{avg_reward:+6.2f} | "

                f"Loss: "
                f"{loss_text:>8s} | "

                f"Epsilon: "
                f"{agent.epsilon:.4f} | "

                f"Memory: "
                f"{len(agent.memory):6d} | "

                f"FPS: "
                f"{fps:6.1f}"

            )

        # -------------------------------------------------
        # 9. CHECKPOINT
        # -------------------------------------------------

        if (

            global_step > 0

            and global_step
            % SAVE_EVERY == 0

        ):

            save_checkpoint(

                agent,

                global_step,

                episode_count,
                options.level

            )

        # -------------------------------------------------
        # NEXT STATE
        # -------------------------------------------------

        states = next_states

        global_step += 1
        monitor.update(agent, global_step, episode_count, last_loss)

    # =====================================================
    # FINISHED
    # =====================================================

    print()

    print("=" * 70)

    print(
        "              TRAINING FINISHED"
    )

    print("=" * 70)

    print()

    print(
        "Steps:",
        global_step
    )

    print(
        "Episodes:",
        episode_count
    )

    print(
        "Experiences:",
        len(agent.memory)
    )

    print(
        "Final epsilon:",
        round(agent.epsilon, 4)
    )

    if last_loss is not None:

        print(
            "Final loss:",
            round(last_loss, 6)
        )

    save_checkpoint(

        agent,

        global_step,

        episode_count,
        options.level

    )

    print()

    print(
        "Mario AI stopped."
    )


# =========================================================
# START
# =========================================================

if __name__ == "__main__":

    main()