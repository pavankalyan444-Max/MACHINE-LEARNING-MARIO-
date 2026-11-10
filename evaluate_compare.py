import time
import numpy as np
import torch

from environment import MarioEnvironment
from agent import DQNAgent
from checkpoint_utils import checkpoint_value, model_path


# =========================================================
# SETTINGS
# =========================================================

CHECKPOINTS = [
    model_path("mario_checkpoint_10000.pth"),
    model_path("mario_checkpoint_100000.pth"),
]

EPISODES = 5

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

def load_model(agent, checkpoint_path):

    print()
    print("=" * 70)
    print(f"LOADING: {checkpoint_path}")
    print("=" * 70)

    checkpoint = torch.load(
        checkpoint_path,
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

    print(
        f"Saved training step: "
        f"{checkpoint.get('step', 'unknown')}"
    )

    print("Epsilon: 0.0")
    print("Evaluation: GREEDY")


# =========================================================
# GET ACTION
# =========================================================

def get_action(agent, state):

    state_tensor = torch.as_tensor(
        state,
        dtype=torch.float32,
        device=agent.device
    )

    state_tensor /= 255.0

    state_tensor = state_tensor.unsqueeze(0)

    with torch.no_grad():

        q_values = agent.policy_net(
            state_tensor
        )

    q_values = q_values[0].cpu().numpy()

    action = int(np.argmax(q_values))

    return action, q_values


# =========================================================
# EVALUATE CHECKPOINT
# =========================================================

def evaluate_checkpoint(checkpoint_path):

    env = MarioEnvironment()

    agent = DQNAgent(
        num_actions=env.get_num_actions()
    )

    load_model(
        agent,
        checkpoint_path
    )

    results = []

    print()
    print("Starting evaluation...")
    print()

    for episode in range(1, EPISODES + 1):

        state, _ = env.reset()

        episode_reward = 0.0
        episode_steps = 0

        max_x = 0
        max_score = 0
        max_coins = 0

        final_info = {}

        start_time = time.time()

        while True:

            # ---------------------------------------------
            # Choose greedy action
            # ---------------------------------------------

            action, q_values = get_action(
                agent,
                state
            )

            # ---------------------------------------------
            # Step environment
            # ---------------------------------------------

            (
                next_state,
                reward,
                terminated,
                truncated,
                info
            ) = env.step(action)

            done = terminated or truncated

            episode_reward += float(reward)
            episode_steps += 1

            # ---------------------------------------------
            # Track actual Mario progress
            # ---------------------------------------------

            if info:

                x_pos = info.get(
                    "x_pos",
                    0
                )

                score = info.get(
                    "score",
                    0
                )

                coins = info.get(
                    "coins",
                    0
                )

                max_x = max(
                    max_x,
                    x_pos
                )

                max_score = max(
                    max_score,
                    score
                )

                max_coins = max(
                    max_coins,
                    coins
                )

                final_info = info.copy()

            state = next_state

            # ---------------------------------------------
            # Episode finished
            # ---------------------------------------------

            if done:
                break

        elapsed = time.time() - start_time

        flag_get = final_info.get(
            "flag_get",
            False
        )

        world = final_info.get(
            "world",
            0
        )

        stage = final_info.get(
            "stage",
            0
        )

        life = final_info.get(
            "life",
            0
        )

        status = final_info.get(
            "status",
            "unknown"
        )

        results.append({
            "reward": episode_reward,
            "steps": episode_steps,
            "max_x": max_x,
            "score": max_score,
            "coins": max_coins,
            "flag": flag_get,
            "world": world,
            "stage": stage,
        })

        print()
        print("-" * 60)
        print(
            f"Episode {episode}"
        )
        print("-" * 60)

        print(
            f"Reward:       {episode_reward:.2f}"
        )

        print(
            f"Steps:        {episode_steps}"
        )

        print(
            f"MAX X:        {max_x}"
        )

        print(
            f"World/Stage:  {world}-{stage}"
        )

        print(
            f"Score:        {max_score}"
        )

        print(
            f"Coins:        {max_coins}"
        )

        print(
            f"Flag reached: {flag_get}"
        )

        print(
            f"Life:         {life}"
        )

        print(
            f"Status:       {status}"
        )

        print(
            f"Time:         {elapsed:.1f}s"
        )

    env.close()

    # =====================================================
    # SUMMARY
    # =====================================================

    avg_reward = np.mean([
        r["reward"]
        for r in results
    ])

    avg_x = np.mean([
        r["max_x"]
        for r in results
    ])

    best_x = max([
        r["max_x"]
        for r in results
    ])

    best_reward = max([
        r["reward"]
        for r in results
    ])

    flags = sum([
        r["flag"]
        for r in results
    ])

    print()
    print("=" * 70)
    print(
        f"RESULTS: {checkpoint_path}"
    )
    print("=" * 70)

    print(
        f"Average reward:    {avg_reward:.2f}"
    )

    print(
        f"Best reward:       {best_reward:.2f}"
    )

    print(
        f"Average MAX X:     {avg_x:.1f}"
    )

    print(
        f"Best MAX X:        {best_x}"
    )

    print(
        f"Flags reached:     {flags}/{EPISODES}"
    )

    print()

    return results


# =========================================================
# MAIN
# =========================================================

def main():

    print()
    print("=" * 70)
    print("       MARIO AI — CHECKPOINT COMPARISON")
    print("=" * 70)

    all_results = {}

    for checkpoint in CHECKPOINTS:

        results = evaluate_checkpoint(
            checkpoint
        )

        all_results[checkpoint] = results

    # =====================================================
    # FINAL COMPARISON
    # =====================================================

    print()
    print()
    print("=" * 70)
    print("              FINAL COMPARISON")
    print("=" * 70)

    for checkpoint, results in all_results.items():

        avg_reward = np.mean([
            r["reward"]
            for r in results
        ])

        avg_x = np.mean([
            r["max_x"]
            for r in results
        ])

        best_x = max([
            r["max_x"]
            for r in results
        ])

        flags = sum([
            r["flag"]
            for r in results
        ])

        print()
        print(checkpoint)

        print(
            f"  Average reward: {avg_reward:.2f}"
        )

        print(
            f"  Average X:      {avg_x:.1f}"
        )

        print(
            f"  Best X:         {best_x}"
        )

        print(
            f"  Flags:          {flags}/{EPISODES}"
        )

    print()
    print("=" * 70)


# =========================================================
# START
# =========================================================

if __name__ == "__main__":
    main()