import numpy as np
import torch

from environment import MarioEnvironment
from model import MarioCNN
from checkpoint_utils import checkpoint_value, model_path


# =========================================================
# SETTINGS
# =========================================================

CHECKPOINTS = [
    model_path("mario_checkpoint_10000.pth"),
    model_path("mario_checkpoint_100000.pth"),
]

MAX_STEPS = 500

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

def load_model(checkpoint_path, device):

    model = MarioCNN(
        num_actions=7
    ).to(device)

    checkpoint = torch.load(
        checkpoint_path,
        map_location=device
    )

    model.load_state_dict(
        checkpoint_value(
            checkpoint,
            "policy_net",
            "model_state_dict"
        )
    )

    model.eval()

    print(
        f"Loaded {checkpoint_path}"
    )

    print(
        f"Saved training step: "
        f"{checkpoint.get('step', 'unknown')}"
    )

    return model


# =========================================================
# GET ACTION
# =========================================================

def get_action(model, state, device):

    state_tensor = torch.as_tensor(
        state,
        dtype=torch.float32,
        device=device
    )

    state_tensor /= 255.0

    state_tensor = state_tensor.unsqueeze(0)

    with torch.no_grad():

        q_values = model(
            state_tensor
        )

    q_values = q_values[0].cpu().numpy()

    action = int(
        np.argmax(q_values)
    )

    return action, q_values


# =========================================================
# RUN MODEL
# =========================================================

def run_model(
    checkpoint_path,
    device
):

    print()
    print("=" * 70)
    print(
        f"TESTING: {checkpoint_path}"
    )
    print("=" * 70)

    env = MarioEnvironment()

    model = load_model(
        checkpoint_path,
        device
    )

    state, info = env.reset()

    total_reward = 0.0

    max_x = 0
    max_score = 0
    max_coins = 0

    action_counts = np.zeros(
        7,
        dtype=np.int32
    )

    last_info = {}

    print()
    print(
        "Starting from fresh Mario state..."
    )

    for step in range(1, MAX_STEPS + 1):

        action, q_values = get_action(
            model,
            state,
            device
        )

        action_counts[action] += 1

        (
            next_state,
            reward,
            terminated,
            truncated,
            info
        ) = env.step(action)

        done = (
            terminated or
            truncated
        )

        total_reward += float(reward)

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

            last_info = info.copy()

        # -------------------------------------------------
        # Print first 20 decisions
        # -------------------------------------------------

        if step <= 20:

            print(
                f"Step {step:3d} | "
                f"Action: "
                f"{ACTION_NAMES[action]:12s} | "
                f"Reward: "
                f"{float(reward):+6.2f} | "
                f"X: "
                f"{last_info.get('x_pos', 0):4d}"
            )

        # -------------------------------------------------
        # Print every 50 steps
        # -------------------------------------------------

        elif step % 50 == 0:

            print(
                f"Step {step:3d} | "
                f"Reward: "
                f"{total_reward:8.2f} | "
                f"Max X: "
                f"{max_x:4d} | "
                f"Last action: "
                f"{ACTION_NAMES[action]}"
            )

        state = next_state

        if done:

            print()
            print(
                f"💀 Episode ended at step {step}"
            )

            break

    # =====================================================
    # RESULTS
    # =====================================================

    print()
    print("-" * 70)
    print(
        f"RESULT: {checkpoint_path}"
    )
    print("-" * 70)

    print(
        f"Steps survived: {step}"
    )

    print(
        f"Total reward:   {total_reward:.2f}"
    )

    print(
        f"Maximum X:      {max_x}"
    )

    print(
        f"Maximum score:  {max_score}"
    )

    print(
        f"Maximum coins:  {max_coins}"
    )

    print(
        f"World:          "
        f"{last_info.get('world', '?')}"
    )

    print(
        f"Stage:          "
        f"{last_info.get('stage', '?')}"
    )

    print(
        f"Flag reached:   "
        f"{last_info.get('flag_get', False)}"
    )

    print()

    print("ACTION DISTRIBUTION")

    for i in range(7):

        percentage = (
            action_counts[i]
            / max(step, 1)
            * 100
        )

        print(
            f"{ACTION_NAMES[i]:15s}: "
            f"{action_counts[i]:4d} "
            f"({percentage:5.1f}%)"
        )

    env.close()

    return {
        "checkpoint": checkpoint_path,
        "steps": step,
        "reward": total_reward,
        "max_x": max_x,
        "score": max_score,
        "coins": max_coins,
        "flag": last_info.get(
            "flag_get",
            False
        ),
        "actions": action_counts,
    }


# =========================================================
# MAIN
# =========================================================

def main():

    print("=" * 70)
    print("       MARIO AI — 500 STEP BEHAVIOR TEST")
    print("=" * 70)

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print()
    print("Device:", device)

    if torch.cuda.is_available():

        print(
            "GPU:",
            torch.cuda.get_device_name(0)
        )

    results = []

    # -----------------------------------------------------
    # Test both models
    # -----------------------------------------------------

    for checkpoint in CHECKPOINTS:

        result = run_model(
            checkpoint,
            device
        )

        results.append(result)

    # =====================================================
    # FINAL COMPARISON
    # =====================================================

    print()
    print()
    print("=" * 70)
    print("                 FINAL VERDICT")
    print("=" * 70)

    for result in results:

        print()

        print(
            result["checkpoint"]
        )

        print(
            f"  Steps:     {result['steps']}"
        )

        print(
            f"  Reward:    {result['reward']:.2f}"
        )

        print(
            f"  Max X:     {result['max_x']}"
        )

        print(
            f"  Score:     {result['score']}"
        )

        print(
            f"  Coins:     {result['coins']}"
        )

        print(
            f"  Flag:      {result['flag']}"
        )

    # -----------------------------------------------------
    # Difference
    # -----------------------------------------------------

    if len(results) == 2:

        old = results[0]
        new = results[1]

        print()
        print("=" * 70)
        print("                 IMPROVEMENT")
        print("=" * 70)

        print()

        print(
            f"Reward change: "
            f"{new['reward'] - old['reward']:+.2f}"
        )

        print(
            f"X position change: "
            f"{new['max_x'] - old['max_x']:+d}"
        )

        print(
            f"Score change: "
            f"{new['score'] - old['score']:+d}"
        )

        print()

        if new["max_x"] > old["max_x"]:

            print(
                "✅ 100k model progressed farther!"
            )

        elif new["max_x"] < old["max_x"]:

            print(
                "❌ 100k model progressed less far."
            )

        else:

            print(
                "⚠️ Both models reached the same X position."
            )

    print()
    print("=" * 70)
    print("                       DONE")
    print("=" * 70)


# =========================================================
# START
# =========================================================

if __name__ == "__main__":
    main()