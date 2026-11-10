import torch
import numpy as np

from environment import MarioEnvironment
from model import MarioCNN
from checkpoint_utils import checkpoint_value, model_path


# =========================================================
# SETTINGS
# =========================================================

CHECKPOINT_1 = model_path("mario_checkpoint_10000.pth")
CHECKPOINT_2 = model_path("mario_checkpoint_100000.pth")

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

    return model, checkpoint


# =========================================================
# GET Q VALUES
# =========================================================

def get_q_values(model, state, device):

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

    return q_values[0].cpu().numpy()


# =========================================================
# COMPARE WEIGHTS
# =========================================================

def compare_weights(model1, model2):

    total_difference = 0.0
    total_weight = 0.0
    max_difference = 0.0

    for p1, p2 in zip(
        model1.parameters(),
        model2.parameters()
    ):

        difference = (
            p1.detach() - p2.detach()
        ).abs()

        total_difference += (
            difference.sum().item()
        )

        total_weight += (
            p1.numel()
        )

        max_difference = max(
            max_difference,
            difference.max().item()
        )

    average_difference = (
        total_difference /
        total_weight
    )

    return (
        total_difference,
        average_difference,
        max_difference
    )


# =========================================================
# MAIN
# =========================================================

def main():

    print("=" * 70)
    print("       MARIO AI — MODEL BEHAVIOR DIAGNOSTIC")
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

    # -----------------------------------------------------
    # Load models
    # -----------------------------------------------------

    print()
    print("Loading 10k model...")

    model_10k, checkpoint_10k = load_model(
        CHECKPOINT_1,
        device
    )

    print(
        "Saved step:",
        checkpoint_10k.get("step")
    )

    print()
    print("Loading 100k model...")

    model_100k, checkpoint_100k = load_model(
        CHECKPOINT_2,
        device
    )

    print(
        "Saved step:",
        checkpoint_100k.get("step")
    )

    # -----------------------------------------------------
    # Compare weights
    # -----------------------------------------------------

    print()
    print("=" * 70)
    print("                 WEIGHT COMPARISON")
    print("=" * 70)

    (
        total_difference,
        average_difference,
        max_difference
    ) = compare_weights(
        model_10k,
        model_100k
    )

    print()
    print(
        f"Total absolute difference: "
        f"{total_difference:.6f}"
    )

    print(
        f"Average weight difference:  "
        f"{average_difference:.10f}"
    )

    print(
        f"Maximum weight difference:  "
        f"{max_difference:.6f}"
    )

    # -----------------------------------------------------
    # Create ONE starting state
    # -----------------------------------------------------

    print()
    print("=" * 70)
    print("             CREATING SAME MARIO STATE")
    print("=" * 70)

    env = MarioEnvironment()

    state, info = env.reset()

    print()
    print("State shape:", state.shape)
    print("State dtype:", state.dtype)

    if info:
        print("Initial info:", info)

    # -----------------------------------------------------
    # Get Q-values
    # -----------------------------------------------------

    q_10k = get_q_values(
        model_10k,
        state,
        device
    )

    q_100k = get_q_values(
        model_100k,
        state,
        device
    )

    # -----------------------------------------------------
    # Print 10k
    # -----------------------------------------------------

    print()
    print("=" * 70)
    print("                  10K MODEL")
    print("=" * 70)

    for i in range(7):

        print(
            f"{ACTION_NAMES[i]:15s}"
            f" Q = {q_10k[i]:12.6f}"
        )

    action_10k = int(
        np.argmax(q_10k)
    )

    print()
    print(
        "BEST ACTION:",
        ACTION_NAMES[action_10k]
    )

    # -----------------------------------------------------
    # Print 100k
    # -----------------------------------------------------

    print()
    print("=" * 70)
    print("                 100K MODEL")
    print("=" * 70)

    for i in range(7):

        print(
            f"{ACTION_NAMES[i]:15s}"
            f" Q = {q_100k[i]:12.6f}"
        )

    action_100k = int(
        np.argmax(q_100k)
    )

    print()
    print(
        "BEST ACTION:",
        ACTION_NAMES[action_100k]
    )

    # -----------------------------------------------------
    # Q-value difference
    # -----------------------------------------------------

    print()
    print("=" * 70)
    print("                 Q-VALUE CHANGE")
    print("=" * 70)

    print()

    for i in range(7):

        difference = (
            q_100k[i] -
            q_10k[i]
        )

        print(
            f"{ACTION_NAMES[i]:15s}"
            f" 10k = {q_10k[i]:10.4f}"
            f" | 100k = {q_100k[i]:10.4f}"
            f" | Change = {difference:+10.4f}"
        )

    # -----------------------------------------------------
    # Decision comparison
    # -----------------------------------------------------

    print()
    print("=" * 70)
    print("                 DECISION TEST")
    print("=" * 70)

    print()

    print(
        "10k best action :",
        ACTION_NAMES[action_10k]
    )

    print(
        "100k best action:",
        ACTION_NAMES[action_100k]
    )

    if action_10k == action_100k:

        print()
        print(
            "⚠️ BOTH MODELS CHOOSE THE SAME ACTION."
        )

    else:

        print()
        print(
            "✅ MODELS CHOOSE DIFFERENT ACTIONS."
        )

    # -----------------------------------------------------
    # Q-value distance
    # -----------------------------------------------------

    q_difference = np.abs(
        q_100k - q_10k
    )

    print()
    print(
        "Average Q-value difference:",
        f"{q_difference.mean():.6f}"
    )

    print(
        "Maximum Q-value difference:",
        f"{q_difference.max():.6f}"
    )

    # -----------------------------------------------------
    # Cleanup
    # -----------------------------------------------------

    env.close()

    print()
    print("=" * 70)
    print("                    DONE")
    print("=" * 70)


# =========================================================
# START
# =========================================================

if __name__ == "__main__":
    main()