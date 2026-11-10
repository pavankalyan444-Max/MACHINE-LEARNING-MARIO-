import torch
from checkpoint_utils import checkpoint_value, model_path

for file in [
    model_path("mario_checkpoint_10000.pth"),
    model_path("mario_checkpoint_100000.pth"),
]:
    checkpoint = torch.load(file, map_location="cpu")

    print("\n==============================")
    print(file)
    print("==============================")

    print("Saved step:", checkpoint.get("step"))
    print("Keys:", checkpoint.keys())

    policy = checkpoint_value(
        checkpoint,
        "policy_net",
        "model_state_dict"
    )

    # Calculate a simple fingerprint of the weights
    total = 0.0
    for value in policy.values():
        total += value.float().abs().sum().item()

    print("Weight fingerprint:", total)