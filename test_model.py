import torch

from model import MarioCNN


# -------------------------------------------------------------
# Create model
# -------------------------------------------------------------

model = MarioCNN(
    num_actions=7
)


# -------------------------------------------------------------
# Create fake Mario state
#
# Shape:
#
# batch = 1
# channels = 4
# height = 84
# width = 84
# -------------------------------------------------------------

dummy_state = torch.randint(
    0,
    256,
    (1, 4, 84, 84),
    dtype=torch.uint8
)


# -------------------------------------------------------------
# Run through neural network
# -------------------------------------------------------------

output = model(
    dummy_state
)


# -------------------------------------------------------------
# Print results
# -------------------------------------------------------------

print("Input shape:")
print(dummy_state.shape)

print("\nOutput shape:")
print(output.shape)

print("\nQ-values:")
print(output)

