import torch

from environment import MarioEnvironment
from model import MarioCNN


# =============================================================
# 1. CREATE ENVIRONMENT
# =============================================================

env = MarioEnvironment()

print("Number of actions:")
print(env.num_actions)


# =============================================================
# 2. RESET ENVIRONMENT
# =============================================================

state, info = env.reset()

print("\nState shape:")
print(state.shape)

print("State dtype:")
print(state.dtype)


# =============================================================
# 3. CREATE MODEL
# =============================================================

model = MarioCNN(
    num_actions=env.num_actions
)


# =============================================================
# 4. CONVERT STATE TO TENSOR
# =============================================================

state_tensor = torch.tensor(
    state,
    dtype=torch.float32
)

state_tensor = state_tensor / 255.0


# =============================================================
# 5. ADD BATCH DIMENSION
# =============================================================

state_tensor = state_tensor.unsqueeze(0)

print("\nTensor shape:")
print(state_tensor.shape)


# =============================================================
# 6. SEND STATE THROUGH CNN
# =============================================================

with torch.no_grad():

    q_values = model(
        state_tensor
    )


# =============================================================
# 7. PRINT OUTPUT
# =============================================================

print("\nQ-values:")
print(q_values)

print("\nQ-value shape:")
print(q_values.shape)


# =============================================================
# 8. CHOOSE ACTION
# =============================================================

action = q_values.argmax(
    dim=1
).item()

print("\nChosen action:")
print(action)


# =============================================================
# 9. EXECUTE ACTION
# =============================================================

next_state, reward, terminated, truncated, info = (
    env.step(action)
)

print("\nEnvironment response:")
print("Next state:", next_state.shape)
print("Reward:", reward)
print("Terminated:", terminated)
print("Truncated:", truncated)


# =============================================================
# 10. CLOSE
# =============================================================

env.close()

