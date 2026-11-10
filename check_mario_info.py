from environment import MarioEnvironment

env = MarioEnvironment()

state, info = env.reset()

print("INFO KEYS:")
print(info.keys())

for i in range(10):
    state, reward, terminated, truncated, info = env.step(1)
    print(info)

env.close()