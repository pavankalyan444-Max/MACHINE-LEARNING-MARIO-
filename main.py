# import gymnasium as gym
# import gym_super_mario_bros

# env = gym.make("SuperMarioBros-v0")

# print(env.action_space)
# print(env.observation_space)

# observation, info = env.reset()

# print(observation.shape)
# print(observation.dtype)

# print(env.action_space)


# # import matplotlib.pyplot as plt

# # plt.imshow(observation)
# # plt.axis("off")
# # plt.show()
# action = 5
# observation, reward, terminated, truncated, info = env.step(action)


# from argparse import Action
# from fileinput import close

# import gymnasium as gym
# import gym_super_mario_bros

# env = gym.make("SuperMarioBros-v0")

# observation, info = env.reset()

# for step in range(1000):

#     action = env.action_space.sample()

#     observation, reward, terminated, truncated, info = env.step(action)

#     print(
#         "Step:", step,
#         "Action:", action,
#         "Reward:", reward
#     )

#     if terminated or truncated:
#         observation, info = env.reset()

# env.close()



# import gymnasium as gym
# import gym_super_mario_bros

# env = gym.make("SuperMarioBros-v0")

# print("Action space:")
# print(env.action_space)

# print("\nObservation space:")
# print(env.observation_space)

# import gym_super_mario_bros.actions as actions

# print(dir(actions))


# import gymnasium as gym
# import gym_super_mario_bros
# from gym_super_mario_bros.actions import SIMPLE_MOVEMENT
# from nes_py.wrappers import JoypadSpace

# env = gym.make("SuperMarioBros-v0")

# env = JoypadSpace(
#     env,
#     SIMPLE_MOVEMENT
# )



# from gym_super_mario_bros.actions import SIMPLE_MOVEMENT
# import gymnasium as gym
# import gym_super_mario_bros
# from gym_super_mario_bros.actions import SIMPLE_MOVEMENT
# from nes_py.wrappers import JoypadSpace

# print(SIMPLE_MOVEMENT)

# for i, action in enumerate(SIMPLE_MOVEMENT):
#     print(i, action)

# actions = [
#     2,
#     2,
#     2,
#     5,
#     5,
#     2,
#     2
# ]

# for action in actions:

#     observation, reward, terminated, truncated, info = env.step(action)
    
#     print(
#         "Action:", action,
#         "Reward:", reward
#     )

#     if terminated or truncated:
#         break

# import gymnasium as gym
# import gym_super_mario_bros

# from gym_super_mario_bros.actions import SIMPLE_MOVEMENT
# from nes_py.wrappers import JoypadSpace

# env = gym.make("SuperMarioBros-v0")

# env = JoypadSpace(
#     env,
#     SIMPLE_MOVEMENT
# )

# print("Action space:")
# print(env.action_space)

# print("\nAvailable actions:")

# for i, action in enumerate(SIMPLE_MOVEMENT):
#     print(i, action)

# print("\nObservation space:")
# print(env.observation_space)

# import matplotlib.pyplot as plt

# observation, info = env.reset()

# plt.imshow(observation)
# plt.title("Mario Observation")
# plt.axis("off")
# plt.show()

# import matplotlib.pyplot as plt

# observation, info = env.reset()

# print("Shape:", observation.shape)
# print("Data type:", observation.dtype)
# print("Minimum:", observation.min())
# print("Maximum:", observation.max())

# plt.imshow(observation)
# plt.axis("off")
# plt.show()

# import cv2

# gray = cv2.cvtColor(observation, cv2.COLOR_RGB2GRAY)

# print("Original:", observation.shape)
# print("Grayscale:", gray.shape)

# plt.imshow(gray, cmap="gray")
# plt.axis("off")
# plt.show()

# resized = cv2.resize(gray, (84, 84))

# print("Resized:", resized.shape)

# plt.imshow(resized, cmap="gray")
# plt.axis("off")
# plt.show()

# normalized = resized / 255.0

# print("Minimum:", normalized.min())
# print("Maximum:", normalized.max())
# print("Data type:", normalized.dtype)