# Gymnsaium Robotics includes a set of environments for robotic manipulation tasks based on mujoco.
# Algorithm from stable baseline 3 can be used to solve these tasks (see previous tutorial). 
# This is a simple tutorial that shows how to load and test and environment from Gymnasium Robotics.

import gymnasium as gym

env = gym.make("FetchPickAndPlace-v2", render_mode="human")
observation, info = env.reset(seed=42)

for _ in range(1000):
   action = env.action_space.sample()  # to replace with a User-defined policy function
   observation, reward, terminated, truncated, info = env.step(action)

   if terminated or truncated:
      observation, info = env.reset()
env.close()