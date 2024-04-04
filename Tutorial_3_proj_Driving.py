import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.evaluation import evaluate_policy
import os

Test_env = True

##########################
## Load the environment ##
##########################

environmant_name = 'CarRacing-v2'
env = gym.make(environmant_name, render_mode='human')

print(env.observation_space)
print(env.action_space)


##################
## Test the env ##
##################

if Test_env:
    print('Testing the environment')
   
    episodes = 5
    for episode in range(episodes+1):
        obs = env.reset()
        done = False
        score = 0

        while not done:
            env.render()
            action = env.action_space.sample()
            obs, reward, done, trun, info = env.step(action)
            score += reward

        print(f'Episode: {episode} Score: {score}')
        env.close()
    