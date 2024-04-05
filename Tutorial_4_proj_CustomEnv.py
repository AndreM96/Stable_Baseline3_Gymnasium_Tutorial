import gymnasium as gym
from gymnasium import Env # super class for all environmentsMultiBinary(4)
from gymnasium.spaces import Box, Discrete, Dict, Tuple, MultiBinary, MultiDiscrete

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.evaluation import evaluate_policy

import numpy as np
import random
import os

Analyse_space = False 
Test_env = False  
Train_agent = False
Use_the_agent = True


#############################
## Analyse kinds of spaces ##
#############################

if Analyse_space:
    # four kind of spaces: Discrete, Box, MultiBinary, MultiDiscrete
    print(Discrete(3).sample()) # 0, 1 or 2
    print(Box(-2, 3, shape=(3,3)).sample()) # 3x3 array of floats between -2 and 3
    print(MultiBinary(4).sample()) # array of 4 zeros or ones
    print(MultiDiscrete([5,2,2]).sample()) # array of 3 integers: 0 to 4, 0 to 1, 0 to 1
    
    # two composite spaces (wrapper space): Tuple, Dict
    print(Dict({"height": Discrete(2), "speed": Box(0, 100, shape=(1,)), "color": MultiBinary(4)}).sample()) # dictionary with 2 keys
    print(Tuple((Discrete(3), Box(0, 1, shape=(4,)))).sample()) # tuple with 2 elements 
                                                                # !!! TUPLE ARE NOT SUPPORTED BY STABLE BASELINES 3, USE DICT !!!


##########################
## Build an environment ##
##########################

# Build an agent to give us the best shower possible
# Temperature varies randomly. The goal is to keep it between 37 and 39 deg

class ShowerEnv(Env):
    def __init__(self):
        self.action_space = Discrete(3) # 3 actions: decrease, keep, increase
        self.observation_space = Box(low = 0, high = 100, shape = (1,)) # or analagously Box(low = np.array([0]), high = np.array([100])). temperature from 0 to 100 deg
        self.state = 38 + random.randint(-3, 3)
        self.shower_lenght = 60

    def step(self, action):
        # Apply action
        self.state += action -1

        # Reduce shower length by 1 second
        self.shower_lenght -= 1 

        # Calculate reward
        if self.state >= 37 and self.state <= 39:
            reward = 1
        else:
            reward = -1
            
        # Check if shower is done
        if self.shower_lenght <= 0:
            done = True
        else:
            done = False

        info = {}
        trun = False

        return self.state, reward, done, trun, info
    
    def render(self):
        pass # to do

    def reset(self,  seed=None):
        self.state = np.array([38 + random.randint(-3, 3)]).astype(float)
        self.shower_lenght = 60
        return self.state, {}

##########################
## Test the environment ##
##########################

my_env = ShowerEnv()
my_env = DummyVecEnv([lambda: my_env])


if Test_env:
    print('Testing the environment')
    episodes = 5
    for episode in range(1, episodes+1):
        state = my_env.reset()
        done = False
        score = 0

        while not done:
            action = my_env.action_space.sample()
            obs, reward, done, trun, info = my_env.step(action)
            score += reward
            
        print(f'Episode: {episode}, Score: {score}')
    my_env.close()

####################
## Train an agent ##
####################

log_path = os.path.join('Training', 'Logs')
PPO_path = os.path.join('Training', 'Saved Models', 'PPO_Model_Shower')


if Train_agent:
    model = PPO('MlpPolicy', my_env, verbose=1, tensorboard_log=log_path)
    model.learn(total_timesteps=100000)
    model.save(PPO_path)
else:
    model = PPO.load(PPO_path, env=my_env)


##########################
## Test the environment ##
##########################

print('Evaluating the model')
mean_rew, std_rew = evaluate_policy(model, my_env, n_eval_episodes=100, render=False) # The evaluate_policy function is used to evaluate the model. The model parameter specifies the model to evaluate. The env parameter specifies the environment to use. The n_eval_episodes=10 parameter specifies that the model should be evaluated over ten episodes. The render=True parameter specifies that the evaluation should be rendered to the screen.
print(f'Mean reward: {mean_rew} +/- {std_rew}')

# Note: in this case you can see that the model is not performing well. This is because the model is not trained for long enough. To improve the performance of the model, you can train it for more timesteps.

if Use_the_agent:
    print('Using the agent')
    episodes = 50
    for episode in range(1, episodes+1):
        obs = my_env.reset()
        done = False
        score = 0

        while not done:
            my_env.render()
            action, _ = model.predict(obs) # The model predicts the action to take based on the observation. output: action and the value of the next state (used in recurrent policies)
            obs, reward, done, info = my_env.step(action) # The environment takes the action and returns the new observation, the reward, if the episode is done, and additional information
            score += reward
            
        print(f'Episode: {episode}, Score: {score}')
    my_env.close()
