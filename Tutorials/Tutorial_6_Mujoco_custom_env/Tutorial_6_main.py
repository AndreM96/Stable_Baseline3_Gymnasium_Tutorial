import pickle
import gymnasium as gym
#import pytest
from manipulate_cable import MujocoManipulateCableEnv
import tensorboard


from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.evaluation import evaluate_policy

import numpy as np
import os
 
Test_env = False
Train_agent = False
Test_evaluation = False
Use_the_agent = True

print('Loading the environment')

#When train the model remember to change the render_mode to 'rgb_array' to not visualize the environment and reduce the computation time
env = gym.make('ManipulateCableEnv-v0', render_mode='human', max_episode_steps=100)

# Analize the environment
#print(env.observation_space)
print(env.action_space)

##########################
## Test the environment ##
##########################

if Test_env:
    print('Testing the environment')
    episodes = 5
    for episode in range(1, episodes+1):
        state = env.reset()
        trun = False
        score = 0
        
        while not trun:
            env.render()
            action = env.action_space.sample()
            obs, reward, done, trun, info = env.step(action)
            score += reward
            

        print(f'Episode: {episode}, Score: {score}')
    env.close()

####################
## Train an agent ## 
####################

# first create folders to save the logs and the model (folder Training, subfolders Logs and Saved Models)
log_path = os.path.join('Training', 'Logs')
PPO_path = os.path.join('Training', 'Saved Models', 'PPO_Model_Cable_Manipulation')

#env = DummyVecEnv([lambda: env]) # The environment must be wrapped in DummyVecEnv (vectorized environment) to be used with Stable Baselines. Lambda is used to create a function that returns the environment.
                                 # Here we did not vectorize the environment, see Tutorial 2. vectorize = run multiple environments in parallel, speeding up training
if Train_agent:
    print('Training the agent')
    model = PPO('MultiInputPolicy', env, verbose = 1, tensorboard_log=log_path) # Create a PPO model: agent with a Multi-Layer Perceptron (Mlp) policy (structure of the NN) and the environment. (verbose = 1) to see the training process.
                                                                         # The tensorboard_log parameter allows to log the training process and visualize it in tensorboard.
    # help(PPO) # To see the documentation of the model

    model.learn(total_timesteps=1000000) # Train the model for 20000 timesteps (iterations)   
    model.save(PPO_path) # Save the model
else:
    print('Loading the model')
    env = Monitor(env, log_path) # Monitor the environment to log the training process and visualize it in tensorboard
    model = PPO.load(PPO_path,env=env)  # Load the model
    env1 = model.get_env() # Get the environment used by the model

#################################
## Test and evaluate the agent ## 
#################################

if Test_evaluation:
    print('Testing and evaluating the agent')
    mean_rew, std_dev = evaluate_policy(model, env1, n_eval_episodes=10, render=True) # Evaluate the model for 10 episodes and render the environment (render = True)pisodes = 5
    print(f'Mean reward: {mean_rew}, Standard deviation: {std_dev}')
    # output: mean reward and standard deviation. It gets +1 for each step the pole is up, and zero otherwise. The maximum score is 500 (max 500 steps per episode).


###################
## Use the agent ## 
###################

if Use_the_agent:
    print('Using the agent')
    episodes = 5
    for episode in range(1, episodes+1):
        state = env1.reset()
        done = False
        score = 0
        
        while not done:
            env.render()
            action, _ = model.predict(state)  # Use the model to select an action
            #print(action)
            obs, reward, done, info = env1.step(action)
            score += reward
            

        print(f'Episode: {episode}, Score: {score}')
    env.close()