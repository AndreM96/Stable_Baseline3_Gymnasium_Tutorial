import pickle
import gymnasium as gym
from manipulate_cable import MujocoManipulateCableEnv


from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3 import A2C
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.vec_env import VecFrameStack

from stable_baselines3.common.callbacks import EvalCallback, StopTrainingOnRewardThreshold
import tensorboard
import numpy as np
import os
 
if __name__ == '__main__':
    Test_env = False
    Train_agent = False
    Test_evaluation = False
    Use_the_agent = True
    Train_with_callbacks = False
    Train_with_mod_NN = False
    Train_with_diff_alg = False

    print('Loading the environment')

    #When train the model remember to change the render_mode to 'rgb_array' to not visualize the environment and reduce the computation time
    env_name = 'ManipulateCableEnv-v0'
    env = gym.make(env_name, render_mode='rgb_array', max_episode_steps=100)

    # Analize the environment
    #print(env.observation_space)
    #print(env.action_space)

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

    ###############################
    ## Vectorise the environment ##
    ###############################
    env8 = make_vec_env("ManipulateCableEnv-v0", n_envs=4, vec_env_cls=SubprocVecEnv)
    env8 = VecFrameStack(env8, n_stack=4) # stack envs together
    env8.metadata['render_fps'] = 30 # set fps for rendering


    env8.reset()
    env8.render('rgb_array')

    #################
    ## Train Model ##
    #################

    log_path = os.path.join('Training', 'Logs')
    A2C_path = os.path.join('Training', 'Saved Models', 'A2C_Model_Breakout')
    model = A2C("MultiInputPolicy", env8, verbose=1, tensorboard_log=log_path) # The A2C algorithm is used to train the model. The policy is a MultiInputPolicy. The device is the CPU.

    if Train_agent:
        print('Training the agent')
        model.learn(total_timesteps=500000)
        model.save(A2C_path)
    else:
        print('Loading the agent')
        model = A2C.load(A2C_path, env=env8)

    ###################
    ## Use the agent ## 
    ###################

    if Use_the_agent:
        print('Using the agent')
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

    