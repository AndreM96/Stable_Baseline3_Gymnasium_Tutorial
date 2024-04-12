import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.evaluation import evaluate_policy
import os

Test_env = False
Train_model = False
Use_agent = False
Evaluate_model = True

##########################
## Load the environment ##
##########################

environmant_name = 'CarRacing-v2'
env = gym.make(environmant_name, render_mode='human')

#The easiest control task to learn from pixels - a top-down racing environment. The generated track is random every episode.
#Some indicators are shown at the bottom of the window along with the state RGB buffer. From left to right: 
#true speed, four ABS sensors, steering wheel position, and gyroscope. 

print(env.observation_space)
# State consists of 96x96 rgb pixels (values from 0 to 255). The car is centered in the field of view, can move 3 directions, and can rotate.
print(env.action_space)
# continuous: There are 3 actions: steering (-1 is full left, +1 is full right), gas, and breaking. Stearing goes from -1 to 1, gas from 0 to 1, and breaking from 0 to 1.

# The reward is -0.1 every frame and +1000/N for every track tile visited, where N is the total number of tiles visited in the track. For example, if you have finished in 
# 732 frames, your reward is 1000 - 0.1*732 = 926.8 points.
# The episode finishes when all of the tiles are visited. The car can also go outside of the playfield - that is, far off the track, in which case it will receive -100 reward and die.
# Game is solverd when reward is greater than 900 points.

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
            print(f'Action: {action}')
            print(f'Observation: {obs}')
            print(f'Reward: {reward}')

        print(f'Episode: {episode} Score: {score}')
        env.close()
    

#####################
## Train the model ##
#####################

log_path = os.path.join('Training', 'Logs')
PPO_path = os.path.join('Training', 'Saved Models', 'PPO_Model_Driving')

env1 = gym.make(environmant_name, render_mode='rgb_array') # rgb otherways it will plot the image while training
envd= DummyVecEnv([lambda: env1])


if Train_model:
    print('Training the model')
    model = PPO('CnnPolicy', envd, verbose=1, tensorboard_log=log_path) # image input -> convolutional neural network
    model.learn(total_timesteps=200000)
    model.save(PPO_path)
else:
    print('Loading the model')
    model = PPO.load(PPO_path, env =envd)


###################
## Use the model ##
###################

if Use_agent:
    print('Using the agent')
    episodes = 10
    for episode in range(1, episodes+1):
        obs = envd.reset()
        done = False
        score = 0

        while not done:
            envd.render()
            action, _ = model.predict(obs) # The model predicts the action to take based on the observation. output: action and the value of the next state (used in recurrent policies)
            obs, reward, done, info = envd.step(action) # The environment takes the action and returns the new observation, the reward, if the episode is done, and additional information
            score += reward
            
        print(f'Episode: {episode}, Score: {score}')
    envd.close()

########################
## Evaluate the model ##
########################

if Evaluate_model:
    print('Evaluating the model')
    mean_rew, std_rew = evaluate_policy(model, env, n_eval_episodes=10, render=True) # Evaluate the model for 10 episodes and render the environment
    print(f'Mean reward: {mean_rew}, Std reward: {std_rew}')