import gymnasium as gym
from stable_baselines3 import A2C
from stable_baselines3.common.vec_env import VecFrameStack
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.env_util import make_atari_env
import os


Test_env = False
Train_agent = False
Use_the_agent = True

##########################
## Load the environment ##
##########################

environment_name = 'Breakout-v0'

env0 = gym.make(environment_name, render_mode='human')
env0.metadata['render_fps'] = 30 # set fps for rendering

print(env0.observation_space)
print(env0.action_space)

# The observation space is a Box space with shape (210, 160, 3). This means that the observation is an image with a height of 210 pixels, a width of 160 pixels, and three color channels (RGB). 
# The action space is a Discrete space with 4 actions. This means that the agent can take one of four actions: 0 (no operation), 1 (fire), 2 (move right), and 3 (move left).

##########################
## Test the environment ##
##########################

if Test_env:
    print('Testing the environment')
   
    episodes = 5
    for episode in range(episodes+1):
        obs = env0.reset()
        done = False
        score = 0

        while not done:
            env0.render()
            action = env0.action_space.sample()
            obs, reward, done, trun, info = env0.step(action)
            score += reward

        print(f'Episode: {episode} Score: {score}')
        env0.close()

###############################
## Vectorise the environment ##
###############################

# Note: in Tutorial_1 we did not vectorize the environment, here we do (VecFrameStack). This allows to run multiple environments in parallel, speeding up training (TRAIN IN PARALLEL)
# To play with 4 different env at once:
env4 = make_atari_env(environment_name, n_envs=4, seed=0) # This function is used to create a vectorized environment for Atari games. The environment_name parameter specifies which Atari game to use. The n_envs=4 parameter indicates that four parallel environments should be created. The seed=0 parameter sets the seed for the random number generator in the environment, ensuring that the results are reproducible.
#env4.metadata['render_modes'] = ["rgb_array"] # set render modes
env4.metadata['render_fps'] = 30 # set fps for rendering
env4 = VecFrameStack(env4, n_stack=4) # stack envs together

env4.reset()
env4.render('human')


#################
## Train Model ##
#################

log_path = os.path.join('Training', 'Logs')
A2C_path = os.path.join('Training', 'Saved Models', 'A2C_Model_Breakout')
model = A2C('CnnPolicy', env4, verbose=1, tensorboard_log=log_path) # The A2C algorithm is used to train the model. The 'CnnPolicy' parameter specifies that a convolutional neural network is used since the obs are images

if Train_agent:
    print('Training the agent')
    model.learn(total_timesteps=200000)
    model.save(A2C_path)
else:
    print('Loading the agent')
    model = A2C.load(A2C_path, env=env4)


########################
## Evaluate the Model ##
########################

print('Evaluating the model')
# To evaluate the model we have to define a env that is not vectorized, but need to be stacked
env1 = make_atari_env(environment_name, n_envs=1, seed=0)
#env1.metadata['render_modes'] = ["rgb_array"] # set render modes
env1.metadata['render_fps'] = 30 # set fps for rendering
env1 = VecFrameStack(env1, n_stack=4)

# env.metadata['render_fps'] = 120 # set fps for rendering
# env.metadata['render_modes'] = 'human'

mean_rew, std_rew = evaluate_policy(model, env1, n_eval_episodes=10, render=True) # The evaluate_policy function is used to evaluate the model. The model parameter specifies the model to evaluate. The env parameter specifies the environment to use. The n_eval_episodes=10 parameter specifies that the model should be evaluated over ten episodes. The render=True parameter specifies that the evaluation should be rendered to the screen.
print(f'Mean reward: {mean_rew} +/- {std_rew}')

# Note: in this case you can see that the model is not performing well. This is because the model is not trained for long enough. To improve the performance of the model, you can train it for more timesteps.


###################
## Use the Agent ##
###################

if Use_the_agent:
    print('Using the agent')
    episodes = 100
    for episode in range(1, episodes+1):
        obs = env1.reset()
        done = False
        score = 0

        while not done:
            env1.render('human')
            action, _ = model.predict(obs) # The model predicts the action to take based on the observation. output: action and the value of the next state (used in recurrent policies)
            obs, reward, done, info = env1.step(action) # The environment takes the action and returns the new observation, the reward, if the episode is done, and additional information
            score += reward
            
        print(f'Episode: {episode}, Score: {score}')
    env1.close()
