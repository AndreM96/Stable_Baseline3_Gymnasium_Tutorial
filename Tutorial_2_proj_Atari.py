import gymnasium as gym
from stable_baselines3 import A2C
from stable_baselines3.common.vec_env import VecFrameStack
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.env_util import make_atari_env
import os


Test_env = False
Train_agent = False

##########################
## Load the environment ##
##########################

environment_name = 'Breakout-v4'
env = gym.make(environment_name, render_mode='human')
env.metadata['render_fps'] = 30 # set fps for rendering

print(env.observation_space)
print(env.action_space)

##########################
## Test the environment ##
##########################

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

###############################
## Vectorise the environment ##
###############################

# Note: in Tutorial_1 we did not vectorize the environment, here we do (VecFrameStack). This allows to run multiple environments in parallel, speeding up training (TRAIN IN PARALLEL)
# To play with 4 different env at once:
env = make_atari_env(environment_name, n_envs=4, seed=0) # This function is used to create a vectorized environment for Atari games. The environment_name parameter specifies which Atari game to use. The n_envs=4 parameter indicates that four parallel environments should be created. The seed=0 parameter sets the seed for the random number generator in the environment, ensuring that the results are reproducible.
env = VecFrameStack(env, n_stack=4) # stack envs together


#################
## Train Model ##
#################

log_path = os.path.join('Training', 'Logs')
A2C_path = os.path.join('Training', 'Saved Models', 'A2C_Model_Breakout')
model = A2C('CnnPolicy', env, verbose=1, tensorboard_log=log_path) # The A2C algorithm is used to train the model. The 'CnnPolicy' parameter specifies that a convolutional neural network is used since the obs are images

if Train_agent:
    print('Training the agent')
    model.learn(total_timesteps=100000)
    model.save(A2C_path)
else:
    print('Loading the agent')
    model = A2C.load(A2C_path, env=env)


########################
## Evaluate the Model ##
########################

print('Evaluating the model')
# To evaluate the model we have to define a env that is not vectorized, but need to be stacked
env = make_atari_env(environment_name, n_envs=1, seed=0)
env = VecFrameStack(env, n_stack=4)

env.metadata['render_fps'] = 30 # set fps for rendering
env.metadata['render_modes'] = 'human'

mean_rew, std_rew = evaluate_policy(model, env, n_eval_episodes=10, render=True) # The evaluate_policy function is used to evaluate the model. The model parameter specifies the model to evaluate. The env parameter specifies the environment to use. The n_eval_episodes=10 parameter specifies that the model should be evaluated over ten episodes. The render=True parameter specifies that the evaluation should be rendered to the screen.
print(f'Mean reward: {mean_rew} +/- {std_rew}')

# Note: in this case you can see that the model is not performing well. This is because the model is not trained for long enough. To improve the performance of the model, you can train it for more timesteps.

