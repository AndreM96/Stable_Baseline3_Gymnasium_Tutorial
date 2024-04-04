import os
import gymnasium as gym 
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.evaluation import evaluate_policy
import tensorboard
from stable_baselines3.common.callbacks import EvalCallback, StopTrainingOnRewardThreshold

Test_env= False
Train_agent = False
Test_evaluation = True
Use_the_agent = False
Train_with_callbacks = False
Train_with_mod_NN = False
Train_with_diff_alg = False


#########################
## Load an environment ## 
#########################

print('Loading the environment')
environment_name = 'CartPole-v1'
env = gym.make(environment_name, render_mode='human')

# Analize the environment
print(env.observation_space)
print(env.action_space)


##########################
## Test the environment ##
##########################

if Test_env:
    print('Testing the environment')
    episodes = 5
    for episode in range(1, episodes+1):
        state = env.reset()
        done = False
        score = 0

        while not done:
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
PPO_path = os.path.join('Training', 'Saved Models', 'PPO_Model_CartPole')

env = DummyVecEnv([lambda: env]) # The environment must be wrapped in DummyVecEnv (vectorized environment) to be used with Stable Baselines. Lambda is used to create a function that returns the environment.
                                 # Here we did not vectorize the environment, see Tutorial 2. vectorize = run multiple environments in parallel, speeding up training
if Train_agent:
    print('Training the agent')
    model = PPO('MlpPolicy', env, verbose = 1, tensorboard_log=log_path) # Create a PPO model: agent with a Multi-Layer Perceptron (Mlp) policy (structure of the NN) and the environment. (verbose = 1) to see the training process.
                                                                         # The tensorboard_log parameter allows to log the training process and visualize it in tensorboard.
    # help(PPO) # To see the documentation of the model

    model.learn(total_timesteps=20000) # Train the model for 20000 timesteps (iterations)   
    model.save(PPO_path) # Save the model
else:
    model = PPO.load(PPO_path, env =env)  # Load the model


#################################
## Test and evaluate the agent ## 
#################################

if Test_evaluation:
    print('Testing and evaluating the agent')
    mean_rew, std_dev = evaluate_policy(model, env, n_eval_episodes=10, render=True) # Evaluate the model for 10 episodes and render the environment (render = True)pisodes = 5
    print(f'Mean reward: {mean_rew}, Standard deviation: {std_dev}')
    # output: mean reward and standard deviation. It gets +1 for each step the pole is up, and zero otherwise. The maximum score is 500 (max 500 steps per episode).


###################
## Use the agent ## 
###################

if Use_the_agent:
    print('Using the agent')
    episodes = 5
    for episode in range(1, episodes+1):
        obs = env.reset()
        done = False
        score = 0

        while not done:
            env.render()
            action, _ = model.predict(obs) # The model predicts the action to take based on the observation. output: action and the value of the next state (used in recurrent policies)
            obs, reward, done, info = env.step(action) # The environment takes the action and returns the new observation, the reward, if the episode is done, and additional information
            score += reward
            
        print(f'Episode: {episode}, Score: {score}')
    env.close()



###################################################
## Visualize the training process in tensorboard ##
###################################################

# import tensorboard and search in the command palette for "tensorboard: Launch Tensorboard" to visualize the training process. Select the log directory and click on "Launch Tensorboard".


################################################################
## Main Metric to consider and quick fix to improve the model ##
################################################################

# The main metric to consider is the mean reward. The model is not learning if the mean reward is not increasing. The average length of episode can be used as a metric as well.
# Quick fix to improve the model:
# 1. Increase the number of timesteps (iterations) in the learn function.
# 2. Hyperparameter tuning: change the learning rate, the number of layers, the number of neurons, the activation function, the optimizer, the discount factor, the entropy coefficient, the gae lambda, the batch size, the number of epochs, the clip range, the value function coefficient, the max gradient norm, the target value function coefficient, the target entropy
#    Stable Baselines provides a hyperparameter optimization tool called Optuna. It can be used to find the best hyperparameters for the model.
# 3. Change the policy: use a different policy (CNN, LSTM, etc). Try different algorithms (A2C, DQN, etc).


###############
## Callbacks ##
###############

# Callbacks are used to save the model, log the training process, and stop the training when a certain condition is met.
# Useful when you have large training processes and you want to save the model at certain intervals, log the training process, and stop the training when the model is not learning anymore.
print('Definining callbacks')
save_path = os.path.join('Training', 'Saved Models')

stop_callback = StopTrainingOnRewardThreshold(reward_threshold=300, verbose=1) # Stop the training when the mean reward is greater than 300

# eval callback is used to evaluate the model at certain intervals and stop the training when the mean reward is greater than 300 (stop_callback)
# Every time a new best model is found, the stop_callback is checked. If the mean reward is greater than 300, the training stops. The best model is saved in the "best_model" parameter.
# The eval callback is evaluated every "eval_freq" timesteps. The best model is saved in the "best_model" parameter.
eval_callback = EvalCallback(env, 
                            callback_on_new_best = stop_callback, 
                            eval_freq = 10000, 
                            best_model_save_path = save_path, 
                            verbose = 1) # Evaluate the model every 10000 timesteps and stop the training when the mean reward is greater than 500


if Train_with_callbacks:
    print('Training the agent with callbacks')
    # create a new PPO model and assign the callback
    model = PPO('MlpPolicy', env, verbose = 1, tensorboard_log=log_path)
    model.learn(total_timesteps=20000, callback=eval_callback) # Train the model for 20000 timesteps (iterations) and assign the eval_callback


############################################
## Modify NN architecture (change policy) ##
############################################

net_arch = dict(pi = [128,128,128,128], vf = [128,128,128,128]) # Change the policy to a 4-layer NN with 128 neurons in each layer. pi = policy (actor), vf = value function (critic). net_arch is a dictionary. This dictionary has two keys: 'pi' and 'vf'. Each key maps to a list of four integers.
# see baseliane3 documentation for more information about custom policies

if Train_with_mod_NN:
    print('Training the agent with modified NN architecture and callbacks')
    model = PPO('MlpPolicy', env, verbose = 1, tensorboard_log=log_path, policy_kwargs = {'net_arch':net_arch}) # Create a PPO model with the new architecture. OBS: `dict(key=value)` is just another way to create a dictionary. It's equivalent to `{'key': value}`. 
    model.learn(total_timesteps=20000, callback = eval_callback) 


##############################
## Use different algorithms ##
##############################

from stable_baselines3 import DQN

if Train_with_diff_alg:
    print('Training the agent with DQN algorithm')
    DQN_path = os.path.join('Training', 'Saved Models', 'DQN_Model_CartPole')
    model = DQN('MlpPolicy', env, verbose = 1, tensorboard_log=log_path) # Create a DQN model
    model.learn(total_timesteps=20000) # Train the model for 20000 timesteps (iterations)
    model.save(DQN_path)
