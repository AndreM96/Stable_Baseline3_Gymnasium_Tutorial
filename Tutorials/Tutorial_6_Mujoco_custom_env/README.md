# Tutorial 6: Mujoco Custom Environment

This folder contains the code and resources for Tutorial 6 of the RL Tutorial series on Stable Baselines3 and Gymnasium. In this tutorial, we will explore how to create a custom environment using Mujoco.

## Useful links
This tutorial is a readaptation of the fetch and push environment developed by Gymnasium Robotics. Please refer to https://github.com/Farama-Foundation/Gymnasium-Robotics/tree/main/gymnasium_robotics/envs/fetch

## Prerequisites

Before running the code in this tutorial, make sure you have the following prerequisites installed:

- Python 3.8
- Stable Baselines3
    - `pip install stable-baselines3[extra]`
- Gymnasium Robotics
    - `pip install gymnasium-robotics`
- Mujoco
    - `pip install mujoco`

## Getting Started

In the repository you will find:

- *robot_env.py* : This file contains the parent classes where the basic attributes and methods are defined. Then, these classes will be inherited by the classes defined in the following scripts. Here,the action and observation spaces are defined.
- *slider_env.py* : This file defines the agent. It allows you to set the variables you want to observe, the action you want the agent to perform, and the reward policy. In this file, you will also interface with Mujoco to retrieve information about the state of the variables of interest and set the subsequent actions to run the Mujoco simulation.
- *manipulate_cable.py* : This file initializes the environment. Here, you can set various arguments such as the model path, initial joint position, and distance threshold. Additionally, the developed environment is registered as a Gymnasium environment. For a more detailed description of the environment, refer to this file.

**!! REMARK !!** In each of these files you will find two types of classes: MujocoPy...Env and Mujoco...Env (e.g. In slider.py you will find the class MujocoPySliderEnv and the class MujocoSliderEnv). These classes allow the interface with the Mujoco engine, let it possible to retrieve the information about the state or set the action of our varibles of interest. They are defined in the same way and with the same structure, the use of one or the other depends on the Mujoco Python bindings used. The class MujocoPy...Env depends on `mujoco_py` which is no longer maintained. On the other hand, the class Mujoco...Env depends on the newest [mujoco python bindings](https://mujoco.readthedocs.io/en/latest/python.html) maintained by the MuJoCo team in Deepmind. The latter is the class used for this tutorial.


- The main: *Tutorial_6_main.py* creates the environment and then the RL algorithm can be applied, it follows the same structure as Tutorial 1,2,3

- The folder *Mujoco_xml* contains the xml file that describes the mujoco scene. The scene.xml file depicts all the scene including the cable.xml where the cable and the slider are defined. For futher information on how to understand the different components refer to [Mujoco XML Reference](https://mujoco.readthedocs.io/en/stable/XMLreference.html)

- The folder *Mujoco_data_log* contains the log files that describes the mujoco scene (e.g the number of joints, the name and position of the bodies etc...). For futher information on how to read these files refer to [Mujoco API Types](https://mujoco.readthedocs.io/en/stable/APIreference/APItypes.html#mjmodel) 


## Resources

- [Stable Baselines3 Documentation](https://stable-baselines3.readthedocs.io/)
- [Gymnasium Documentation](https://github.com/Farama-Foundation/Gymnasium)
- [Gymnasium Robotics Documentation](https://github.com/Farama-Foundation/Gymnasium-Robotics)
- [Mujoco Documentation](https://mujoco.org/)


