import os

from gymnasium.utils.ezpickle import EzPickle
from gymnasium.envs.registration import register

from slider_env import MujocoSliderEnv, MujocoPySliderEnv

# Ensure we get the path separator correct on windows
xml_path = "Mujoco_xml/scene.xml"
dirname = os.path.dirname(__file__)
abspath = os.path.join(dirname + "/" + xml_path)
MODEL_XML_PATH  = abspath
#MODEL_XML_PATH = os.path.join("scene.xml")


class MujocoPyManipulateCableEnv(MujocoPySliderEnv, EzPickle):
    def __init__(self, reward_type="sparse", **kwargs):
        initial_qpos = {
            "slider:joint": -0.3,
        }
        MujocoPySliderEnv.__init__(
            self,
            model_path=MODEL_XML_PATH,
            n_substeps=20,
            target_offset=0.0,
            sld_range=0.1,
            target_range=[0.1, 0, 0.1],
            distance_threshold=0.05,
            initial_qpos=initial_qpos,
            reward_type=reward_type,
            **kwargs,
        )
        EzPickle.__init__(self, reward_type=reward_type, **kwargs)


class MujocoManipulateCableEnv(MujocoSliderEnv, EzPickle):
    """
    ## Description

    The task in the environment is for a slider to move the cable and let a specific point of it (B_11) reach a target position by going backward and forward with respect to its starting position.
    The environment is a 3D scene with the cable, the slider, and a target position. The slider is controlled by applying a force in the x direction.  The cable is attached to the slider and the target position is a 3D point in the space that a specific part of the cable has to reach. 
    The environment is considered solved when the desired part of the cable reaches the target position.
    

    The control frequency of the robot is of `f = 25 Hz`. This is achieved by applying the same action in 20 subsequent simulator step (with a time step of `dt = 0.002 s`) before returning the control to the robot.

    ## Action Space

    The action space is a `Box(-0.2, 0.2, (1,), float32)`. An action represents the Cartesian displacement dx, dy, and dz of the end effector. In addition to a last action that controls the slider movement.

    | Num | Action                                                 | Control Min | Control Max | Name (in corresponding XML file)                                | Joint  | Unit         |
    | --- | ------------------------------------------------------ | ----------- | ----------- | --------------------------------------------------------------- | -----  | ------------ |
    | 0   | Displacement of the slider in the x direction dx       | -1          | 1           | slider:mocap                                                    | slider | position (m) |
    
    ## Observation Space

    The observation is a `goal-aware observation space`. It consists of a dictionary with information about the slider state, the cable pos (achieved goal) and goal. The kinematics observations are derived from Mujoco bodies known as [sites](https://mujoco.readthedocs.io/en/latest/XMLreference.html?highlight=site#body-site) attached to the body of interest such as the cable part or the slider.
    Also to take into account the temporal influence of the step time, velocity values are multiplied by the step time dt=number_of_sub_steps*sub_step_time. The dictionary consists of the following 3 keys:

    * `observation`: its value is an `ndarray` of shape `(9,)`. It consists of kinematic information of the cable part we are interested in and the slider. The elements of the array correspond to the following:

    | Num | Observation                                                                                                                           | Min    | Max    | Site Name (in corresponding XML file) | Joint Name (in corresponding XML file) |Joint Type| Unit                     |
    |-----|---------------------------------------------------------------------------------------------------------------------------------------|--------|--------|---------------------------------------|----------------------------------------|----------|--------------------------|
    | 0   | Slider x position in global coordinates                                                                                               | -Inf   | Inf    | "slider:site"                         |-                                       |-         | position (m)             |
    | 1   | Slider y position in global coordinates                                                                                               | -Inf   | Inf    | "slider:site"                         |-                                       |-         | position (m)             |
    | 2   | Slider z position in global coordinates                                                                                               | -Inf   | Inf    | "slider:site"                         |-                                       |-         | position (m)             |
    | 3   | Cable part (B_11) x position in global coordinates                                                                                    | -Inf   | Inf    | cable (*) | B_11                      |-                                       |-         | position (m)             |
    | 4   | Cable part (B_11) y position in global coordinates                                                                                    | -Inf   | Inf    | cable (*) | B_11                      |-                                       |-         | position (m)             |
    | 5   | Cable part (B_11) z position in global coordinates                                                                                    | -Inf   | Inf    | cable (*) | B_11                      |-                                       |-         | position (m)             |
    | 6   | Slider linear velocity x direction                                                                                                    | -Inf   | Inf    | "slider:site"                         |-                                       |-         | velocity (m/s)           |
    | 7   | Slider linear velocity y direction                                                                                                    | -Inf   | Inf    | "slider:site"                         |-                                       |-         | velocity (m/s)           |
    | 8   | Slider linear velocity z direction                                                                                                    | -Inf   | Inf    | "slider:site"                         |-                                       |-         | velocity (m/s)           |
    
    * The cable is NOT a site, but a composite with multiple bodies. The cable is composed of 20 parts (B_first, B_2, ..., B_last) and the body of interest is B_11. To better understand the cable structure, look at the MJMODEL.txt file generated by the model which describes all present bodies in the model.

    * `desired_goal`: this key represents the final goal to be achieved. In this environment it is a 3-dimensional `ndarray`, `(3,)`, that consists of the three cartesian coordinates of the desired final target position `[x,y,z]`. In order for the slider to position the B_11 body of the cable upon the target position. The elements of the array are the following:

    | Num | Observation                                                                                                                           | Min    | Max    | Site Name (in corresponding XML file) |Unit          |
    |-----|---------------------------------------------------------------------------------------------------------------------------------------|--------|--------|---------------------------------------|--------------|
    | 0   | Final target position in the x coordinate                                                                                             | -Inf   | Inf    | target0                               | position (m) |
    | 1   | Final target position in the y coordinate                                                                                             | -Inf   | Inf    | target0                               | position (m) |
    | 2   | Final target position in the z coordinate                                                                                             | -Inf   | Inf    | target0                               | position (m) |

    * `achieved_goal`: this key represents the current state of the B_11 of the cable, as if it would have achieved a goal. This is useful for goal orientated learning algorithms such as those that use [Hindsight Experience Replay](https://arxiv.org/abs/1707.01495) (HER). The value is an `ndarray` with shape `(3,)`. The elements of the array are the following:

    | Num | Observation                                                                                                                           | Min    | Max    | Site Name (in corresponding XML file) |Unit          |
    |-----|---------------------------------------------------------------------------------------------------------------------------------------|--------|--------|---------------------------------------|--------------|
    | 0   | Current cable B_11 position in the x coordinate                                                                                       | -Inf   | Inf    | cable (*) | B_11                      | position (m) |
    | 1   | Current cable B_11 position in the y coordinate                                                                                       | -Inf   | Inf    | cable (*) | B_11                      | position (m) |
    | 2   | Current cable B_11 position in the z coordinate                                                                                       | -Inf   | Inf    | cable (*) | B_11                      | position (m) |


    ## Rewards

    The reward can be initialized as `sparse` or `dense`:
    - *sparse*: the returned reward can have two values: `-1` if the B_11 of the cable hasn't reached its final target position, and `0` if the B_11 of the cable is in the final target position (the B_11 of the cable is considered to have reached the goal if the Euclidean distance between both is lower than 0.05 m).
    - *dense*: the returned reward is the negative Euclidean distance between the achieved goal position and the desired goal.

    To initialize this environment with one of the mentioned reward functions the type of reward must be specified in the id string when the environment is initialized. For `sparse` reward the id is the default of the environment, `ManipulateCableEnv-v0`. However, for `dense` reward the id must be modified to `ManipulateCableEnv-v0` and initialized as follows:

    ```python
    import gymnasium as gym
    import gymnasium_robotics

    from manipulate_cable import MujocoManipulateCableEnv #see the test_man_cable.py file

    env = gym.make('ManipulateCableEnv-v0', reward_type='dense') #if you want to use the dense reward
    ```

    ## Starting State

    When the environment is reset the slider is placed in the following global cartesian coordinates `(x,y,z) = [0.7 0 0.6] m`, and its orientation in quaternions is `(w,x,y,z) = [1.0, 0.0, 0.0, 0.0]`. The joint positions are computed by inverse kinematics internally by MuJoCo. The base of the cable will always be fixed at `(x,y,z) = [-0.3, 0.0, 0.6]` in global coordinates.
    Since the cable is constrained to the slider, the cable configuration is determined by the slider position.
    
    Finally the target position where the slider has to move the B_11 of the cable is generated. The target is at a fixed position of '(x,y,z) = [0.15, 0, 0.35]'. In future development the random target will be generated by adding an offset to the initial target position `(x,z)` sampled from a uniform distribution with a range of `[-0.1, 0.1] m`.


    ## Episode End

    The episode will be `truncated` when the duration reaches a total of `max_episode_steps` which by default is set to 50 timesteps.
    The episode is never `terminated` since the task is continuing with infinite horizon.

    ## Arguments

    To increase/decrease the maximum number of timesteps before the episode is `truncated` the `max_episode_steps` argument can be set at initialization.
    The default value is 50. For example, to increase the total number of timesteps to 100 make the environment as follows:

    ```python
    import gymnasium as gym
    import gymnasium_robotics

    gym.register_envs(gymnasium_robotics)

    env = gym.make('ManipulateCableEnv-v0', max_episode_steps=100)
    ```

    ## Version History

    * v0: First tutorial. The environment depends on the newest [mujoco python bindings](https://mujoco.readthedocs.io/en/latest/python.html) maintained by the MuJoCo team in Deepmind.
    
    """
    def __init__(self, reward_type="sparse", **kwargs):
        initial_qpos = {
            "slider:joint": -0.3,
        }
        MujocoSliderEnv.__init__(
            self,
            model_path=MODEL_XML_PATH,
            n_substeps=20,
            target_offset=0.0,
            sld_range=0.1,
            target_range=0.1,
            distance_threshold=0.05,
            initial_qpos=initial_qpos,
            reward_type=reward_type,
            **kwargs,
        )
        EzPickle.__init__(self, reward_type=reward_type, **kwargs)



register(
    id='ManipulateCableEnv-v0',
    entry_point='manipulate_cable:MujocoManipulateCableEnv',
)
