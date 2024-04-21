from typing import Union

import numpy as np

#from gymnasium_robotics.envs.robot_env import MujocoPyRobotEnv, MujocoRobotEnv
from robot_env import MujocoPyRobotEnv, MujocoRobotEnv
from gymnasium_robotics.utils import rotations
import mujoco

DEFAULT_CAMERA_CONFIG = {
    "distance": 2.5,
    "azimuth": 132.0,
    "elevation": -14.0,
    "lookat": np.array([1.3, 0.75, 0.55]),
}


def goal_distance(goal_a, goal_b):
    assert goal_a.shape == goal_b.shape
    return np.linalg.norm(goal_a - goal_b, axis=-1)


def get_base_slider_env(RobotEnvClass: Union[MujocoPyRobotEnv, MujocoRobotEnv]):
    """Factory function that returns a BaseSliderEnv class that inherits
    from MujocoPyRobotEnv or MujocoRobotEnv depending on the mujoco python bindings.
    """

    class BaseSliderEnv(RobotEnvClass):
        """Superclass for all Fetch environments."""

        def __init__(
            self,
            target_offset,
            target_range,
            distance_threshold,
            reward_type,
            sld_range,
            **kwargs
        ):
            """Initializes a new Fetch environment.

            Args:
                model_path (string): path to the environments XML file
                n_substeps (int): number of substeps the simulation runs on every call to step
                gripper_extra_height (float): additional height above the table when positioning the gripper
                block_gripper (boolean): whether or not the gripper is blocked (i.e. not movable) or not
                has_object (boolean): whether or not the environment has an object
                target_in_the_air (boolean): whether or not the target should be in the air above the table or on the table surface
                target_offset (float or array with 3 elements): offset of the target
                obj_range (float): range of a uniform distribution for sampling initial object positions
                target_range (float): range of a uniform distribution for sampling a target
                distance_threshold (float): the threshold after which a goal is considered achieved
                initial_qpos (dict): a dictionary of joint names and values that define the initial configuration
                reward_type ('sparse' or 'dense'): the reward type, i.e. sparse or dense
            """

            self.sld_range = sld_range
            self.target_offset = target_offset
            self.target_range = target_range
            self.distance_threshold = distance_threshold
            self.reward_type = reward_type
            
            super().__init__(n_actions=1, **kwargs)


        # GoalEnv methods
        # ----------------------------

        def compute_reward(self, achieved_goal, goal, info):
            # Compute distance between goal and the achieved goal.
            d = goal_distance(achieved_goal, goal)
            if self.reward_type == "sparse":
                return -(d > self.distance_threshold).astype(np.float32)
            else:
                return -d

        # RobotEnv methods
        # ----------------------------

        def _set_action(self, action):
            assert action.shape == (1,)
            action = (
                action.copy()
            )  # ensure that we don't change the action outside of this scope
            #pos_ctrl = self.initial_slider_xpos + action[:]
            pos_ctrl = action[:]
            y_z_coord = np.array([0.0, 0.6])
            pos_ctrl = np.concatenate((pos_ctrl, y_z_coord))
            pos_ctrl[0] *= 0.5  # limit maximum change in position
            rot_ctrl = [
                0.0,
                0.0,
                0.0,
                1.0,
            ]  # fixed rotation of the end effector, expressed as a quaternion
            garbage = np.zeros(2)
            action = np.concatenate([pos_ctrl, rot_ctrl, garbage])

            return action

        def _get_obs(self):
            (
                slider_pos,
                cable_pos,
                slider_vel,
            ) = self.generate_mujoco_observations()

            achieved_goal = np.squeeze(cable_pos.copy())

            obs = np.concatenate(
                [
                    slider_pos,
                    cable_pos.ravel(),
                    slider_vel,
                ]
            )

            return {
                "observation": obs.copy(),
                "achieved_goal": achieved_goal.copy(),
                "desired_goal": self.goal.copy(),
            }

        def generate_mujoco_observations(self):

            raise NotImplementedError

        def _get_gripper_xpos(self):

            raise NotImplementedError

        def _sample_goal(self):
            
            goal = self.initial_target_xpos[:3] #+ self.np_random.uniform(
                #-self.target_range, self.target_range, size=3
            #)
            return goal.copy()

        def _is_success(self, achieved_goal, desired_goal):
            d = goal_distance(achieved_goal, desired_goal)
            return (d < self.distance_threshold).astype(np.float32)

    return BaseSliderEnv


class MujocoPySliderEnv(get_base_slider_env(MujocoPyRobotEnv)):
    def _step_callback(self):
        pass

    def _set_action(self, action):
        action = super()._set_action(action)
        
        # Apply action to simulation.
        self._utils.ctrl_set_action(self.sim, action)
        self._utils.mocap_set_action(self.sim, action)

    def generate_mujoco_observations(self):
        # positions
        slider_pos = self.sim.data.get_site_xpos("slider:site")
        #cable_pos = self.sim.data.get_joint_qpos("target0") #before B_11
        cable_pos = self._get_body_xpos("B11")
        dt = self.sim.nsubsteps * self.sim.model.opt.timestep
        slider_vel = self.sim.data.get_site_xvelp("slider:site") * dt
        

        return (
            slider_pos,
            cable_pos,
            slider_vel,
        )

    def _get_slider_xpos(self):
        body_id = self.sim.model.body_name2id("slider")
        return self.sim.data.body_xpos[body_id]
    
    def _get_body_xpos(self,body_name):
        body_id = self.sim.model.body_name2id(body_name)
        return self.sim.data.body_xpos[body_id]

    def _render_callback(self):
        # Visualize target.
        sites_offset = (self.sim.data.site_xpos - self.sim.model.site_pos).copy()
        site_id = self.sim.model.site_name2id("target0")
        self.sim.model.site_pos[site_id] = self.goal - sites_offset[0]
        self.sim.forward()

    def _viewer_setup(self):
        lookat = self._get_slider_xpos()
        for idx, value in enumerate(lookat):
            self.viewer.cam.lookat[idx] = value
        assert self.viewer is not None
        for key, value in DEFAULT_CAMERA_CONFIG.items():
            if isinstance(value, np.ndarray):
                getattr(self.viewer.cam, key)[:] = value
            else:
                setattr(self.viewer.cam, key, value)

    def _reset_sim(self):
        self.sim.set_state(self.initial_state)

        # Randomize start position of the slider.
        # slider_xpos = self.initial_slider_xpos[0] + self.np_random.uniform(
        #             -self.sld_range, self.sld_range, size=1
        #         )
        slider_xpos = 0.7 + self.np_random.uniform(
                    -self.sld_range, self.sld_range, size=1
                )
        slider_qpos = self.sim.data.get_joint_qpos("slider:joint")
        assert slider_qpos.shape == (1,)
        slider_qpos = slider_xpos
        self.sim.data.set_joint_qpos("slider:joint", slider_qpos)
        
        self.sim.forward()
        return True

    def _env_setup(self, initial_qpos):
        for name, value in initial_qpos.items():
            self.sim.data.set_joint_qpos(name, value)
        self._utils.reset_mocap_welds(self.sim)
        self.sim.forward()

        # Move end effector into position.
        slider_target = np.array(
            [0, 0, 0 ]
        ) + self.sim.data.get_site_xpos("slider:site")
        slider_rotation = np.array([1.0, 0.0, 0.0, 0.0])
        self.sim.data.set_mocap_pos("slider:mocap", slider_target)
        self.sim.data.set_mocap_quat("slider:mocap", slider_rotation)
        for _ in range(10):
            self.sim.step()

        # Extract information for sampling goals.
        self.initial_slider_xpos = self.sim.data.get_site_xpos("slider:site").copy()
        self.initial_target_xpos = self.sim.data.get_site_xpos("target0").copy()
        

#print(help(MujocoPyRobotEnv))
class MujocoSliderEnv(get_base_slider_env(MujocoRobotEnv)):
    def __init__(self, default_camera_config: dict = DEFAULT_CAMERA_CONFIG, **kwargs):
        super().__init__(default_camera_config=default_camera_config, **kwargs)

    def _step_callback(self):
        pass

    def _set_action(self, action):
        action = super()._set_action(action)

        # Apply action to simulation.
        self._utils.ctrl_set_action(self.model, self.data, action)
        self._utils.mocap_set_action(self.model, self.data, action)

    def generate_mujoco_observations(self):
        # positions
        slider_pos = self._utils.get_site_xpos(self.model, self.data,"slider:site")
        #cable_pos = self._utils.get_joint_qpos(self.model, self.data,"target0")
        cable_pos = self._get_body_xpos("B_11")
        dt = self.n_substeps * self.model.opt.timestep
        slider_vel = self._utils.get_site_xvelp(self.model, self.data,"slider:site") * dt
        site_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE,"slider:site")
        #print("site_id", site_id)
        #print("xvelp = ", self._utils.get_site_xvelp(self.model, self.data,"slider:site"))
        

        return (
            slider_pos,
            cable_pos,
            slider_vel,
        )

    def _get_slider_xpos(self):
        body_id = self._model_names.body_name2id["slider"]
        return self.data.xpos[body_id]
    
    def _get_body_xpos(self,body_name):
        body_id = self._model_names.body_name2id[body_name]
        return self.data.xpos[body_id]

    def _render_callback(self):
        # Visualize target.
        sites_offset = (self.data.site_xpos - self.model.site_pos).copy()
        site_id = self._mujoco.mj_name2id(
            self.model, self._mujoco.mjtObj.mjOBJ_SITE, "target0"
        )
        self.model.site_pos[site_id] = self.goal - sites_offset[0]
        self._mujoco.mj_forward(self.model, self.data)

    def _reset_sim(self):
        self.data.time = self.initial_time
        self.data.qpos[:] = np.copy(self.initial_qpos)
        self.data.qvel[:] = np.copy(self.initial_qvel)
        if self.model.na != 0:
            self.data.act[:] = None

        # Randomize start position of the slider.
        slider_xpos = 0.7 + self.np_random.uniform(
                    -self.sld_range, self.sld_range, size=1
                )
        slider_qpos = self._utils.get_joint_qpos(
                self.model, self.data,"slider:joint")
        assert slider_qpos.shape == (1,)
        slider_qpos[:1] = slider_xpos
        self._utils.set_joint_qpos(
                self.model, self.data,"slider:joint", slider_qpos)
        
        self._mujoco.mj_forward(self.model, self.data)
        return True

    def _env_setup(self, initial_qpos):
        for name, value in initial_qpos.items():
            self._utils.set_joint_qpos(self.model, self.data, name, value)
        self._utils.reset_mocap_welds(self.model, self.data)
        self._mujoco.mj_forward(self.model, self.data)

        # Move end effector into position.
        slider_target = np.array(
            [0, 0, 0 ]
        ) + self._utils.get_site_xpos(self.model, self.data, "slider:site")
        slider_rotation = np.array([1.0, 0.0, 0.0, 0.0])
        
        self._utils.set_mocap_pos(self.model, self.data,"slider:mocap", slider_target)
        self._utils.set_mocap_quat(
            self.model, self.data,"slider:mocap", slider_rotation)

        for _ in range(10):
            self._mujoco.mj_step(self.model, self.data, nstep=self.n_substeps)

        # Extract information for sampling goals.
        self.initial_slider_xpos = self._utils.get_site_xpos(
            self.model, self.data,"slider:site").copy()
        self.initial_target_xpos = self._utils.get_site_xpos(
            self.model, self.data,"target0").copy()
