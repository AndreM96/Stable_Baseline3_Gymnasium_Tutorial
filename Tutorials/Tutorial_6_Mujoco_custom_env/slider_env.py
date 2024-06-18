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
                if d < self.distance_threshold:
                    return 1.0
                else:
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
            #pos_ctrl[0] *= 0.5  # limit maximum change in position
            rot_ctrl = [
                1.0, #w
                0.0, #x
                0.0, #y
                0.0, #z
            ]  # fixed rotation of the end effector, expressed as a quaternion
            action = np.concatenate([pos_ctrl, rot_ctrl])

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


class MujocoSliderEnv(get_base_slider_env(MujocoRobotEnv)):
    def __init__(self, default_camera_config: dict = DEFAULT_CAMERA_CONFIG, **kwargs):
        super().__init__(default_camera_config=default_camera_config, **kwargs)

    def _step_callback(self):
        pass

    def _mocap_aid(self, action):
        # Controller which realizes gravity compensation
        #action = super()._set_action(action)
        #self.q_ref = self.data.qpos[2] + action[2]
        self.q_ref += action[0]
        print("q_ref = ", self.q_ref)
        # if np.abs(action[2]) < 1e-6:
        #     self.q_ref = self.q_ref_prev
        # else:
        #     self.q_ref = self.data.qpos[2]
        #     self.q_ref_prev = self.q_re
        Kp = 13035
        self.data.ctrl[0] = -Kp * (self.data.qpos[0] - self.q_ref) + self.data.qfrc_bias[0] 


    def _set_action(self, action):
        action = super()._set_action(action)

        # Apply action to simulation.
        self._mocap_aid(action)
        #self._utils.ctrl_set_action(self.model, self.data, action)
        self._utils.mocap_set_action(self.model, self.data, action)

    def generate_mujoco_observations(self):
        # positions
        slider_pos = self._utils.get_site_xpos(self.model, self.data,"slider:site")
        
        cable_pos = self._get_body_xpos("B_11")
        dt = self.n_substeps * self.model.opt.timestep
        slider_vel = self._utils.get_site_xvelp(self.model, self.data,"slider:site") * dt
        site_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE,"slider:site")
        #print("xvelp = ", self._utils.get_site_xvelp(self.model, self.data,"slider:site"))
        
        print("slider_pos = ", slider_pos)

        return (
            slider_pos,
            cable_pos,
            slider_vel,
        )

    def _get_body_xpos(self,body_name):
        body_id = self._model_names.body_name2id[body_name]
        return self.data.xpos[body_id]

    def _render_callback(self):
        # Visualize target.
        sites_offset = (self.data.site_xpos - self.model.site_pos).copy()
        site_id = self._mujoco.mj_name2id(
            self.model, self._mujoco.mjtObj.mjOBJ_SITE, "target0"
        )
        #self.model.site_pos[site_id] = self.goal - sites_offset[0]
        self._mujoco.mj_forward(self.model, self.data)

    def _reset_sim(self):
        self.data.time = self.initial_time
        self.data.qpos[:] = np.copy(self.initial_qpos)
        self.data.qvel[:] = np.copy(self.initial_qvel)
        if self.model.na != 0:
            self.data.act[:] = None

        # Randomize start position of the slider.
        
        
        slider_qpos = self._utils.get_joint_qpos(
                self.model, self.data,"slider:joint")
        self.q_ref = slider_qpos
        print("q_ref reset = ", self.q_ref)
        
        assert slider_qpos.shape == (1,)
        
        # slider_qpos = slider_qpos + self.np_random.uniform(
        #             -0.3, 0.1, size=1
        #         ) #-0.3 and 0.1 are chosen because 0.69 is the x position of the slider in the xml file, the the initial q_pos is set to -0.1 so the initial position in the simulation of the slider is of 0.59.
        # When startin a new simulation the slider_qpos value will be added to this value (0.59 + slider_qpos) this means that the slider will start at a point in a range of 0.29 and 0.69. In this way the cable will not start totally streched.  
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
        slider_target =  self._utils.get_site_xpos(self.model, self.data, "slider:site")
        slider_rotation = np.array([1.0, 0.0, 0.0, 0.0])
        
        self._utils.set_mocap_pos(self.model, self.data,"slider:mocap", slider_target)
        self._utils.set_mocap_quat(
            self.model, self.data,"slider:mocap", slider_rotation)

        for _ in range(10):
            self._mujoco.mj_step(self.model, self.data, nstep=self.n_substeps)

        # Extract information for sampling goals.
        self.initial_slider_xpos = self._utils.get_site_xpos(
            self.model, self.data,"slider:site").copy()
        print("initial_slider_xpos = ", self.initial_slider_xpos)
        self.initial_target_xpos = self._utils.get_site_xpos(
            self.model, self.data,"target0").copy()
        #print("initial_target_xpos = ", self.initial_target_xpos)






#Left Just for completion of the tutorial but not used.
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
        lookat = self._get_body_xpos("slider")
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
        