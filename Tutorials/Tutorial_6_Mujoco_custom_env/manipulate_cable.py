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
