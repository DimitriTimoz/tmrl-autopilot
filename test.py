import json
import logging
import time
from argparse import ArgumentParser, ArgumentTypeError

# local imports
import tmrl.config.config_constants as cfg
import tmrl.config.config_objects as cfg_obj
from tmrl.envs import GenericGymEnv
from tmrl.networking import Server, Trainer, RolloutWorker
from tmrl.tools.check_environment import check_env_tm20lidar, check_env_tm20full
from tmrl.tools.record import record_reward_dist
from tmrl.util import partial
from tmrl.actor import TorchActorModule
import numpy as np

class Actor(TorchActorModule):
    def __init__(self, observation_space, action_space):
        super().__init__(observation_space, action_space)
        self.obs_space = observation_space
        self.act_space = action_space
        self.device = 'cuda' if cfg.CUDA_INFERENCE else 'cpu'
        print(f"Actor initialized with obs_space: {observation_space}, act_space: {action_space}")

    def act(self, obs, test=False):
        # 0 Speed (0.0 to 1.0)
        # 1 Backward (0.0 to 1.0)
        # 2 Steering right (-1.0 to 1.0)
        # 3 ??
        return np.array([0.0, 1.0, -1.0, 1.0], dtype=np.float32)


def main():
    config = cfg_obj.CONFIG_DICT
    
    rw = RolloutWorker(env_cls=partial(GenericGymEnv, id=cfg.RTGYM_VERSION, gym_kwargs={"config": config}),
                        actor_module_cls=Actor,
                        sample_compressor=cfg_obj.SAMPLE_COMPRESSOR,
                        device='cuda' if cfg.CUDA_INFERENCE else 'cpu',
                        server_ip=cfg.SERVER_IP_FOR_WORKER,
                        max_samples_per_episode=cfg.RW_MAX_SAMPLES_PER_EPISODE,
                        model_path=cfg.MODEL_PATH_WORKER,
                        obs_preprocessor=cfg_obj.OBS_PREPROCESSOR,
                        crc_debug=cfg.CRC_DEBUG,)
    rw.run_episodes(10000)


if __name__ == "__main__":
    main()
    