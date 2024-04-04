
import tmrl.config.config_constants as cfg
import tmrl.config.config_objects as cfg_obj
from tmrl.envs import GenericGymEnv
from tmrl.networking import RolloutWorker
from tmrl.util import partial
from actor import Actor


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
    