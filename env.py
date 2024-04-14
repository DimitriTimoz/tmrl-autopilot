from typing import Tuple
import gym
from gym import spaces
import numpy as np
import sys
from tminterface.client import Client, run_client
from tminterface.interface import TMInterface

class Environement(gym.tmrl_autopilot.env):
    def __init__(self) -> None:
        self.observation_space = spaces.Dict(
            {
                "position": spaces.Box(low=-1000, high=1000, shape=(3,), dtype=np.float32),
                "velocity": spaces.Box(low=-1000, high=1000, shape=(3,), dtype=np.float32),
                "yaw_pitch_roll": spaces.Box(low=-np.pi, high=np.pi, shape=(3,), dtype=np.float32),
                "wheel_contacts": spaces.MultiBinary(4),
                "checkpoint": spaces.Box(low=-1000, high=1000, shape=(3,), dtype=np.float32),
                "checkpoint_count": spaces.Discrete(10),
                "wheel_sliding": spaces.MultiBinary(4),
            }
        )
        
        self.action_space = spaces.Box(low=-1, high=1, shape=(3,), dtype=np.float32)

    def reset(self, *, seed: int | None = None, options: dict | None = None) -> Tuple[Any | dict]:
        return super().reset(seed=seed, options=options)    

    def step(self, action: Any) -> Tuple[Any, float, bool, bool, dict]:
        pass
    