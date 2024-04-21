import time
from typing import TypeVar

import numpy as np
from gym import Env
from gym.spaces import Box, MultiBinary, Dict

from .TMIClient import ThreadedClient
from .utils.GameCapture import GameViewer

from .utils.GameLaunch import GameLauncher

ControllerActionSpace = Box(
    low=np.array([0.0, -1.0]), high=np.array([1.0, 1.0]), shape=(2,), dtype=np.float32
)  

IMAGE_SIZE = (256, 256, 3)

ObservationSpace = Dict(
    {
        "camera": Box(low=0, high=255, shape=IMAGE_SIZE, dtype=np.uint8),
        "position": Box(low=-10000.0, high=10000.0, shape=(3,), dtype=np.float32),
        "velocity": Box(low=-1000.0, high=1000.0, shape=(3,), dtype=np.float32),
        "yaw_pitch_roll": Box(low=-3.15, high=3.15, shape=(3,), dtype=np.float32),
        "wheel_contacts": MultiBinary(4),
        "wheel_sliding": MultiBinary(4),
        
    }
)
ActType = TypeVar("ActType")
ObsType = TypeVar("ObsType")


class TrackmaniaEnv(Env):
    """
    Gym env interfacing the game.
    Observations are the rays of the game viewer.
    Controls are the arrow keys or the gas and steer.
    """

    def __init__(
        self,
        action_space: str = "arrows",
        n_rays: int = 16,
    ):
        self.action_space = (
           ControllerActionSpace
        )
        self.observation_space = Box(
            low=0.0, high=1.0, shape=(n_rays + 1,), dtype=np.float32
        )

        game_launcher = GameLauncher()
        if not game_launcher.game_started:
            game_launcher.start_game()
            print("game started")
            input("press enter when game is ready")
            time.sleep(4)

        self.viewer = GameViewer()
        self.simthread = ThreadedClient()
        self.total_reward = 0.0
        self.n_steps = 0
        self.max_steps = 1000
        self.command_frequency = 50
        self.last_action = None
        self.low_speed_steps = 0

    def step(self, action):
        self.last_action = action
        # plays action
        self.action_to_command(action)
        done = (
            True
            if self.n_steps >= self.max_steps or self.total_reward < -300
            else False
        )
        self.total_reward += self.reward
        self.n_steps += 1
        info = {}
        time.sleep(self.command_frequency * 10e-3)
        return self.observation, self.reward, done, info

    def reset(self):
        print("reset")
        self.total_reward = 0.0
        self.n_steps = 0
        self._restart_race()
        self.time = 0
        self.last_action = None
        self.low_speed_steps = 0
        print("reset done")

        return self.observation

    def render(self, mode="human"):
        print(f"total reward: {self.total_reward}")
        print(f"speed: {self.speed}")
        print(f"time = {self.state.time}")

    def action_to_command(self, action):
        steer = np.clip(action[1], -1, 1)
        steer = int(65536.0 * steer)
        gas = action[0]
        self.simthread.apply_action(steer=steer, gas=gas)

    @property
    def state(self):
        return self.simthread.data

    @property
    def speed(self):
        return self.state.display_speed

    @property
    def observation(self):
        wheels = self.state.simulation_wheels
        wheels_states = [wheel.real_time_state  for wheel in wheels]
        is_contact = [state.has_ground_contact for state in wheels_states]
        is_sliding = [state.is_sliding for state in wheels_states]

        return {
            "camera": np.array(self.viewer.get_frame()),
            "position": np.array(self.state.position, dtype=np.float32),
            "velocity": np.array(self.state.velocity, dtype=np.float32),
            "yaw_pitch_roll": np.array(self.state.yaw_pitch_roll, dtype=np.float32),
            "wheel_contacts": np.array(is_contact, dtype=np.uint8),
            "wheel_sliding": np.array(is_sliding, dtype=np.uint8),
        }

    @property
    def reward(self):
        speed = self.speed
        if self.state.time < 3000:
            return 0

        speed_reward = speed / 400
        roll_reward = -abs(self.state.yaw_pitch_roll[2]) / 3.15
        constant_reward = -0.3
        gas_reward = self.last_action[0] * 2

        if self.last_action[0] < 0:
            constant_reward -= 10
            gas_reward = 0


        elif 10 < speed < 100:
            speed_reward = -1
            gas_reward = 0

        elif speed < 10:
            self.low_speed_steps += 1
            speed_reward = -5 * self.low_speed_steps
            gas_reward = 0

        else:
            self.low_speed_steps = 0

        return speed_reward + roll_reward + constant_reward + gas_reward
    
    def _restart_race(self):
        self.simthread.restart()
        