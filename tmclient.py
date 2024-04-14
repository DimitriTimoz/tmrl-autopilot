import sys
from pathlib import Path

import numpy as np
from tminterface.client import Client, run_client
from tminterface.interface import TMInterface
import matplotlib.pyplot as plt
from multiprocessing import Process


class TMClient(Client):
    def __init__(self) -> None:
        super(TMClient, self).__init__()
        self.race_finished = False
        self.period_save_pos_ms = 50
        self.recording = False
        self.reset_data()
        self.data = {}
        self.action_buffer = []

    def reset_data(self):
        self.data = {
            "positions": np.zeros((3)),
            "velocity": np.zeros((3)),
            "checkpoints": 0,
            "yaw_pitch_roll": np.zeros((3)),
            "wheel_contacts": np.zeros((4)),
            "wheel_sliding": np.zeros((4))
        }

    def on_registered(self, iface: TMInterface) -> None:
        print(f"Registered to {iface.server_name}")


    def on_run_step(self, iface: TMInterface, _time: int):
        state = iface.get_simulation_state()

        # Apply action
        if len(self.action_buffer) > 0:
            gas, steer = self.action_buffer.pop(0)
            iface.set_input_state(gas=gas, steer=steer)

        if _time == 0:
            self.recording = True
            self.reset_data()

        if _time >= 0 and _time % self.period_save_pos_ms == 0 and self.recording:            
            velocity = np.array(state.velocity)
            rotation_matrix = np.array(state.rotation_matrix).reshape(3, 3)
            relative_velocity = np.dot(velocity.T, rotation_matrix)
            if not self.race_finished:
                self.data["positions"] = np.array(state.position)
                self.data["velocity"] = np.array(relative_velocity)
                self.data["yaw_pitch_roll"] = np.array(state.yaw_pitch_roll)
                wheels = state.simulation_wheels
                wheels_states = [wheel.real_time_state  for wheel in wheels]
                is_contact = [state.has_ground_contact for state in wheels_states]
                is_sliding = [state.is_sliding for state in wheels_states]
                self.data["wheel_contacts"] = is_contact
                self.data["wheel_sliding"] = is_sliding
            #print(self.data)
                
    def apply_action(self, gas: float, steer: float):
        # Convert action to input state
        gas = np.clip(gas, -1, 1)
        steer = np.clip(steer, -1, 1)
        gas = int(65536.0 * gas)
        steer = int(65536.0 * steer)
        self.action_buffer.append((gas, steer))
        
    def get_state(self):
        return self.data
   
    def on_simulation_end(self, iface, result: int):
        self.recording = False
    
    def on_checkpoint_count_changed(self, iface, current: int, target: int):
        """
        Called when the current checkpoint count changed (a new checkpoint has been passed by the vehicle).
        The `current` and `target` parameters account for the total amount of checkpoints to be collected,
        taking lap count into consideration.

        Args:
            iface (TMInterface): the TMInterface object
            current (int): the current amount of checkpoints passed
            target (int): the total amount of checkpoints on the map (including finish)
        """
        state = iface.get_simulation_state()
        self.data["checkpoints_position"] = (state.position)
        self.data["checkpoints"] = (current)
        if current == target:
            self.race_finished = True
            self.recording = False
        
