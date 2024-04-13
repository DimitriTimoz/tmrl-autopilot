import sys
from pathlib import Path

import numpy as np
from tminterface.client import Client, run_client
from tminterface.interface import TMInterface
import matplotlib.pyplot as plt


Path("maps").mkdir(exist_ok=True)

class MainClient(Client):
    def __init__(self) -> None:
        super(MainClient, self).__init__()
        self.race_finished = False
        self.raw_position_list = []
        self.period_save_pos_ms = 50
        self.recording = False
        self.reset_data()

    def reset_data(self):
        self.data = {}
        self.data["positions"] = []
        self.data["velocity"] = []
        self.data["checkpoints"] = []
        self.data["checkpoints_position"] = []
        self.data["yaw_pitch_roll"] = []
        self.data["wheel_contacts"] = []
        self.data["wheel_sliding"] = []
        

    def on_registered(self, iface: TMInterface) -> None:
        print(f"Registered to {iface.server_name}")


    def on_run_step(self, iface: TMInterface, _time: int):
        state = iface.get_simulation_state()

        if _time == 0:
            self.recording = True
            self.reset_data()

        if _time >= 0 and _time % self.period_save_pos_ms == 0:
            if not self.race_finished:
                velocity = np.array(state.velocity, dtype=np.float32)
                rotation_matrix = np.array(state.rotation_matrix).reshape(3, 3)
                relative_velocity = np.dot(velocity.T, rotation_matrix)

                print(f'\x1b[1K\r time {_time} Velocity: {relative_velocity[0]:>8.1f}, {relative_velocity[1]:>8.1f}, {relative_velocity[2]:>8.1f}', flush=True)

                self.data["positions"].append(np.array(state.position, dtype=np.float32))
                self.data["velocity"].append(relative_velocity)
                self.data["yaw_pitch_roll"].append(np.array(state.yaw_pitch_roll, dtype=np.float32))
                wheels = state.simulation_wheels
                wheels_states = [wheel.real_time_state  for wheel in wheels]
                is_contact = [state.has_ground_contact for state in wheels_states]
                is_sliding = [state.is_sliding for state in wheels_states]
                self.data["wheel_contacts"].append(is_contact)
                self.data["wheel_sliding"].append(is_sliding)
                print(f'\x1b[1K\rWheel contacts: {is_contact}', flush=True)
            # print(
            #     f'Time: {_time}\n'
            #     f'Display Speed: {state.display_speed}\n'
            #     f'Position: {state.position}\n'
            #     f'Velocity: {state.velocity}\n'
            #     f'YPW: {state.yaw_pitch_roll}\n'
            # )
    def on_simulation_end(self, iface, result: int):
        self.recording = False
        self.save_stats()
    
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
        self.data["checkpoints_position"].append(state.position)
        self.data["checkpoints"].append(current)
        if current == target:
            self.race_finished = True
            self.recording = False
            self.save_stats()


    def plot_stats(self):
        velocity = np.array(self.data["velocity"], dtype=np.float32)
        print(velocity.shape)
        velocity_norm = np.linalg.norm(velocity, axis=1)
        
        acceleration = np.diff(velocity, axis=0)
        acceleration_norm = np.linalg.norm(acceleration, axis=1)
        
        figure_1, ax1 = plt.subplots()

        ax1.scatter(np.array(self.data["positions"])[:, 0], np.array(self.data["positions"])[:, 2], c=velocity_norm)
        ax1.set_xlabel("X")  # Corrected from ax1.xlabel("X")
        ax1.set_ylabel("Z")  # Corrected from ax1.ylabel("Z")
        
        # Get checkpoint positions
        checkpoints = np.array(self.data["checkpoints_position"])
        ax1.scatter(checkpoints[:, 0], checkpoints[:, 2], c="red", s=100)
        # Annotate checkpoints
        for i, txt in enumerate(self.data["checkpoints"]):
            ax1.annotate(txt, (checkpoints[i, 0], checkpoints[i, 2]))
        cbar = ax1.figure.colorbar(ax1.collections[0], ax=ax1)
        cbar.set_label('Velocity')
        
        # According to the time statistics
        figure_2, ax2 = plt.subplots(nrows=6)
        ax2[0].plot(velocity, label=["X", "Y", "Z"])
        ax2[0].set_xlabel("Time")
        ax2[0].set_ylabel("Velocity")
        
        ax2[0].legend()
        ax2[1].plot(velocity_norm, label="Velocity Norm")
        ax2[1].set_xlabel("Time")
        ax2[1].set_ylabel("Velocity Norm")
        ax2[1].legend()
        
        ax2[2].plot(acceleration, label=["X", "Y", "Z"])
        ax2[2].set_xlabel("Time")
        ax2[2].set_ylabel("Acceleration")
        ax2[2].legend()
        
        ax2[3].plot(acceleration_norm, label="Acceleration Norm")
        ax2[3].set_xlabel("Time")
        ax2[3].set_ylabel("Acceleration Norm")
        ax2[3].legend()

        ax2[4].plot(np.array(self.data["yaw_pitch_roll"]))
        ax2[4].set_xlabel("Time")
        ax2[4].set_ylabel("Yaw Pitch Roll")
        ax2[4].legend(["Yaw", "Pitch", "Roll"])
        wheel_contacts = np.array(self.data["wheel_contacts"])
        print(wheel_contacts.shape)
        ax2[5].plot(wheel_contacts, label=["Wheel 1", "Wheel 2", "Wheel 3", "Wheel 4"])
        ax2[5].set_xlabel("Time")
        ax2[5].set_ylabel("Wheel contacts")
        ax2[5].legend()
        plt.show()

    def save_stats(self):
        self.plot_stats()
        np.save(base_dir / "maps" / "positions.npy", np.array(self.data["positions"]))
        np.save(base_dir / "maps" / "velocity.npy", np.array(self.data["velocity"]))
        
         
        

base_dir = Path(__file__).resolve().parents[1]
server_name = f"TMInterface{sys.argv[1]}" if len(sys.argv) > 1 else "TMInterface0"
print(f"Connecting to {server_name}...")
client = MainClient()
run_client(client, server_name)
# %%
