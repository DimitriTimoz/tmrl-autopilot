import random
import tmrl.config.config_constants as cfg
import tmrl.config.config_objects as cfg_obj

from tmrl.envs import GenericGymEnv
from tmrl.util import partial
from tmrl.training import TrainingAgent
from tmrl.networking import Trainer
from tmrl.training_offline import TorchTrainingOffline
from tmrl.memory import TorchMemory

import numpy as np
import torch

from threading import Thread

from actor import Actor

weights_folder = cfg.WEIGHTS_FOLDER  # path to the weights folder
checkpoints_folder = cfg.CHECKPOINTS_FOLDER
my_run_name = "tutorial"
CRC_DEBUG = False

model_path = str(weights_folder / (my_run_name + "_t.tmod"))
checkpoints_path = str(checkpoints_folder / (my_run_name + "_t.tcpt"))
4
config = cfg_obj.CONFIG_DICT

class MyMemory(TorchMemory):
    def __init__(self,
                 act_buf_len=None,
                 device=None,
                 nb_steps=None,
                 sample_preprocessor: callable = None,
                 memory_size=1000000,
                 batch_size=32,
                 dataset_path=""):

        self.act_buf_len = act_buf_len  # length of the action buffer

        super().__init__(device=device,
                         nb_steps=nb_steps,
                         sample_preprocessor=sample_preprocessor,
                         memory_size=memory_size,
                         batch_size=batch_size,
                         dataset_path=dataset_path,
                         crc_debug=CRC_DEBUG)

    def append_buffer(self, buffer):
        """
        buffer.memory is a list of compressed (act_mod, new_obs_mod, rew_mod, terminated_mod, truncated_mod, info_mod) samples
        """

        # decompose compressed samples into their relevant components:

        list_action = [b[0] for b in buffer.memory]
        list_x_position = [b[1][0] for b in buffer.memory]
        list_y_position = [b[1][1] for b in buffer.memory]
        list_x_target = [b[1][2] for b in buffer.memory]
        list_y_target = [b[1][3] for b in buffer.memory]
        list_reward = [b[2] for b in buffer.memory]
        list_terminated = [b[3] for b in buffer.memory]
        list_truncated = [b[4] for b in buffer.memory]
        list_info = [b[5] for b in buffer.memory]
        list_done = [b[3] or b[4] for b in buffer.memory]

        # append to self.data in some arbitrary way:

        if self.__len__() > 0:
            self.data[0] += list_action
            self.data[1] += list_x_position
            self.data[2] += list_y_position
            self.data[3] += list_x_target
            self.data[4] += list_y_target
            self.data[5] += list_reward
            self.data[6] += list_terminated
            self.data[7] += list_info
            self.data[8] += list_truncated
            self.data[9] += list_done
        else:
            self.data.append(list_action)
            self.data.append(list_x_position)
            self.data.append(list_y_position)
            self.data.append(list_x_target)
            self.data.append(list_y_target)
            self.data.append(list_reward)
            self.data.append(list_terminated)
            self.data.append(list_info)
            self.data.append(list_truncated)
            self.data.append(list_done)

        # trim self.data in some arbitrary way when self.__len__() > self.memory_size:

        to_trim = self.__len__() - self.memory_size
        if to_trim > 0:
            self.data[0] = self.data[0][to_trim:]
            self.data[1] = self.data[1][to_trim:]
            self.data[2] = self.data[2][to_trim:]
            self.data[3] = self.data[3][to_trim:]
            self.data[4] = self.data[4][to_trim:]
            self.data[5] = self.data[5][to_trim:]
            self.data[6] = self.data[6][to_trim:]
            self.data[7] = self.data[7][to_trim:]
            self.data[8] = self.data[8][to_trim:]
            self.data[9] = self.data[9][to_trim:]

    def __len__(self):
        if len(self.data) == 0:
            return 0  # self.data is empty
        result = len(self.data[0]) - self.act_buf_len - 1
        if result < 0:
            return 0  # not enough samples to reconstruct the action buffer
        else:
            return result  # we can reconstruct that many samples

    def get_transition(self, item):
        """
        Args:
            item: int: indice of the transition that the Trainer wants to sample
        Returns:
            full transition: (last_obs, new_act, rew, new_obs, terminated, truncated, info)
        """
        while True:  # this enables modifying item in edge cases

            # if item corresponds to a transition from a terminal state to a reset state
            if self.data[9][item + self.act_buf_len - 1]:
                # this wouldn't make sense in RL, so we replace item by a neighbour transition
                if item == 0:  # if first item of the buffer
                    item += 1
                elif item == self.__len__() - 1:  # if last item of the buffer
                    item -= 1
                elif random.random() < 0.5:  # otherwise, sample randomly
                    item += 1
                else:
                    item -= 1

            idx_last = item + self.act_buf_len - 1  # index of previous observation
            idx_now = item + self.act_buf_len  # index of new observation

            # rebuild the action buffer of both observations:
            actions = self.data[0][item:(item + self.act_buf_len + 1)]
            last_act_buf = actions[:-1]  # action buffer of previous observation
            new_act_buf = actions[1:]  # action buffer of new observation

            # correct the action buffer when it goes over a reset transition:
            # (NB: we have eliminated the case where the transition *is* the reset transition)
            eoe = last_true_in_list(self.data[9][item:(item + self.act_buf_len)])  # the last one is not important
            if eoe is not None:
                # either one or both action buffers are passing over a reset transition
                if eoe < self.act_buf_len - 1:
                    # last_act_buf is concerned
                    if item == 0:
                        # we have a problem: the previous action has been discarded; we cannot recover the buffer
                        # in this edge case, we randomly sample another item
                        item = random.randint(1, self.__len__())
                        continue
                    last_act_buf_eoe = eoe
                    # replace everything before last_act_buf_eoe by the previous action
                    prev_act = self.data[0][item - 1]
                    for idx in range(last_act_buf_eoe + 1):
                        act_tmp = last_act_buf[idx]
                        last_act_buf[idx] = prev_act
                        prev_act = act_tmp
                if eoe > 0:
                    # new_act_buf is concerned
                    new_act_buf_eoe = eoe - 1
                    # replace everything before new_act_buf_eoe by the previous action
                    prev_act = self.data[0][item]
                    for idx in range(new_act_buf_eoe + 1):
                        act_tmp = new_act_buf[idx]
                        new_act_buf[idx] = prev_act
                        prev_act = act_tmp

            # rebuild the previous observation:
            last_obs = (self.data[1][idx_last],  # x position
                        self.data[2][idx_last],  # y position
                        self.data[3][idx_last],  # x target
                        self.data[4][idx_last],  # y target
                        *last_act_buf)  # action buffer

            # rebuild the new observation:
            new_obs = (self.data[1][idx_now],  # x position
                       self.data[2][idx_now],  # y position
                       self.data[3][idx_now],  # x target
                       self.data[4][idx_now],  # y target
                       *new_act_buf)  # action buffer

            # other components of the transition:
            new_act = self.data[0][idx_now]  # action
            rew = np.float32(self.data[5][idx_now])  # reward
            terminated = self.data[6][idx_now]  # terminated signal
            truncated = self.data[8][idx_now]  # truncated signal
            info = self.data[7][idx_now]  # info dictionary

            break

        return last_obs, new_act, rew, new_obs, terminated, truncated, info


memory_cls = partial(MyMemory,
                     act_buf_len=config["act_buf_len"])


class MyCriticModule(torch.nn.Module):
    def __init__(self, observation_space, action_space, activation=torch.nn.ReLU):
        super().__init__()

    def forward(self, obs, act):
        return torch.zeros(3)  # dummy output


class MyActorCriticModule(torch.nn.Module):
    def __init__(self, observation_space, action_space, activation=torch.nn.ReLU):
        super().__init__()
        self.actor = Actor(observation_space, action_space)  
        self.q1 = MyCriticModule(observation_space, action_space, activation)  # Q network 1
        self.q2 = MyCriticModule(observation_space, action_space, activation)  # Q network 2


class MyTrainingAgent(TrainingAgent):

    def __init__(self,
                 observation_space=None,
                 device='cuda' if cfg.CUDA_TRAINING else 'cpu',
                 model_cls=MyActorCriticModule,
                 action_space=None):
        super().__init__(observation_space=observation_space,
                         action_space=action_space,
                         device=device)
        self.model = model_cls(observation_space, action_space).to(device)
    def get_actor(self):
        return self.model_nograd.actor

    def train(self, batch):
        ret_dict = dict(
            loss_actor=loss_pi.detach().item(),
            loss_critic=loss_q.detach().item(),
        )
        return ret_dict


training_agent_cls = partial(MyTrainingAgent,
                             model_cls=MyActorCriticModule,
                             device='cuda' if cfg.CUDA_TRAINING else 'cpu')

server_ip = "127.0.0.1"
server_port = 55555
password = cfg.PASSWORD

epochs = 10  # maximum number of epochs, usually set this to np.inf
rounds = 10  # number of rounds per epoch
steps = 1000  # number of training steps per round
update_buffer_interval = 100
update_model_interval = 100
max_training_steps_per_env_step = 2.0
start_training = 400
device = None

env_cls = partial(GenericGymEnv, id="real-time-gym-ts-v1", gym_kwargs={"config": config})

training_cls = partial(
    TorchTrainingOffline,
    env_cls=env_cls,
    memory_cls=memory_cls,
    training_agent_cls=training_agent_cls,
    epochs=10,
    rounds=rounds,
    steps=steps,
    update_buffer_interval=update_buffer_interval,
    update_model_interval=update_model_interval,
    max_training_steps_per_env_step=max_training_steps_per_env_step,
    start_training=start_training,
    device=device)

if __name__ == "__main__":
    my_trainer = Trainer(
        training_cls=training_agent_cls,
        server_ip=server_ip,
        server_port=server_port,
        password=password,
        model_path=model_path,
        checkpoint_path=checkpoints_path)  
    
def run_worker(worker):
    worker.run(test_episode_interval=10)

def run_trainer(trainer):
    trainer.run()
    
if __name__ == "__main__":
    run_trainer(my_trainer)
    