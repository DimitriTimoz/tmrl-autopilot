import numpy as np
from tmrl.actor import TorchActorModule
import tmrl.config.config_constants as cfg
import functools
import operator
import torchvision.models as models
import torch
from ncps.torch import CfC
from ncps.wirings import AutoNCP

def prod(iterable):
    return functools.reduce(operator.mul, iterable, 1)

class Actor(TorchActorModule):
    def __init__(self, observation_space, action_space):
        super().__init__(observation_space, action_space)
        self.obs_space = observation_space
        self.act_space = action_space
        self.device = 'cuda' if cfg.CUDA_INFERENCE else 'cpu'
        print(f"Actor initialized with obs_space: {observation_space}, act_space: {action_space}")


        self.model_ft = models.resnet18(weights='IMAGENET1K_V1')
        self.num_ftrs = self.model_ft.fc.in_features
        self.model_ft.fc = torch.nn.Linear(self.num_ftrs, 64)
        self.model_ft = self.model_ft.to(self.device)
        
        self.middle_input_size = 73
        
        wiring = AutoNCP(30, 3) # 30 neurons, 3 outputs
        self.rnn = CfC(self.middle_input_size, wiring)
        self.hx = None


    def forward(self, obs, test=False, with_logprob=True):
        # Ensure obs is a list of tensors and move them to the correct device
        obs = [torch.as_tensor(x, device=self.device) for x in obs]

        # Pop the image tensor from the obs list
        img = obs.pop(3)

        # Process image with the model to get vision embedding
        vision_embedding = self.model_ft(img[0].permute(0, 3, 1, 2))

        # Flatten all observation tensors and the vision embedding
        obs_flattened = [torch.flatten(x) for x in obs]
        vision_embedding_flattened = torch.flatten(vision_embedding)

        all_features = torch.cat([vision_embedding_flattened] + obs_flattened).unsqueeze(0)  


        # Now, all_features has a shape of [1, feature_length], representing a single batch
        action_pi, self.hx = self.rnn(all_features, hx=self.hx)  # Use hx directly without reassigning it to None
        print(action_pi)

        return action_pi[0], ()
        
    def act(self, obs, test=False):
        # 0 Speed (0.0 to 1.0)
        # 1 Backward (0.0 to 1.0)
        # 2 Steering right (-1.0 to 1.0)
        return np.array(self.forward(obs, test, False)[0], dtype=np.float32)
