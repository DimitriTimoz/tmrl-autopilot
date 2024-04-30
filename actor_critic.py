import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical
from ncps.torch import CfC
from ncps.wirings import AutoNCP
import torchvision.models as models
import time

def obs_space_to_tensor_concat(obs):
    # Handle camera data: normalize, convert to float tensor, and flatten
    camera = torch.tensor(obs['camera'], dtype=torch.float32) / 255.0
    camera = camera.permute(2, 0, 1)
    
    # Handle position, velocity, and yaw_pitch_roll: convert to float tensor and flatten
    position = torch.tensor(obs['position'], dtype=torch.float32).view(-1)
    velocity = torch.tensor(obs['velocity'], dtype=torch.float32).view(-1)
    yaw_pitch_roll = torch.tensor(obs['yaw_pitch_roll'], dtype=torch.float32).view(-1)
    
    # Handle MultiBinary data: convert to float tensor and flatten
    wheel_contacts = torch.tensor(obs['wheel_contacts'], dtype=torch.float32).view(-1)
    wheel_sliding = torch.tensor(obs['wheel_sliding'], dtype=torch.float32).view(-1)
    
    # Concatenate all tensors into a single tensor
    concatenated_tensor = torch.cat([
        position,
        velocity,
        yaw_pitch_roll,
        wheel_contacts,
        wheel_sliding
    ], dim=0).cuda()
    
    return concatenated_tensor, camera.cuda()


class Actor(nn.Module):
    def __init__(self, action_space, observation_space):
        super(Actor, self).__init__()
        self.obs_space = observation_space
        self.act_space = action_space
        self.image_length = self.obs_space['camera'].shape[0] * self.obs_space['camera'].shape[1] * self.obs_space['camera'].shape[2]
        print(f"Actor initialized with obs_space: {observation_space}, act_space: {action_space}")

        # Use EfficientNet as a feature extractor, not the full model
        base_model = models.efficientnet_b5(weights="EfficientNet_B5_Weights.DEFAULT")
        self.vision_model = nn.Sequential(*list(base_model.children())[:-2]).cuda()
        print("Vision model loaded as feature extractor")
                
        wiring = AutoNCP(30, 2)
        self.rnn = CfC(460800 + 3 * 3 + 2*4, wiring).cuda()
        self.hx = None
        
    def forward(self, obs, camera, test=False):
        # Process image with the model to get vision embedding
        vision_embedding = self.vision_model(camera)
        
        # Flatten vision embedding
        vision_embedding_flattened = vision_embedding.view(vision_embedding.size(0), -1)

        # Concatenate the vision embedding with the non-image features
        combined_features = torch.cat([obs, vision_embedding_flattened], dim=1)

        # Process the combined features with the RNN
        action, self.hx = self.rnn(combined_features, hx=self.hx if self.hx is not None else None)
        return torch.clamp(action, -1.0, 1.0)
    
    def act(self, obs):
        obs, camera = obs_space_to_tensor_concat(obs)
        print(obs.shape, camera.shape)
        return self.forward(torch.tensor(obs).unsqueeze(0), torch.tensor(camera).unsqueeze(0))[0].cpu().detach().numpy()

class Critic(nn.Module):
    def __init__(self, state_size, action_size):
        super(Critic, self).__init__()
        self.state_size = state_size
        self.action_size = action_size
        self.linear1 = nn.Linear(self.state_size, 128)
        self.linear2 = nn.Linear(128, 256)
        self.linear3 = nn.Linear(256, 1)

    def forward(self, state):
        output = F.relu(self.linear1(state))
        output = F.relu(self.linear2(output))
        value = self.linear3(output)
        return value