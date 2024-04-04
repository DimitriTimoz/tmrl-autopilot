from tmrl.actor import TorchActorModule
import tmrl.config.config_constants as cfg
import torchvision.models as models
import torch
from ncps.torch import CfC
from ncps.wirings import AutoNCP

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
        self.dim_action = 3
        wiring = AutoNCP(30, self.dim_action) # 30 neurons, dim_action outputs
        self.rnn = CfC(self.middle_input_size, wiring)
        self.hx = None



        
    def forward(self, obs, test=False, with_logprob=True):
        # TODO: normalize the observations
        # obs is a list of tuple of Box that are the embeddings of the observations
        # observation space: tensor(Tuple(
            # Box(0.0, 1000.0, (1,), float32), 
            # Box(0.0, 6.0, (1,), float32), 
            # Box(0.0, inf, (1,), float32), 
            # Box(0.0, 255.0, (1, 256, 256, 3), float32), 
            # Box(-1.0, 1.0, (3,), float32), 
            # Box(-1.0, 1.0, (3,), float32)))
        # Split the images of the observations
        images = obs[3].squeeze(1).permute(0, 3, 1, 2)
        images = images.float() / 255.0
        
        # Process image with the model to get vision embedding
        vision_embedding = self.model_ft(images)
        
        # Merge the vision embedding with the rest of the observations
        non_image_obs = [o for i, o in enumerate(obs) if i != 3]

        # Assuming all non-image observations are already tensors, flatten them if necessary
        non_image_obs_flattened = [o.view(o.size(0), -1) for o in non_image_obs]

        # Concatenate the flattened non-image observations along the feature dimension
        non_image_features = torch.cat(non_image_obs_flattened, dim=1)

        # Ensure the vision embedding is also flattened (if not already)
        vision_embedding_flattened = vision_embedding.view(vision_embedding.size(0), -1)

        # Concatenate the vision embedding with the non-image features
        combined_features = torch.cat([non_image_features, vision_embedding_flattened], dim=1)
        combined_features = combined_features.float()
        # Process the combined features with the RNN
        if self.hx is not None:
            self.hx = self.hx.float()

        # TODO: Check batch collission
        action, self.hx = self.rnn(combined_features, hx=self.hx)

        return action
        
    
    def act(self, obs, test=False):
        # 0 Speed (0.0 to 1.0)
        # 1 Backward (0.0 to 1.0)
        # 2 Steering right (-1.0 to 1.0)

        with torch.no_grad():
            a, _ = self.forward(obs, test, False)
            return a.cpu().numpy()
