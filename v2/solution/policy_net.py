import torch
import torch.nn as nn
import torch.nn.functional as F
from  tensordict import TensorDict

class PolicyNet(nn.Module):
    def __init__(self, input_channels: int, grid_size: int, num_actions: int, state_size: int, belief_size: int, traits_size : int, device="cpu"):
        super(PolicyNet, self).__init__()

        self.grid_size = grid_size

        # Define a simple CNN for the vision input
        self.conv11 = nn.Conv2d(in_channels=input_channels, out_channels=16, kernel_size=3, stride=1, padding=1)
        self.conv12 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=1)

        
        self.conv21 = nn.Conv2d(in_channels=input_channels, out_channels=16, kernel_size=3, stride=1, padding=1)
        self.conv22 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=1)

        # Calculate the size after convolution
        conv_output_size = grid_size * grid_size * 32
        intermediate_embed_size = 5

        # Fully connected layer to reduce convolutional output to size 5
        self.conv_fc1 = nn.Linear(conv_output_size, intermediate_embed_size)
        self.conv_fc2 = nn.Linear(conv_output_size, intermediate_embed_size)

        # Adjust the input size for the first fully connected layer
        total_input_size = 2 * intermediate_embed_size + state_size + belief_size + traits_size

        # Fully connected layers for decision making (Q-values)
        self.fc1 = nn.Linear(total_input_size, 128)
        self.fc2 = nn.Linear(128, num_actions)

        self.device = device

    def to(self, device):
        super().to(device)
        self.device = device

    def forward(self, idx : int, obs : dict[int, list[TensorDict]]) -> torch.Tensor:
        """
        Forward pass for the policy network.

        :param idx: the idx of the agent.
        :param x: dictionary representing the joint observation. Each obs is of the form (batch, obs)

        :return: Q-values for each action
        """
        x = obs[idx]

        return self._forward(x)
    
    def _forward(self, obs: TensorDict):
        # Process the vision tensor
        x = obs["vision"]
        y = obs['terrain']

        # Get the current grid size
        current_size = x.size(-1)

        # Calculate padding required for each side
        if current_size < self.grid_size:
            pad_total = self.grid_size - current_size
            pad_left = pad_total // 2
            pad_right = pad_total - pad_left
            pad_top = pad_total // 2
            pad_bottom = pad_total - pad_top

            # Apply padding
            x = F.pad(x, (pad_left, pad_right, pad_top, pad_bottom), value=-1)
            y = F.pad(y, (pad_left, pad_right, pad_top, pad_bottom), value=-1)

        # Pass through the convolutional layers
        x = F.relu(self.conv11(x))
        x = F.relu(self.conv12(x))

        # Flatten the output from the convolutional layers
        x = x.view(x.size(0), -1)

        # Pass through the additional FC layer to reduce dimensions
        x = self.conv_fc1(x)
        x = F.relu(x)  # Apply activation

        # Pass through the convolutional layers
        y = F.relu(self.conv21(y))
        y = F.relu(self.conv22(y))

        # Flatten the output from the convolutional layers
        y = y.view(y.size(0), -1)

        # Pass through the additional FC layer to reduce dimensions
        y = self.conv_fc2(y)
        y = F.relu(y)  # Apply activation

        # Retrieve state and belief tensors
        state = torch.clone(obs["state"]).detach()
        belief = torch.clone(obs["belief"]).detach()
        traits = torch.clone(obs["traits"]).detach()

        # Concatenate vision, state, and belief tensors
        x = torch.cat((x, y, state, belief, traits), dim=1)

        # Pass through fully connected layers
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)

        return x


import torch
import numpy as np


def feature_extractor(obs : dict, device : str = "cpu") -> TensorDict:
    """
    Extracts features from the environment observations for each agent.
    
    :param obs: Dictionary where keys are agent IDs and values are dictionaries containing the relevant observation data
    :return: A dictionary where keys are agent IDs and values are PyTorch tensors representing the features.
    """
    features_dict = {}


    for agent_id, agent_obs in obs.items():
        # Extract individual components from agent's observation
        vision_grid = agent_obs['Vision']
        terrain_grid = agent_obs['Terrain']
        state = agent_obs['State']
        belief = agent_obs['Belief']
        traits = agent_obs['Traits']

        vision_tensor = torch.from_numpy(vision_grid).float().unsqueeze(0)
        terrain_grid = torch.from_numpy(terrain_grid).float().unsqueeze(0)

        # Convert energy and belief to tensors (they could be single values or vectors)
        state_tensor = state
        belief_tensor = belief

        # Concatenate all feature tensors into a one-dimensional feature vector
        features = TensorDict({
            "vision" : vision_tensor,
            "state": state_tensor,
            "belief" : belief_tensor,
            "terrain": terrain_grid,
            "traits": traits
        }, device = device)

        # Add the concatenated tensor to the features dictionary, keyed by agent ID
        features_dict[agent_id] = features


    return features_dict