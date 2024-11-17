import torch
import torch.nn as nn
import torch.nn.functional as F
from  tensordict import TensorDict

class PolicyNet(nn.Module):
    def __init__(self, input_channels: int, grid_size: int, num_actions: int, device = "cpu"):
        """
        Initialize the Policy Network

        :param input_channels: Number of input channels (for vision, it's likely 1)
        :param grid_size: Size of the observation grid (e.g., 5x5 or 7x7)
        :param num_actions: Number of possible actions (Discrete(12) in this case)
        """
        super(PolicyNet, self).__init__()

        self.grid_size = grid_size

        # Define a simple CNN for the vision input
        self.conv1 = nn.Conv2d(in_channels=input_channels, out_channels=16, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=1)

        # Calculate the size after convolution for the fully connected layer
        conv_output_size = grid_size * grid_size * 32

        # Fully connected layers for decision making (Q-values)
        self.fc1 = nn.Linear(conv_output_size, 128)
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

    def _forward(self, obs : TensorDict):
        x = obs["vision"]

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
            x = F.pad(x, (pad_left, pad_right, pad_top, pad_bottom))


        # Pass through the convolutional layers
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))

        # Flatten the output from conv layers
        x = x.view(x.size(0), -1)

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
        state = agent_obs['State']
        belief = agent_obs['Belief']

        vision_tensor = torch.from_numpy(vision_grid).float().unsqueeze(0)

        # Convert energy and belief to tensors (they could be single values or vectors)
        state_tensor = state
        belief_tensor = belief

        # Concatenate all feature tensors into a one-dimensional feature vector
        features = TensorDict({
            "vision" : vision_tensor,
            "state": state_tensor,
            "belief" : belief_tensor
        }, device = device)

        # Add the concatenated tensor to the features dictionary, keyed by agent ID
        features_dict[agent_id] = features


    return features_dict