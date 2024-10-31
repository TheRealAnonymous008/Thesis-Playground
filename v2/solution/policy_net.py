import torch
import torch.nn as nn
import torch.nn.functional as F

from dataclasses import dataclass

@dataclass 
class FeatureDict:
    vision_tensor : torch.Tensor  = None 
    energy :  torch.Tensor = None 
    belief : torch.Tensor = None 


class PolicyNet(nn.Module):
    def __init__(self, input_channels: int, grid_size: int, num_actions: int):
        """
        Initialize the Policy Network

        :param input_channels: Number of input channels (for vision, it's likely 1)
        :param grid_size: Size of the observation grid (e.g., 5x5 or 7x7)
        :param num_actions: Number of possible actions (Discrete(12) in this case)
        """
        super(PolicyNet, self).__init__()

        # Define a simple CNN for the vision input
        self.conv1 = nn.Conv2d(in_channels=input_channels, out_channels=16, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=1)

        # Calculate the size after convolution for the fully connected layer
        conv_output_size = grid_size * grid_size * 32

        # Fully connected layers for decision making (Q-values)
        self.fc1 = nn.Linear(conv_output_size, 128)
        self.fc2 = nn.Linear(128, num_actions)

    def forward(self, idx : int, obs : dict[int, FeatureDict]) -> torch.Tensor:
        """
        Forward pass for the policy network.

        :param idx: the idx of the agent.
        :param x: dictionary representing the joint observation. Each obs is of the form (batch, obs)

        :return: Q-values for each action
        """
        try: 
            x = obs[idx]
            return self._forward(x)
        except:
            raise Exception(f"There's something wrong here {idx} with data {obs[idx]} of length {len(obs[idx])}")

    def _forward(self, obs : FeatureDict):
        x = obs.vision_tensor

        # Pass through the convolutional layers
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))

        # Flatten the output from conv layers
        x = x.view(x.size(0), -1)

        # Pass through fully connected layers
        x = F.relu(self.fc1(x))
        x = self.fc2(x)

        return x

import torch


def feature_extractor(obs : dict) -> dict:
    """
    Extracts features from the environment observations for each agent.
    
    :param obs: Dictionary where keys are agent IDs and values are dictionaries containing the relevant observation data
    :return: A dictionary where keys are agent IDs and values are PyTorch tensors representing the features.
    """
    features_dict = {}

    for agent_id, agent_obs in obs.items():
        # Extract individual components from agent's observation
        vision_grid = agent_obs['Victims']
        energy = agent_obs['Energy']
        belief = agent_obs['Belief']

        vision_tensor = torch.tensor(vision_grid, dtype=torch.float32).flatten()

        # Convert energy and belief to tensors (they could be single values or vectors)
        energy_tensor = torch.tensor([energy], dtype=torch.float32)
        belief_tensor = torch.tensor(belief, dtype=torch.float32)

        # Concatenate all feature tensors into a one-dimensional feature vector
        features = FeatureDict(
            vision_tensor = vision_tensor, 
            energy = energy_tensor ,
            belief = belief_tensor
        )

        # Add the concatenated tensor to the features dictionary, keyed by agent ID
        features_dict[agent_id] = features


    return features_dict