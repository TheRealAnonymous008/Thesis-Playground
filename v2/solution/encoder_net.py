import torch
import torch.nn as nn
import torch.nn.functional as F

from core.agent import * 
from sar.sar_agent import *


class Encoder(nn.Module):
    def __init__(self, state_dims, trait_dims , hidden_dim=64, output_dim=32, device = "cpu"):
        """
        Initialize an Encoder Network. The Encoder Network is used to provide latent space embeddings for each 
        heterogeneous agent based on traits and state information.
        
        :param input_dim: The input dimension, corresponding to the concatenated agent features.
        :param hidden_dim: The hidden dimension size.
        :param output_dim: The size of the latent space embedding.
        """
        super(Encoder, self).__init__()
        self.fc1 = nn.Linear(state_dims + trait_dims, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)


        # self.dfc1 = nn.Linear(output_dim,)

        self.device = device

    def to(self, device):
        super().to(device)
        self.device = device

    def encoder_forward_batch(self, agents: list[SARAgent]) -> torch.Tensor:
        """
        Returns an embedding corresponding to an `agent`.
        
        :param agent: An instance of Agent with `traits` and `state` attributes.
        :return: A latent vector representing the agent's embedding.
        """
        # Extract traits and state features
        batch_features = []

        for agent in agents: 
            traits = agent.trait_as_tensor            
            state = agent.state_as_tensor
            
            # Concatenate all features
            features = torch.cat((traits, state), dim=0)
            batch_features.append(features)

        # Need to cast to torch.dtype = float32     
        batch_features = torch.stack(batch_features)

        # Forward pass through the network 
        x = batch_features
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        
        return x