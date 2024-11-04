import torch
import torch.nn as nn
import torch.nn.functional as F
from tensordict import TensorDict
import numpy as np

from core.agent import * 
from sar.sar_agent import *

INPUT_DIMS = 5

class EncoderDecoder(nn.Module):
    def __init__(self, hidden_dim=64, output_dim=32, device = "cpu"):
        """
        Initialize an Encoder Network. The Encoder Network is used to provide latent space embeddings for each 
        heterogeneous agent based on traits and state information.
        
        :param input_dim: The input dimension, corresponding to the concatenated agent features.
        :param hidden_dim: The hidden dimension size.
        :param output_dim: The size of the latent space embedding.
        """
        super(EncoderDecoder, self).__init__()
        self.fc1 = nn.Linear(INPUT_DIMS, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)


        # self.dfc1 = nn.Linear(output_dim,)

        self.device = device

    def encoder_forward_batch(self, agents: list[SARAgent]) -> torch.Tensor:
        """
        Returns an embedding corresponding to an `agent`.
        
        :param agent: An instance of Agent with `traits` and `state` attributes.
        :return: A latent vector representing the agent's embedding.
        """
        try:
            # Extract traits and state features
            batch_features = []

            for agent in agents: 
                traits = torch.tensor([
                    agent._traits._visibility_range,
                    agent._traits._energy_capacity,
                    agent._traits._max_slope
                ], dtype=torch.float32)
                
                state = torch.tensor([
                    agent._current_state.current_energy,
                    agent._current_state.victims_rescued
                ], dtype=torch.float32)
                
                # Concatenate all features
                features = torch.cat((traits, state), dim=0)
                batch_features.append(features)
            
            batch_features = torch.stack(batch_features)
            batch_features.to(device=self.device)
            
            # Forward pass through the network
            x = F.relu(self.fc1(batch_features))
            embedding = self.fc2(x)
            
            return embedding
        except Exception as e:
            raise Exception(f"There's something wrong here. Input {agent}. Error: {e}")
