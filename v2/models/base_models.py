import torch
import torch.nn as nn
import torch.nn.functional as F 
from core.agent import * 
from core.message import *


class BaseDecoder(nn.Module):
    def __init__(self,  belief_dims = 5, packet_dims = 5, input_dims = 32, hidden_dim=32, grid_size = 5, device = "cpu"):
        """
        Initialize a Decoder Network. The Decoder Network updates an agent's belief based on incoming messages.
        
        :param packet_dim: The input dimension of the message packet data.
        :param hidden_dim: Hidden layer dimension.
        :param belief_dim: Output dimension representing the updated agent belief.
        """
        super(BaseDecoder, self).__init__()
        self.grid_size = grid_size
        self.device = device

    def to(self, device):
        super().to(device)
        self.device = device

    def decoder_forward(self, belief: torch.Tensor, packet: object, sender_embedding: torch.Tensor) -> torch.Tensor:
        """
        Updates the agent's belief based on the incoming message and sender embedding.

        :param belief: Previous belief tensor of shape [belief_dims].
        :param packet: An instance of SARMessagePacket containing the message data.
        :param sender_embedding: Sender's embedding tensor of shape [input_dims].
        :return: Updated belief tensor of shape [belief_dims].
        """
        # Process the vision tensor
        pass 

from core.agent import * 


class BaseEncoder(nn.Module):
    def __init__(self, state_dims, trait_dims , hidden_dim=64, output_dim=32, device = "cpu"):
        """
        Initialize an Encoder Network. The Encoder Network is used to provide latent space embeddings for each 
        heterogeneous agent based on traits and state information.
        
        :param input_dim: The input dimension, corresponding to the concatenated agent features.
        :param hidden_dim: The hidden dimension size.
        :param output_dim: The size of the latent space embedding.
        """
        super(BaseEncoder, self).__init__()
        self.device = device

    def to(self, device):
        super().to(device)
        self.device = device

    def encoder_forward_batch(self, agents: list[Agent]) -> torch.Tensor:
        """
        Returns an embedding corresponding to an `agent`.
        
        :param agent: An instance of Agent with `traits` and `state` attributes.
        :return: A latent vector representing the agent's embedding.
        """
        pass 

from  tensordict import TensorDict

class BasePolicyNet(nn.Module):
    def __init__(self, input_channels: int, grid_size: int, num_actions: int, state_size: int, belief_size: int, traits_size : int, device="cpu"):
        super(BasePolicyNet, self).__init__()

        self.grid_size = grid_size
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
       pass