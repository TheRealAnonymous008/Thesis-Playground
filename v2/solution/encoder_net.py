import torch
import torch.nn as nn
import torch.nn.functional as F

from core.agent import * 
from sar.sar_agent import *

from sar.sar_comm import SARMessagePacket

INPUT_DIMS = 5
from sar.sar_env_params import BELIEF_DIMS

class Encoder(nn.Module):
    def __init__(self, hidden_dim=64, output_dim=32, device = "cpu"):
        """
        Initialize an Encoder Network. The Encoder Network is used to provide latent space embeddings for each 
        heterogeneous agent based on traits and state information.
        
        :param input_dim: The input dimension, corresponding to the concatenated agent features.
        :param hidden_dim: The hidden dimension size.
        :param output_dim: The size of the latent space embedding.
        """
        super(Encoder, self).__init__()
        self.fc1 = nn.Linear(INPUT_DIMS, hidden_dim)
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
        batch_features = torch.stack(batch_features).to(dtype = torch.float32)    

        # Forward pass through the network 
        x = batch_features
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        
        return x

PACKET_DIMS = 2
class Decoder(nn.Module):
    def __init__(self, hidden_dim=32, device = "cpu"):
        """
        Initialize a Decoder Network. The Decoder Network updates an agent's belief based on incoming messages.
        
        :param packet_dim: The input dimension of the message packet data.
        :param hidden_dim: Hidden layer dimension.
        :param belief_dim: Output dimension representing the updated agent belief.
        """
        super(Decoder, self).__init__()
        self.fc1 = nn.Linear(PACKET_DIMS + 32, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, BELIEF_DIMS)

        self.device = device

    def to(self, device):
        super().to(device)
        self.device = device

    def decoder_forward(self, packet : SARMessagePacket, sender_embedding: torch.Tensor) -> torch.Tensor:
        """
        Returns a tensor representing the updated agent's belief based on the message content.

        :param message: A Message instance containing a SARMessagePacket.
        :param sender_embedding: The sender's embedding tensor based on their ID.
        :return: A belief vector for the agent, of dimension `BELIEF`.
        """
        # Extract the packet data and sender information
        location_data = torch.tensor(packet.location, dtype=torch.float32, device= self.device)
        
        x = torch.concat([location_data, sender_embedding])

        # Pass the location data through the network
        x = F.relu(self.fc1(x))
        belief_update = self.fc2(x)

        return belief_update
        