import torch
import torch.nn as nn
import torch.nn.functional as F

from core.agent import * 
from sar.sar_agent import *

from sar.sar_comm import SARMessagePacket


class Decoder(nn.Module):
    def __init__(self,  belief_dims = 5, packet_dims = 5, input_dims = 32, hidden_dim=32,device = "cpu"):
        """
        Initialize a Decoder Network. The Decoder Network updates an agent's belief based on incoming messages.
        
        :param packet_dim: The input dimension of the message packet data.
        :param hidden_dim: Hidden layer dimension.
        :param belief_dim: Output dimension representing the updated agent belief.
        """
        super(Decoder, self).__init__()
        self.fc1 = nn.Linear(packet_dims + input_dims, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, belief_dims)

        self.device = device

    def to(self, device):
        super().to(device)
        self.device = device

    def decoder_forward(self, belief: torch.Tensor, packet: SARMessagePacket, sender_embedding: torch.Tensor) -> torch.Tensor:
        """
        Updates the agent's belief based on the incoming message and sender embedding.

        :param belief: Previous belief tensor of shape [belief_dims].
        :param packet: An instance of SARMessagePacket containing the message data.
        :param sender_embedding: Sender's embedding tensor of shape [input_dims].
        :return: Updated belief tensor of shape [belief_dims].
        """
        # Extract the packet data
        location_data = packet.location  # Assuming shape [packet_dims]

        # Concatenate the location data and sender embedding
        x = torch.cat([location_data, sender_embedding], dim=-1)

        # Compute the belief update
        x = F.relu(self.fc1(x))  # Hidden layer activation
        belief_update = self.fc2(x)  # Compute belief adjustment

        # Combine previous belief and belief update (convex combination)
        updated_belief = F.softmax(belief + belief_update, dim=-1)

        return updated_belief

        