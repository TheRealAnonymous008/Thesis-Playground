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

    def decoder_forward(self, packet : SARMessagePacket, sender_embedding: torch.Tensor) -> torch.Tensor:
        """
        Returns a tensor representing the updated agent's belief based on the message content.

        :param message: A Message instance containing a SARMessagePacket.
        :param sender_embedding: The sender's embedding tensor based on their ID.
        :return: A belief vector for the agent, of dimension `BELIEF`.
        """
        # Extract the packet data and sender information
        location_data = packet.location
        
        x = torch.concat([location_data, sender_embedding])

        # Pass the location data through the network
        x = F.relu(self.fc1(x))
        belief_update = self.fc2(x)
        return belief_update
        