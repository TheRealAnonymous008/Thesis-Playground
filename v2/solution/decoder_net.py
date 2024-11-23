import torch
import torch.nn as nn
import torch.nn.functional as F

from core.agent import * 
from sar.sar_agent import *

from sar.sar_comm import SARMessagePacket


class Decoder(nn.Module):
    def __init__(self,  belief_dims = 5, packet_dims = 5, input_dims = 32, hidden_dim=32, grid_size = 5, device = "cpu"):
        """
        Initialize a Decoder Network. The Decoder Network updates an agent's belief based on incoming messages.
        
        :param packet_dim: The input dimension of the message packet data.
        :param hidden_dim: Hidden layer dimension.
        :param belief_dim: Output dimension representing the updated agent belief.
        """
        super(Decoder, self).__init__()

        self.out_channels = 4
        self.conv11 = nn.Conv2d(in_channels=1, out_channels=16, kernel_size=3, stride=1, padding=1)
        self.conv12 = nn.Conv2d(in_channels=16, out_channels=self.out_channels, kernel_size=3, stride=1, padding=1)
        
        self.conv21 = nn.Conv2d(in_channels=1, out_channels=16, kernel_size=3, stride=1, padding=1)
        self.conv22 = nn.Conv2d(in_channels=16, out_channels=self.out_channels, kernel_size=3, stride=1, padding=1)
        conv_output_size = grid_size * grid_size * self.out_channels
        intermediate_embed_size = 5

        self.conv_fc1 = nn.Linear(conv_output_size, intermediate_embed_size)
        self.conv_fc2 = nn.Linear(conv_output_size, intermediate_embed_size)

        fc1_input_size = 2 * intermediate_embed_size
        self.fc1 = nn.Linear(fc1_input_size, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, belief_dims)

        self.grid_size = grid_size
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
        # Process the vision tensor
        x = packet.exploration
        y = packet.victims

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
            x = F.pad(x, (pad_left, pad_right, pad_top, pad_bottom), value=-torch.inf)
            y = F.pad(y, (pad_left, pad_right, pad_top, pad_bottom), value=-torch.inf)
        x = F.relu(self.conv11(x))
        x = F.relu(self.conv12(x))
        x = x.flatten()
        x = self.conv_fc1(x)
        x = F.relu(x)  

        y = F.relu(self.conv21(y))
        y = F.relu(self.conv22(y))
        y = y.flatten()
        y = self.conv_fc2(y)
        y = F.relu(y) 

        # Compute the belief update
        h = torch.cat((x, y))
        h = F.relu(self.fc1(h))  # Hidden layer activation
        belief_update = self.fc2(h)  # Compute belief adjustment

        # Combine previous belief and belief update (convex combination)
        updated_belief = F.softmax(belief + belief_update, dim=-1)

        return updated_belief

        