import torch
import torch.nn as nn
import torch.nn.functional as F

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

    def forward(self, x):
        """
        Forward pass for the policy network.

        :param x: Input tensor of shape (batch_size, channels, grid_size, grid_size)
        :return: Q-values for each action
        """
        # Pass through the convolutional layers
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))

        # Flatten the output from conv layers
        x = x.view(x.size(0), -1)

        # Pass through fully connected layers
        x = F.relu(self.fc1(x))
        x = self.fc2(x)

        return x
