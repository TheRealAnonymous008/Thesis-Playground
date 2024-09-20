from collections import deque
from gymnasium import Env
import os 
import numpy as np 
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random

from typing import Type

T_Optimizer = Type[optim.Optimizer]
T_Loss = nn.modules.loss._Loss

class BaseModel: 
    """
    Base Class for MARL solutions
    """

    def __init__(self, 
                 env : Env, 
                 policy_net : nn.Module,
                 buffer_size : int = 100000, 
                 batch_size : int = 64, 
                 gamma: float = 0.99, 
                 optimizer : T_Optimizer = torch.optim.Adam,
                 loss_fn : T_Loss = nn.MSELoss(),
                 lr : float = 1e-3,
                 ):
        """
        Initialize the model 

        :param env: The environment to learn from
        :param policy_net:  The policy network. 
        :param buffer_size: The size of the experience replay buffer
        :param batch_size: Learning batch size
        :param gamma: Discount factor for future rewards.
        :param optimizer: The optimizer class to be used for training (default assumed to be ADAM)
        :papram loss_fn: The loss function used for the optimizer
        :param lr : Learning rate for the optimizer  
        """
        self.env : Env= env
        self.policy_net : torch.nn.Module = policy_net

        self.rollout_buffer = deque(maxlen=buffer_size)  
        self.batch_size : int = batch_size,
        self.gamma : float = gamma 
        self.loss_fn : T_Loss = loss_fn
        self.optimizer : T_Optimizer = optimizer(self.policy_net.parameters(), lr = lr)

        self.t_step = 0


    def learn(self, total_timesteps : int): 
        """
        Learn for the specified number of time steps
        """
        state, _ = self.env.reset()
        print(state)
        state = torch.tensor(state, dtype = torch.float32).unsqueeze(0) # Add batch dimension

        for t in range(total_timesteps):
            action = self.select_action(state)
            next_state, reward, terminated, truncated, _ = self.env.step(action.item())
            done = torch.tensor([terminated or truncated], dtype = torch.float32) 
            reward = torch.tensor([reward], type = torch.float32)
            next_state = torch.tensor(next_state, dtype = torch.float32).unsqueeze(0) # Add batch dimension

            self.rollout_buffer.append(
                (state, action, reward, next_state, done)
            )

            state = next_state

            if done: 
                state, _ = self.env.reset()
                state = torch.tensor(state, dtype = torch.float32).unsqueeze(0) # Add batch dimension

        

    def select_action(self, state) -> torch.Tensor:
        """
        Select an action. Derived classes should overwrite this
        """
        raise NotImplementedError 
    
    def sample_experiences(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Randomly sample a batch of experiences from memory.
        """
        experiences = np.random.sample(self.rollout_buffer, k=self.batch_size)

        states = torch.cat([e[0] for e in experiences])
        actions = torch.cat([e[1] for e in experiences]).unsqueeze(1)
        rewards = torch.cat([e[2] for e in experiences]).unsqueeze(1)
        next_states = torch.cat([e[3] for e in experiences])
        dones = torch.cat([e[4] for e in experiences]).unsqueeze(1)

        return states, actions, rewards, next_states, dones

    def optimize_model(self, experiences):
        """
        Perform a learning step: update the policy network using a batch of experiences.
        """
        raise NotImplementedError

    def save(self, model_path : str): 
        """
        Savee the policy network to the specified path
        """
        torch.save(self.policy_net.state_dict(), model_path) 

    def load(self, model_path: str):
        """
        Load the policy network from the specified path
        """
        self.policy_net.load_state_dict(torch.load(model_path))