from collections import deque
from gymnasium import Env
import os 
import numpy as np 
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from solution.custom_gym import CustomGymEnviornment
from typing import Type, Callable, Dict

T_Optimizer = Type[optim.Optimizer]
T_Loss = nn.modules.loss._Loss
T_FeatureExtractor = Callable[[dict], dict[torch.Tensor]]

from  tensordict import TensorDict, LazyStackedTensorDict
from .complex_model import *


class BaseModel: 
    """
    Base Class for MARL solutions
    """

    def __init__(self, 
                 env : CustomGymEnviornment, 
                 model : ComplexModel,
                 feature_extractor : T_FeatureExtractor,
                 buffer_size : int = 100000, 
                 batch_size : int = 64, 
                 gamma: float = 0.99, 
                 optimizer : T_Optimizer = torch.optim.Adam,
                 loss_fn : T_Loss = nn.MSELoss(),
                 lr : float = 1e-3,
                 device : str = "cuda",
                 ):
        """
        Initialize the model 

        :param env: The environment to learn from
        :param feature_extractor: The feature extractor to apply to each state. Note that it must return a tensor with a batch dimension already defined.
        :param policy_net:  The policy network. 
        :param policy_net:  The encoder network. 
        :param buffer_size: The size of the experience replay buffer
        :param batch_size: Learning batch size
        :param gamma: Discount factor for future rewards.
        :param optimizer: The optimizer class to be used for training (default assumed to be ADAM)
        :papram loss_fn: The loss function used for the optimizer
        :param lr : Learning rate for the optimizer  
        """
        self.env : Env= env
        self._model = model

        self.rollout_buffer = deque(maxlen=buffer_size)  
        self.batch_size : int = batch_size,
        self.gamma : float = gamma 
        self.loss_fn : T_Loss = loss_fn
        
        self.policy_optimizer : T_Optimizer = optimizer(self._model._policy_net.parameters(), lr = lr)
        self.encoder_optimizer : T_Optimizer = optimizer(self._model._encoder_net.parameters(), lr = lr)
        self.decoder_optimizer : T_Optimizer = optimizer(self._model._decoder_net.parameters(), lr = lr)
        
        self.feature_extractor = feature_extractor
        self.t_step = 0
        self.device = device

        self._model.to(self.device)


    def learn(self, total_timesteps : int, optimization_passes : int = 1): 
        """
        Learn for the specified number of time steps
        """
        state, _ = self.env.reset()
        state = self.feature_extractor(state)

        for t in range(total_timesteps):
            action = self.select_joint_action(state)
            next_state, reward, terminated, truncated, _ = self.env.step(action)

            # TODO: Potentially refactor this? 

            next_state = self.feature_extractor(next_state)
            terminated = torch.tensor(list(terminated.values()), dtype = torch.bool,)
            truncated = torch.tensor(list(truncated.values()), dtype = torch.bool,) 
            done = torch.logical_or(terminated, truncated).to(dtype = torch.int8,)       # Note that we need this to be int so that we can do some arithmetic with it.
            reward = torch.tensor(list(reward.values()), dtype = torch.float32,)

            self.rollout_buffer.append(
                (state, action, reward, next_state, done)
            )

            state = next_state

            if done.all(): 
                state, _ = self.env.reset()
                state = self.feature_extractor(state)

        

    def select_joint_action(self, state : dict, deterministic : bool = False) -> dict:
        """
        Select a joint action
        
        `state` is assumed to be a dictionary keyed with agent ids. 
        `deterministic` is used to force deterministic action selection
        
        """
        actions = {}
        state = self._flatten_state_dict([state])
        for a, s in state.items():
            action = self.select_action(a, state, deterministic) 
            actions[a] = action

        return actions 
    
    def select_action(self, agent : int, state : dict, deterministic : bool = False) -> torch.Tensor:
        """
        Select an action for the specified agent. Derived classes should override this.
        """
        raise NotImplementedError
    
    def sample_experiences(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Randomly sample a batch of experiences from memory.
        """
        experience_idxs = np.random.choice(len(self.rollout_buffer), size=self.batch_size)

        states = [self.rollout_buffer[e][0] for e in experience_idxs]
        next_states = [self.rollout_buffer[e][3] for e in experience_idxs]

        # Note that it is more appropriate to have the state dict such that it is keyed on agents and eeach value is an entire batch.
        states = self._flatten_state_dict([self.rollout_buffer[e][0] for e in experience_idxs])
        actions = self._flatten_action_dict([self.rollout_buffer[e][1] for e in experience_idxs])
        rewards = torch.stack([self.rollout_buffer[e][2] for e in experience_idxs]).transpose(0, 1).to(self.device)
        next_states = self._flatten_state_dict([self.rollout_buffer[e][3] for e in experience_idxs])
        dones = torch.stack([self.rollout_buffer[e][4] for e in experience_idxs]).transpose(0, 1).to(self.device)

        return states, actions, rewards, next_states, dones

    @staticmethod
    def _flatten_state_dict(states : list[dict[int, TensorDict]]): 
        """
        Given a batch of states (in list form), returns a dictionary of states keyed on the agents.
        It is assumed that the dataclasses' members are all tensor types tto allow concatenation
        """
        flattened_states = {}
        for agent_id in states[0].keys():
            # First get all relevant agent states
            agent_states = LazyStackedTensorDict.lazy_stack([state[agent_id] for state in states])
            
            flattened_states[agent_id] = agent_states

        return flattened_states 

    @staticmethod
    def _flatten_action_dict(actions : list[dict]):
        """
        Given a batch of actions (in list form), returns a dictionary of states keyed on the agents
        """
        flattened_actions = {}
        for agent_id in actions[0].keys():
            agent_actions = [[action[agent_id]] for action in actions]
            flattened_actions[agent_id] = torch.tensor(agent_actions)

        return flattened_actions
    
    def optimize_model(self, experiences):
        """
        Perform a learning step: update the policy network using a batch of experiences.
        """
        raise NotImplementedError

    def save(self, model_path: str):
        """
        Save the entire model (policy, encoder, and decoder networks) to the specified path
        """
        torch.save({
            'policy_net': self._model._policy_net.state_dict(),
            'encoder_net': self._model._encoder_net.state_dict(),
            'decoder_net': self._model._decoder_net.state_dict()
        }, model_path)

    def load(self, model_path: str):
        """
        Load the entire model (policy, encoder, and decoder networks) from the specified path
        """
        checkpoint = torch.load(model_path)
        self._model._policy_net.load_state_dict(checkpoint['policy_net'])
        self._model._encoder_net.load_state_dict(checkpoint['encoder_net'])
        self._model._decoder_net.load_state_dict(checkpoint['decoder_net'])
