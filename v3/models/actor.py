import torch 
import torch.nn as nn 
import torch.functional as F 
import numpy as np 

from .param_settings import ParameterSettings
from .net_maker import *

class ActorEncoder(nn.Module):
    def __init__(self, config : ParameterSettings):
        super().__init__()
        self.config = config
        self.policy_network : RNNWrapper = make_net([config.d_obs + config.d_beliefs, 128, 128, config.d_het_weights])
        self.belief_update : DenseWrapper= make_net([config.d_beliefs + config.d_comm_state, 128, 128, config.d_het_weights], heterogeneous_activation="tanh")
        self.encoder_network : DenseWrapper= make_net([config.d_obs + config.d_beliefs, 128, config.d_het_weights], heterogeneous_activation="tanh")

    def to(self, device):
        self.policy_network.to(device)
        self.belief_update.to(device)
        self.encoder_network.to(device)

    def forward(self, o, h, z, p_weights, b_weights, e_weights): 
        """
        Input: 
        o - observation. Dimensions (num_agents, d_obs)
        h - hidden state. Dimensions (num_agents, d_beliefs)
        z - communication state from previous time step. Dimensions (num_agents, d_comm_state)
        whp, whb, whe - the heterogeneous layers. Dimensions (num_agents, *)
        """
        Q, h, ze = self.homogeneous_forward(o, h, z, b_weights)
        Q = self.policy_network.apply_heterogeneous_weights(Q, p_weights) 
        ze = self.encoder_network.apply_heterogeneous_weights(ze, e_weights)

        return Q, h, ze
    
    def homogeneous_forward(self, o, h, z, b_weights):
        input = torch.cat([h, z], dim = 1)
        h = self.belief_update(input)
        h = self.belief_update.apply_heterogeneous_weights(h, b_weights)

        input = torch.cat([o, h], dim = 1)
        Q= self.policy_network.forward(input)
        ze = self.encoder_network.forward(input)

        return Q, h, ze 

class CriticEncoder(nn.Module):
    def __init__(self, config: ParameterSettings):
        super().__init__()
        self.config = config
        input_dim = config.d_obs + config.d_beliefs + config.d_comm_state
        
        # Value estimation core
        self.value_network : DenseWrapper = make_net([input_dim, 128, config.d_het_weights])

    def to(self, device):
        self.value_network.to(device)

    def forward(self, o, h, z, crit_weights):
        """Returns (value_estimate)"""
        input = torch.cat([o, h, z], dim=1)
        V = self.value_network(input)
        # Value estimation (primary output)
        V = self.value_network.apply_heterogeneous_weights(V, crit_weights)
        return V

class Filter(nn.Module):
    def __init__(self, config : ParameterSettings):
        super().__init__()
        input_dims = config.d_comm_state + config.d_relation
        self.net : DenseWrapper = make_net([input_dims, 32, 32, config.d_het_weights])

    def to(self, device):
        self.net.to(device)

    def forward(self, zei,  Mij, f_weights):
        """
        Input:
        ze - encoder latent state. Dimension (d_latent).
        M - relation matrix entry.  Dimension: (d_relation)
        """
        zei = torch.Tensor.expand(zei, (Mij.shape[0], -1))
        input = torch.cat([zei, Mij], dim = 1)
        message = self.net.apply_heterogeneous_weights(self.net(input), f_weights)
        return message 
    
class DecoderUpdate(nn.Module):
    def __init__(self, config : ParameterSettings):
        super().__init__()
        input_dims = config.d_relation + config.d_message
        self.dec_net : DenseWrapper= make_net([input_dims, 128, config.d_het_weights])

        self.update_mean_net : DenseWrapper= make_net([input_dims, 128, config.d_het_weights])
        self.update_cov_net : DenseWrapper= make_net([input_dims, 128, config.d_het_weights])

    def to(self, device):
        self.dec_net.to(device)
        self.update_mean_net.to(device)
        self.update_cov_net.to(device)

    def forward(self, message, Mij, d_weights, um_weights, us_weights):
        input = torch.cat([message, Mij], dim = 1)


        zdj = self.dec_net.apply_heterogeneous_weights(self.dec_net(input), d_weights)
        mu = self.update_mean_net.apply_heterogeneous_weights(self.update_mean_net(input), um_weights)
        sigma = self.update_cov_net.apply_heterogeneous_weights(self.update_cov_net(input), us_weights)

        eps = torch.randn_like(sigma)
        newMij = mu + eps * sigma

        # Apply tanh to normalize
        newMij = torch.tanh(newMij)

        return zdj, newMij
    