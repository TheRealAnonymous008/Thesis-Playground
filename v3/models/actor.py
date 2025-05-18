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
        input_dim = config.d_obs + config.d_beliefs + config.d_comm_state
        self.policy_network = make_net([input_dim, 128, config.d_het_weights])
        self.belief_update = make_net([input_dim, 128, config.d_het_weights])
        self.encoder_network = make_net([input_dim, 128, config.d_het_weights])

    def forward(self, o, h, z, p_weights, b_weights, e_weights): 
        """
        Input: 
        o - observation. Dimensions (num_agents, d_obs)
        h - hidden state. Dimensions (num_agents, d_beliefs)
        z - communication state from previous time step. Dimensions (num_agents, d_comm_state)
        whp, whb, whe - the heterogeneous layers. Dimensions (num_agents, *)
        """
        input = torch.cat([o, h, z], dim = 1)

        # Get the logits for the actions
        Q = apply_heterogeneous_weights(self.policy_network(input), p_weights, sigmoid = False )
        h = apply_heterogeneous_weights(self.belief_update(input), b_weights, sigmoid = True )
        ze = apply_heterogeneous_weights(self.encoder_network(input), e_weights, sigmoid = False )

        return Q, h, ze
    
    def homogeneous_forward(self, o, h, z):
        input = torch.cat([o, h, z], dim = 1)
        Q = self.policy_network(input)
        h = self.belief_update(input)
        ze = self.encoder_network(input)

        return Q, h, ze 

class CriticEncoder(nn.Module):
    def __init__(self, config: ParameterSettings):
        super().__init__()
        self.config = config
        input_dim = config.d_obs + config.d_beliefs + config.d_comm_state
        
        # Value estimation core
        self.value_network = make_net([input_dim, 128, config.d_het_weights])

    def forward(self, o, h, z, crit_weights):
        """Returns (value_estimate)"""
        input = torch.cat([o, h, z], dim=1)
        V = self.value_network(input)
        # Value estimation (primary output)
        V = apply_heterogeneous_weights(V, crit_weights, sigmoid = False)
        return V

class Filter(nn.Module):
    def __init__(self, config : ParameterSettings):
        super().__init__()
        input_dims = config.d_comm_state + config.d_relation
        self.net = make_net([input_dims, 32, 32, config.d_het_weights])

    def forward(self, zei,  Mij, f_weights):
        """
        Input:
        ze - encoder latent state. Dimension (d_latent).
        M - relation matrix entry.  Dimension: (d_relation)
        """
        zei = torch.Tensor.expand(zei, (Mij.shape[0], -1))
        input = torch.cat([zei, Mij], dim = 1)
        message = apply_heterogeneous_weights(self.net(input), f_weights, sigmoid = False )

        return message 
    
class DecoderUpdate(nn.Module):
    def __init__(self, config : ParameterSettings):
        super().__init__()
        input_dims = config.d_relation + config.d_message
        self.dec_net = make_net([input_dims, 128, config.d_het_weights])

        self.update_mean_net = make_net([input_dims, 128, config.d_het_weights])
        self.update_cov_net = make_net([input_dims, 128, config.d_het_weights])

    def forward(self, message, Mij, d_weights, u_weights):
        input = torch.cat([message, Mij], dim = 1)

        whum, bum, whus, bus = u_weights 
        um_weights = (whum, bum)
        us_weights = (whus, bus)

        zdj = apply_heterogeneous_weights(self.dec_net(input), d_weights, sigmoid = False )
        mu = apply_heterogeneous_weights(self.update_mean_net(input), um_weights, sigma = False)
        sigma = apply_heterogeneous_weights(self.update_cov_net(input), us_weights, sigma = False)

        std = torch.sqrt(sigma)
        eps = torch.randn_like(std)
        newMij = mu + eps * std

        return zdj, newMij
    