import torch 
import torch.nn as nn 
import torch.functional as F 
import numpy as np 

from .param_settings import ParameterSettings
from .net_maker import make_net
from tensordict import TensorDict

class LatentEncoder(nn.Module): 
    def __init__(self, config: ParameterSettings):
        super().__init__()
        input_dim = config.d_traits + config.d_beliefs
        self.net = make_net([input_dim, 256, 2 * config.d_het_latent], last_activation=False)

    def forward(self, inputs):
        """
        Outputs the MVN parameters for the latent distribution for sampling.
        """
        out = self.net(inputs)
        mu, log_var = torch.chunk(out, 2, dim=1)
        sigma = torch.exp(log_var)  # Convert log variance to variance

        # Ensure sigma is positive
        sigma = torch.abs(sigma)
        return mu, sigma

class LatentDecoder(nn.Module): 
    def __init__(self, config: ParameterSettings):
        super().__init__()
        self.config = config
        self.enable_spectral_norm = True
        # Policy
        self.p_net_weights = make_net([config.d_het_latent, 256, 256, 256, config.d_het_weights * config.d_action], enable_spectral_norm= self.enable_spectral_norm, last_activation = False)
        self.p_net_biases = make_net([config.d_het_latent, 256, 256, 256, config.d_action], enable_spectral_norm= self.enable_spectral_norm, last_activation = False)

        # Critic
        self.crit_weights = make_net([config.d_het_latent, 256, 256, 256, config.d_het_weights], enable_spectral_norm= self.enable_spectral_norm, last_activation = False)
        self.crit_biases = make_net([config.d_het_latent, 256, 256, 256, 1] , enable_spectral_norm= self.enable_spectral_norm, last_activation = False)

        # Belief
        self.b_net_weights = make_net([config.d_het_latent, 256, 256, 256, config.d_het_weights * config.d_beliefs] , enable_spectral_norm= self.enable_spectral_norm, last_activation = False)
        self.b_net_biases = make_net([config.d_het_latent, 256, 256, 256, config.d_beliefs] , enable_spectral_norm= self.enable_spectral_norm, last_activation = False)

        # Encoder
        self.e_net_weights = make_net([config.d_het_latent, 256, 256, 256, config.d_het_weights * config.d_comm_state] , enable_spectral_norm= self.enable_spectral_norm, last_activation = False)
        self.e_net_biases = make_net([config.d_het_latent, 256, 256, 256, config.d_comm_state] , enable_spectral_norm= self.enable_spectral_norm, last_activation = False)

        # Filter
        self.f_net_weights = make_net([config.d_het_latent, 256, 256, 256, config.d_het_weights * config.d_message] , enable_spectral_norm= self.enable_spectral_norm , last_activation = False)
        self.f_net_biases = make_net([config.d_het_latent, 256, 256, 256, config.d_message] , enable_spectral_norm= self.enable_spectral_norm, last_activation = False)

        # Decoder
        self.d_net_weights = make_net([config.d_het_latent, 256, 256, 256, config.d_het_weights * config.d_comm_state] , enable_spectral_norm= self.enable_spectral_norm, last_activation = False)
        self.d_net_biases = make_net([config.d_het_latent, 256, 256, 256, config.d_comm_state] , enable_spectral_norm= self.enable_spectral_norm, last_activation = False)

        # Update
        self.u_mean_net_weights = make_net([config.d_het_latent, 256, 256, 256, config.d_het_weights * config.d_relation] , enable_spectral_norm= self.enable_spectral_norm, last_activation = False)
        self.u_mean_net_biases = make_net([config.d_het_latent, 256, 256, 256, config.d_relation] , enable_spectral_norm= self.enable_spectral_norm, last_activation = False)
        
        self.u_bias_net_weights = make_net([config.d_het_latent, 256, 256, 256, config.d_het_weights * config.d_relation] , enable_spectral_norm= self.enable_spectral_norm, last_activation = False)
        self.u_bias_net_biases = make_net([config.d_het_latent, 256, 256, 256, config.d_relation] , enable_spectral_norm= self.enable_spectral_norm, last_activation = False)

    def forward(self, lv):
        """
        Outputs the heterogeneous weights using the latent variable
        """

        return TensorDict({
            "policy": self.get_weights(lv, self.p_net_weights, self.p_net_biases, self.config.d_action),
            "critic": self.get_weights(lv, self.crit_weights, self.crit_biases, 1), 
            "q2": self.get_weights(lv, self.crit_weights, self.crit_biases, 1),
            "belief": self.get_weights(lv, self.b_net_weights, self.b_net_biases, self.config.d_beliefs),
            "encoder": self.get_weights(lv, self.e_net_weights, self.e_net_biases, self.config.d_comm_state),
            "filter": self.get_weights(lv, self.f_net_weights, self.f_net_biases, self.config.d_message), 
            "decoder": self.get_weights(lv, self.d_net_weights, self.d_net_biases, self.config.d_comm_state), 
            "update_mean": self.get_weights(lv, self.u_mean_net_weights, self.u_mean_net_biases, self.config.d_relation),
            'update_std': self.get_weights(lv, self.u_bias_net_weights, self.u_bias_net_biases, self.config.d_relation),
        }, device = self.config.device)
    
    def get_weights(self, lv, weight_net, bias_net, dims):
        lv.detach()

        w = weight_net(lv)
        w = w.reshape((-1, dims, self.config.d_het_weights))
        b = bias_net(lv)

        # w = torch.nn.functional.layer_norm(w, [self.config.d_het_weights])
        # b = torch.nn.functional.layer_norm(b, [dims])
        w = w.cpu() * self.config.hypernet_scale_factor
        b = b.cpu() * self.config.hypernet_scale_factor

        return TensorDict({
            "weight": w, 
            "bias": b,
        })


class HyperNetwork (nn.Module):
    def __init__(self, config : ParameterSettings):
        super().__init__()
        self.latent_encoder = LatentEncoder(config)
        self.latent_decoder = LatentDecoder(config)


    def forward(self, c, h):
        """
        Inputs: 
        
        c - the context vector representing the agents. Dimensions (n_agents, d_traits)
        h - the current belief state of the agent. Dimensions (n_agents, d_belief)

        Outputs: 
        lv - latent variable. Dimensions (n_agents, d_het_latents)
        wh - heterogeneous weights Dimensions (n_agents, d_het_weights)
        mu, sigma - the parameters of the latent distribution (n_agents, 2 * d_het_latents)
        """
        inputs = torch.cat([c, h], dim=1)  # Concatenate along feature dimension
        mu, sigma = self.latent_encoder(inputs)

        cov_matrix = torch.diag_embed(torch.sqrt(sigma))  # Create diagonal covariance matrix from variances
        dist = torch.distributions.MultivariateNormal(mu, covariance_matrix=cov_matrix)
        lv = dist.rsample() 

        weights  = self.latent_decoder(lv)
        # Return the latent variable and the heterogeneous weights
        return lv, weights, mu, sigma




