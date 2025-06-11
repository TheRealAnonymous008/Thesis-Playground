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
        self.het_latent = config.d_het_latent
        input_dim = config.d_traits + config.d_beliefs + config.d_obs + config.d_comm_state
        self.mean_net = make_net([input_dim, 64, config.d_het_latent], last_activation=False)
        self.std_net = make_net([input_dim, 64,config.d_het_latent ], last_activation = "softplus")

    def to(self, device):
        self.mean_net.to(device)
        self.std_net.to(device)

    def forward(self, inputs):
        """
        Outputs the MVN parameters for the latent distribution for sampling.
        """
        mu= self.mean_net(inputs)
        sigma = self.std_net(inputs) + 1e-8

        return mu, sigma
    
class PPOLatentDecoder(nn.Module): 
    def __init__(self, config: ParameterSettings):
        super().__init__()
        self.config = config
        self.enable_spectral_norm = True
        self.dropout_rate = -1
        # Policy
        self.p_net_weights = make_net([config.d_het_latent, 256, 256, 256, config.d_het_weights * config.d_action], enable_spectral_norm= self.enable_spectral_norm, last_activation = False, dropout_rate=self.dropout_rate)
        self.p_net_biases = make_net([config.d_het_latent, 256, 256, 256, config.d_action], enable_spectral_norm= self.enable_spectral_norm, last_activation = False, dropout_rate=self.dropout_rate)

        # Critic
        self.crit_weights = make_net([config.d_het_latent, 256, 256, 256, config.d_het_weights], enable_spectral_norm= self.enable_spectral_norm, last_activation = False, dropout_rate=self.dropout_rate)
        self.crit_biases = make_net([config.d_het_latent, 256, 256, 256, 1] , enable_spectral_norm= self.enable_spectral_norm, last_activation = False, dropout_rate=self.dropout_rate)

        # Belief
        self.b_net_weights = make_net([config.d_het_latent, 256, 256, 256, config.d_het_weights * config.d_beliefs] , enable_spectral_norm= self.enable_spectral_norm, last_activation = False, dropout_rate=self.dropout_rate)
        self.b_net_biases = make_net([config.d_het_latent, 256, 256, 256, config.d_beliefs] , enable_spectral_norm= self.enable_spectral_norm, last_activation = False , dropout_rate=self.dropout_rate)

        # Encoder
        self.e_net_weights = make_net([config.d_het_latent, 256, 256, 256, config.d_het_weights * config.d_comm_state] , enable_spectral_norm= self.enable_spectral_norm, last_activation = False , dropout_rate=self.dropout_rate)
        self.e_net_biases = make_net([config.d_het_latent, 256, 256, 256, config.d_comm_state] , enable_spectral_norm= self.enable_spectral_norm, last_activation = False , dropout_rate=self.dropout_rate)

        # Filter
        self.f_net_weights = make_net([config.d_het_latent, 256, 256, 256, config.d_het_weights * config.d_message] , enable_spectral_norm= self.enable_spectral_norm , last_activation = False , dropout_rate=self.dropout_rate)
        self.f_net_biases = make_net([config.d_het_latent, 256, 256, 256, config.d_message] , enable_spectral_norm= self.enable_spectral_norm, last_activation = False , dropout_rate=self.dropout_rate)

        # Decoder
        self.d_net_weights = make_net([config.d_het_latent, 256, 256, 256, config.d_het_weights * config.d_comm_state] , enable_spectral_norm= self.enable_spectral_norm, last_activation = False , dropout_rate=self.dropout_rate)
        self.d_net_biases = make_net([config.d_het_latent, 256, 256, 256, config.d_comm_state] , enable_spectral_norm= self.enable_spectral_norm, last_activation = False, dropout_rate=self.dropout_rate)

        # Update
        self.u_mean_net_weights = make_net([config.d_het_latent, 256, 256, 256, config.d_het_weights * config.d_relation] , enable_spectral_norm= self.enable_spectral_norm, last_activation = False , dropout_rate=self.dropout_rate)
        self.u_mean_net_biases = make_net([config.d_het_latent, 256, 256, 256, config.d_relation] , enable_spectral_norm= self.enable_spectral_norm, last_activation = False , dropout_rate=self.dropout_rate)
        
        self.u_bias_net_weights = make_net([config.d_het_latent, 256, 256, 256, config.d_het_weights * config.d_relation] , enable_spectral_norm= self.enable_spectral_norm, last_activation = False , dropout_rate=self.dropout_rate)
        self.u_bias_net_biases = make_net([config.d_het_latent, 256, 256, 256, config.d_relation] , enable_spectral_norm= self.enable_spectral_norm, last_activation = False , dropout_rate=self.dropout_rate) 

    def forward(self, lv):
        """
        Outputs the heterogeneous weights using the latent variable
        """

        return TensorDict({
            "policy": self.get_weights(lv, self.p_net_weights, self.p_net_biases, self.config.d_action),
            "critic": self.get_weights(lv, self.crit_weights, self.crit_biases, 1), 
            "belief": self.get_weights(lv, self.b_net_weights, self.b_net_biases, self.config.d_beliefs),
            "encoder": self.get_weights(lv, self.e_net_weights, self.e_net_biases, self.config.d_comm_state),
            "filter": self.get_weights(lv, self.f_net_weights, self.f_net_biases, self.config.d_message), 
            "decoder": self.get_weights(lv, self.d_net_weights, self.d_net_biases, self.config.d_comm_state), 
            "update_mean": self.get_weights(lv, self.u_mean_net_weights, self.u_mean_net_biases, self.config.d_relation),
            'update_std': self.get_weights(lv, self.u_bias_net_weights, self.u_bias_net_biases, self.config.d_relation),
        }, device = self.config.device)
    

    def to(self, device):
        # Policy networks
        self.p_net_weights.to(device)
        self.p_net_biases.to(device)

        # Critic networks
        self.crit_weights.to(device)
        self.crit_biases.to(device)

        # Belief networks
        self.b_net_weights.to(device)
        self.b_net_biases.to(device)

        # Encoder networks
        self.e_net_weights.to(device)
        self.e_net_biases.to(device)

        # Filter networks
        self.f_net_weights.to(device)
        self.f_net_biases.to(device)

        # Decoder networks
        self.d_net_weights.to(device)
        self.d_net_biases.to(device)

        # Update networks (mean and bias)
        self.u_mean_net_weights.to(device)
        self.u_mean_net_biases.to(device)
        self.u_bias_net_weights.to(device)
        self.u_bias_net_biases.to(device)

    def get_weights(self, lv, weight_net, bias_net, dims):
        w = weight_net(lv)
        w = w.reshape((-1, dims, self.config.d_het_weights))
        b = bias_net(lv)

        w *= self.config.hypernet_scale_factor
        b *= self.config.hypernet_scale_factor

        # Normalize weight matrices per agent (Frobenius norm)
        w_norm = torch.norm(w, p=2, dim=(1,2), keepdim=True)
        w = w / (w_norm + 1e-8)

        # Normalize bias vectors per agent (L2 norm)
        b_norm = torch.norm(b, p=2, dim=1, keepdim=True)
        b = b / (b_norm + 1e-8)

        return TensorDict({
            "weight": w, 
            "bias": b,
        })


class HyperNetwork (nn.Module):
    def __init__(self, config : ParameterSettings):
        super().__init__()
        self.latent_encoder = LatentEncoder(config)
        if config.type == "ppo":
            self.latent_decoder = PPOLatentDecoder(config)

    def to(self, device):
        self.latent_encoder.to(device)
        self.latent_decoder.to(device)

    def forward(self, c, o, h, z):
        """
        Inputs: 
        
        c - the context vector representing the agents. Dimensions (n_agents, d_traits)
        o - the current observation of the agent. DImensions (n_agents, d_obs)
        h - the current belief state of the agent. Dimensions (n_agents, d_belief)
        z - the current comm state of the agent. Dimension (n_agent , d_comm_state)

        Outputs: 
        lv - latent variable. Dimensions (n_agents, d_het_latents)
        wh - heterogeneous weights Dimensions (n_agents, d_het_weights)
        mu, sigma - the parameters of the latent distribution (n_agents, 2 * d_het_latents)
        """

        
        inputs = torch.cat([c, o, h, z], dim=1)  # Concatenate along feature dimension
        mu, sigma = self.latent_encoder(inputs)

        cov_matrix = torch.diag_embed(sigma) + 1e-8  # Create diagonal covariance matrix from variances

        dist = torch.distributions.MultivariateNormal(mu, covariance_matrix=cov_matrix)
        lv = dist.rsample() 
        
        lv_norm = torch.norm(lv, p=2, dim=1, keepdim=True)
        lv = lv / (lv_norm + 1e-8)

        weights  = self.latent_decoder(lv)
        # Return the latent variable and the heterogeneous weights
        return lv, weights, mu, sigma




