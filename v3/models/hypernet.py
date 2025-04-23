import torch 
import torch.nn as nn 
import torch.functional as F 
import numpy as np 

from .param_settings import ParameterSettings
from .net_maker import make_net

class LatentEncoder(nn.Module): 
    def __init__(self, config: ParameterSettings):
        super().__init__()
        input_dim = config.d_traits + config.d_beliefs
        self.net = make_net([input_dim, 128, 64, 2 * config.d_het_latent])

    def forward(self, inputs):
        """
        Outputs the MVN parameters for the latent distribution for sampling.
        """
        out = self.net(inputs)
        mu, log_var = torch.chunk(out, 2, dim=1)
        sigma = torch.exp(log_var)  # Convert log variance to variance
        return mu, sigma

class LatentDecoder(nn.Module): 
    def __init__(self, config: ParameterSettings):
        super().__init__()
        self.config = config
        # Policy
        self.p_net_weights = make_net([config.d_het_latent, 128, 256, 512, config.d_het_weights * config.d_action])
        self.p_net_biases = make_net([config.d_het_latent, 128, 256, 512, config.d_action])

        # Critic
        self.crit_weights = make_net([config.d_het_latent, 128, 256, 512, config.d_het_weights])
        self.crit_biases = make_net([config.d_het_latent, 128, 256, 512, 1])

        # Belief
        self.b_net_weights = make_net([config.d_het_latent, 128, 256, 512, config.d_het_weights * config.d_beliefs])
        self.b_net_biases = make_net([config.d_het_latent, 128, 256, 512, config.d_beliefs])

        # Encoder
        self.e_net_weights = make_net([config.d_het_latent, 128, 256, 512, config.d_het_weights * config.d_comm_state])
        self.e_net_biases = make_net([config.d_het_latent, 128, 256, 512, config.d_comm_state])

        # Filter
        self.f_net_weights = make_net([config.d_het_latent, 128, 256, 512, config.d_het_weights * config.d_message])
        self.f_net_biases = make_net([config.d_het_latent, 128, 256, 512, config.d_message])

        # Decoder
        self.d_net_weights = make_net([config.d_het_latent, 128, 256, 512, config.d_het_weights * config.d_comm_state])
        self.d_net_biases = make_net([config.d_het_latent, 128, 256, 512, config.d_comm_state])

        # Update
        self.u_mean_net_weights = make_net([config.d_het_latent, 128, 256, 512, config.d_het_weights * config.d_relation])
        self.u_mean_net_biases = make_net([config.d_het_latent, 128, 256, 512, config.d_relation])
        
        self.u_bias_net_weights = make_net([config.d_het_latent, 128, 256, 512, config.d_het_weights * config.d_relation])
        self.u_bias_net_biases = make_net([config.d_het_latent, 128, 256, 512, config.d_relation])

    def forward(self, lv):
        """
        Outputs the heterogeneous weights using the latent variable
        """
        # Policy
        whp = self.p_net_weights(lv)
        whp = whp.reshape((-1, self.config.d_action, self.config.d_het_weights))
        bp = self.p_net_biases(lv)

        # Critic
        whc = self.crit_weights(lv)
        whc = whc.reshape((-1, 1, self.config.d_het_weights))
        bc = self.crit_biases(lv)

        # Belief
        whb = self.b_net_weights(lv)
        whb = whb.reshape((-1, self.config.d_beliefs, self.config.d_het_weights))
        bb = self.b_net_biases(lv)

        # Encoder
        whe = self.e_net_weights(lv)
        whe = whe.reshape((-1, self.config.d_comm_state, self.config.d_het_weights))
        be = self.e_net_biases(lv)

        # Filter
        whf = self.f_net_weights(lv)
        whf = whf.reshape((-1, self.config.d_message, self.config.d_het_weights))
        bf = self.f_net_biases(lv)

        # Decoder
        whd = self.d_net_weights(lv)
        whd = whd.reshape((-1, self.config.d_comm_state, self.config.d_het_weights))
        bd = self.d_net_biases(lv)

        # Update
        whum = self.u_mean_net_weights(lv)
        whum = whum.reshape((-1, self.config.d_relation, self.config.d_het_weights))
        bum = self.u_mean_net_biases(lv)

        whus = self.u_bias_net_weights(lv)
        whus = whus.reshape((-1, self.config.d_relation, self.config.d_het_weights))
        bus = self.u_bias_net_biases(lv)

        return {
            "policy": (whp, bp),
            "critic": (whc, bc), 
            "belief": (whb, bb),
            "encoder": (whe, be),
            "filter": (whf, bf), 
            "decoder": (whd, bd), 
            "update": (whum, bum, whus, bus)
        }

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
        """
        inputs = torch.cat([c, h], dim=1)  # Concatenate along feature dimension
        mu, sigma = self.latent_encoder(inputs)

        # Reparameterization trick to sample latent variable
        std = torch.sqrt(sigma)
        eps = torch.randn_like(std)
        lv = mu + eps * std

        weights  = self.latent_decoder(lv)

        # Return the latent variable and the heterogeneous weights
        return lv, weights




