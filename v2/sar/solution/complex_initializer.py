from models.complex_model import * 
from .decoder_net import *
from .encoder_net import *
from .policy_net import * 

@dataclass
class SARModelConfig: 
    belief_dims : int  = 5 
    action_dims : int  = 4 
    state_dims  : int  = 4
    trait_dims  : int  = 3 
    grid_size   : int = 3 
    latent_dims : int = 16
    packet_dims : int = 5

def sar_initializer_fn(model : ComplexModel, config : SARModelConfig):


    model._policy_net = PolicyNet(
        input_channels=1, 
        grid_size = config.grid_size, 
        num_actions = config.action_dims, 
        state_size= config.state_dims, 
        belief_size= config.belief_dims, 
        traits_size= config.trait_dims,
        device = model.device)
    model._encoder_net = Encoder(
        state_dims= config.state_dims,
        trait_dims= config.trait_dims,
        hidden_dim = 32, 
        output_dim= config.latent_dims, 
        device = model.device
    ) 
    model._decoder_net = Decoder(
        input_dims= config.latent_dims, 
        belief_dims= config.belief_dims,
        hidden_dim = config.latent_dims, 
        packet_dims= config.packet_dims,
        grid_size= config.grid_size,
        device= model.device) 
    
    model._target_net = PolicyNet(
            input_channels=1,
            grid_size= config.grid_size,
            num_actions= config.action_dims,
            traits_size= config.trait_dims,
            state_size= config.state_dims,
            belief_size= config.belief_dims,
            device = model.device
        )