from .policy_net import PolicyNet
from .encoder_net import Encoder 
from .decoder_net import Decoder

class ComplexModel:
    def __init__(self, 
                 belief_dims : int  = 5, 
                 action_dims : int  = 4, 
                 state_dims  : int  = 4,
                 trait_dims  : int  = 3, 
                 grid_size   : int = 3, 
                 latent_dims : int = 16,
                 packet_dims : int = 5,
                 device = "cpu",
                 ):
        
        self._belief_dims = belief_dims
        self._action_dims = action_dims
        self._state_dims = state_dims
        self._trait_dims = trait_dims
        self._grid_size = grid_size
        self._latent_dims = latent_dims
        self._packet_dims = packet_dims

        self._policy_net = PolicyNet(
            input_channels=1, 
            grid_size = grid_size, 
            num_actions = action_dims, 
            state_size= state_dims, 
            belief_size= belief_dims, 
            traits_size= trait_dims,
            device = device)
        self._encoder_net = Encoder(
            state_dims= state_dims,
            trait_dims= trait_dims,
            hidden_dim = 32, 
            output_dim= latent_dims, 
            device = device
        ) 
        self._decoder_net = Decoder(
            input_dims= latent_dims, 
            hidden_dim = latent_dims, 
            packet_dims= packet_dims,
            device= device) 
    
    def to(self, device : str):
        """
        Move model to a specific device
        """
        self._policy_net.to(device)
        self._encoder_net.to(device)
        self._decoder_net.to(device)

    def eval(self):
        self._policy_net.eval()
        self._encoder_net.eval()
        self._decoder_net.eval()

    def train(self):
        self._policy_net.train()
        self._encoder_net.train()
        self._decoder_net.train()

    def get_target_net(self):
        return PolicyNet(
            input_channels=1,
            grid_size= self._grid_size,
            num_actions= self._action_dims,
            traits_size= self._trait_dims,
            state_size= self._state_dims,
            belief_size= self._belief_dims,
            device = self._policy_net.device
        )