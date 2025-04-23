from .actor import * 
from .hypernet import * 

class Model: 
    def __init__(self, config : ParameterSettings):
        self.config = config
        self.hypernet = HyperNetwork(config)
        self.actor_encoder = ActorEncoder(config)
        self.actor_encoder_critic = CriticEncoder(config)

        self.filter = Filter(config)
        self.decoder_update = DecoderUpdate(config)

        self.to(config.device)

    def to(self, device):
        self.device = device

        self.hypernet.to(device)
        self.actor_encoder.to(device)
        self.actor_encoder_critic.to(device)
        self.filter.to(device)
        self.decoder_update.to(device)

    def requires_grad_(self, val):
        self.hypernet.requires_grad_(val)
        self.actor_encoder.requires_grad_(val)
        self.actor_encoder_critic.requires_grad_(val)
        self.filter.requires_grad_(val)
        self.decoder_update.requires_grad_(val)

    def param_count(self):
        hypernet_params = sum(p.numel() for p in self.hypernet.parameters() if p.requires_grad)
        actor_encoder_params = sum(p.numel() for p in self.actor_encoder.parameters() if p.requires_grad)
        filter_params = sum(p.numel() for p in self.filter.parameters() if p.requires_grad)
        decoder_update_params = sum(p.numel() for p in self.decoder_update.parameters() if p.requires_grad)
        total_params = hypernet_params + actor_encoder_params + filter_params + decoder_update_params

        report = f"""
        Hypernet: {hypernet_params} 
        Actor Encoder: {actor_encoder_params} 
        Acrot Encoder Critic: {actor_encoder_params}
        Filter: {filter_params} 
        Decoder: {decoder_update_params} 
        Total: {total_params}
        """ 
        print(report)