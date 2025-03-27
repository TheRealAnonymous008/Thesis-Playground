from .actor import * 
from .hypernet import * 

class Model: 
    def __init__(self, config : ParameterSettings):
        self.hypernet = HyperNetwork(config)
        self.actor_encoder = ActorEncoder(config)
        self.filter = Filter(config)
        self.decoder_update = DecoderUpdate(config)

        self.to(config.device)

    def to(self, device):
        self.device = device

        self.hypernet.to(device)
        self.actor_encoder.to(device)
        self.filter.to(device)
        self.decoder_update.to(device)

    def param_count(self):
        hypernet_params = sum(p.numel() for p in self.hypernet.parameters() if p.requires_grad)
        actor_encoder_params = sum(p.numel() for p in self.actor_encoder.parameters() if p.requires_grad)
        filter_params = sum(p.numel() for p in self.filter.parameters() if p.requires_grad)
        decoder_update_params = sum(p.numel() for p in self.decoder_update.parameters() if p.requires_grad)
        total_params = hypernet_params + actor_encoder_params + filter_params + decoder_update_params

        report = f"""
        Hypernet: {hypernet_params} 
        Actor Encoder: {actor_encoder_params} 
        Filter: {filter_params} 
        Decoder: {decoder_update_params} 
        Total: {total_params}
        """ 
        print(report)