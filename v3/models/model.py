from .actor import * 
from .hypernet import * 

class SACModel: 
    def __init__(self, config : ParameterSettings):
        self.config = config
        self.hypernet = HyperNetwork(config)
        self.actor_encoder = ActorEncoder(config)
        self.q1 = CriticEncoder(config)
        self.q2 = CriticEncoder(config)

        self.filter = Filter(config)
        self.decoder_update = DecoderUpdate(config)

        self.target_q1 = CriticEncoder(config)
        self.target_q2 = CriticEncoder(config)

        self.log_alpha = nn.Parameter(torch.tensor(0.2, dtype = torch.float16), requires_grad=True)
        self.to(config.device)

        self._parameters = list(self.hypernet.parameters()) + \
            list(self.actor_encoder.parameters())  + \
            list(self.q1.parameters()) + \
            list(self.q2.parameters()) + \
            list(self.filter.parameters()) + \
            list(self.decoder_update.parameters()) + \
            list([self.log_alpha])
    
    @property
    def alpha(self):
        return self.log_alpha.exp()
    
    def set_alpha(self, x ):
        self.log_alpha = torch.tensor(x, dtype = torch.float16, device = self.device, requires_grad=True)

    def to(self, device):
        self.device = device

        self.hypernet.to(device)
        self.actor_encoder.to(device)
        self.q1.to(device)
        self.q2.to(device)
        self.filter.to(device)
        self.decoder_update.to(device)

        self.target_q1.to(device)
        self.target_q2.to(device)

        self.log_alpha.to(device)

    def requires_grad_(self, val):
        self.hypernet.requires_grad_(val)
        self.actor_encoder.requires_grad_(val)
        self.q1.requires_grad_(val)
        self.q2.requires_grad_(val)
        self.filter.requires_grad_(val)
        self.decoder_update.requires_grad_(val)

        self.log_alpha.requires_grad_(val)

    def param_count(self):
        hypernet_params = sum(p.numel() for p in self.hypernet.parameters() if p.requires_grad)
        actor_encoder_params = sum(p.numel() for p in self.actor_encoder.parameters() if p.requires_grad)
        actor_critic_params = sum(p.numel() for p in self.q1.parameters() if p.requires_grad)
        filter_params = sum(p.numel() for p in self.filter.parameters() if p.requires_grad)
        decoder_update_params = sum(p.numel() for p in self.decoder_update.parameters() if p.requires_grad)
        total_params = hypernet_params + actor_encoder_params + actor_critic_params +  filter_params + decoder_update_params

        report = f"""
        Hypernet: {hypernet_params} 
        Actor Encoder: {actor_encoder_params} 
        Acrot Encoder Critic: {actor_critic_params}
        Filter: {filter_params} 
        Decoder: {decoder_update_params} 
        Total: {total_params}
        """ 
        print(report)

    def parameters(self):
        return self._parameters