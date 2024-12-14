from .base_models import BasePolicyNet, BaseEncoder, BaseDecoder

class ComplexModel:
    def __init__(self, 
                 initializer_fn, 
                 config, 
                 device = "cpu",
                 ):
        """
        Helper class to store the policy, decoder and encoder net in one package.
        """
        self._policy_net : BasePolicyNet= None
        self._target_net : BasePolicyNet = None 
        self._encoder_net : BaseEncoder= None 
        self._decoder_net : BaseDecoder= None 
        self.device = device
        initializer_fn(self, config)
    
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
        return self._target_net