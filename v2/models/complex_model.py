import torch
import torch.nn as nn

class ComplexModel:
    def __init__(self, 
                 policy_net : nn.Module,
                 encoder_net : nn.Module,
                 decoder_net : nn.Module, 
                 ):
        self._policy_net = policy_net 
        self._encoder_net = encoder_net 
        self._decoder_net = decoder_net 
    
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