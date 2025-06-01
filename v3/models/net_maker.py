import torch 
import torch.nn as nn 
import torch.functional as F 
import numpy as np 


class DenseWrapper(nn.Module):
    def __init__(self, 
                 params: list[int], 
                 last_activation: str = "leaky", 
                 heterogeneous_activation : str = None, 
                 dropout_rate: float = -1,
                 enable_batch_norm: bool = False,
                 enable_spectral_norm: bool = False):
        super().__init__()
        self.layers = nn.ModuleList()
        self.params = params
        self.num_layers = len(params) - 1

        if heterogeneous_activation == "sigmoid":
            self.het_act = torch.sigmoid 
        elif heterogeneous_activation == "tanh":
            self.het_act = torch.tanh 
        elif heterogeneous_activation == "softplus":
            self.het_act = nn.Softplus()
        else: 
            self.het_act = None 
            
        for i in range(self.num_layers):
            # Create linear layer
            layer = nn.Linear(params[i], params[i+1])
            
            # Initialize weights
            nn.init.xavier_normal_(layer.weight, gain=nn.init.calculate_gain('leaky_relu'))
            
            # Apply spectral norm if needed
            if enable_spectral_norm and i > 0:
                layer = nn.utils.spectral_norm(layer)
                nn.init.orthogonal_(layer.weight)
            
            self.layers.append(layer)
            
            # Add activation, batch norm, and dropout after all layers except last
            if i < self.num_layers - 1:
                self.layers.append(nn.PReLU())
                if enable_batch_norm:
                    self.layers.append(nn.BatchNorm1d(params[i+1]))
                if dropout_rate > 0:
                    self.layers.append(nn.Dropout(dropout_rate))
            else:
                # Handle last activation
                if last_activation == "leaky":
                    self.layers.append(nn.PReLU())
                elif last_activation == "sigmoid":
                    self.layers.append(nn.Sigmoid())
                elif last_activation == "tanh":
                    self.layers.append(nn.Tanh())
                elif last_activation == "softplus":
                    self.layers.append(nn.Softplus())

    def to(self, device):
        self.layers.to(device)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            x = layer(x)
        return x
        
    def apply_heterogeneous_weights(self, x, weights):
        w = weights["weight"]
        b = weights["bias"]

        y = torch.bmm(w, torch.unsqueeze(x, 2))
        y = torch.squeeze(y, 2) + b

        if self.het_act != None: 
            y = self.het_act(y)
        return y



def make_net(params: list[int], 
            last_activation: str = "leaky", 
            heterogeneous_activation : str = None, 
            dropout_rate: float = -1,
            enable_batch_norm: bool = False,
            enable_spectral_norm: bool = False) -> nn.Module:
    return DenseWrapper(
        params=params,
        last_activation=last_activation,
        dropout_rate=dropout_rate,
        enable_batch_norm=enable_batch_norm,
        enable_spectral_norm=enable_spectral_norm,
        heterogeneous_activation = heterogeneous_activation
    )