import torch 
import torch.nn as nn 
import torch.functional as F 
import numpy as np 


class DenseWrapper(nn.Module):
    def __init__(self, 
                 params: list[int], 
                 last_activation: str = "leaky", 
                 dropout_rate: float = -1,
                 enable_batch_norm: bool = False,
                 enable_spectral_norm: bool = False):
        super().__init__()
        self.layers = nn.ModuleList()
        self.params = params
        self.num_layers = len(params) - 1

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
                self.layers.append(nn.LeakyReLU())
                if enable_batch_norm:
                    self.layers.append(nn.BatchNorm1d(params[i+1]))
                if dropout_rate > 0:
                    self.layers.append(nn.Dropout(dropout_rate))
            else:
                # Handle last activation
                if last_activation == "leaky":
                    self.layers.append(nn.LeakyReLU())
                elif last_activation == "sigmoid":
                    self.layers.append(nn.Sigmoid())
                elif last_activation == "tanh":
                    self.layers.append(nn.Tanh())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            x = layer(x)
        return x
        
    def apply_heterogeneous_weights(self, x, weights, sigmoid = False, tanh = False ):
        w = weights["weight"]
        b = weights["bias"]

        y = torch.bmm(w, torch.unsqueeze(x, 2))
        y = torch.squeeze(y, 2) + b
        if sigmoid:
            y = torch.sigmoid(y)
        if tanh:
            y = torch.tanh(y)
        return y



def make_net(params: list[int], 
            last_activation: str = "leaky", 
            dropout_rate: float = -1,
            enable_batch_norm: bool = False,
            enable_spectral_norm: bool = False) -> nn.Module:
    return DenseWrapper(
        params=params,
        last_activation=last_activation,
        dropout_rate=dropout_rate,
        enable_batch_norm=enable_batch_norm,
        enable_spectral_norm=enable_spectral_norm
    )


class RNNWrapper(nn.Module):
    def __init__(self, params, num_rnn_layers=1, dropout_rate=0.0, 
                 enable_batch_norm=False, enable_spectral_norm=False,
                 last_activation='leaky'):
        super(RNNWrapper, self).__init__()
        input_size, hidden_size = params[0], params[1]
        output_size = params[-1]
        
        self.rnn = nn.RNN(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_rnn_layers,
            batch_first=True,
            dropout=dropout_rate if num_rnn_layers > 1 else 0.0
        )
        
        # Post-RNN layers
        post_layers = []
        if hidden_size != output_size:
            linear = nn.Linear(hidden_size, output_size)
            if enable_spectral_norm:
                linear = nn.utils.spectral_norm(linear)
            post_layers.append(linear)
        
        if enable_batch_norm:
            post_layers.append(nn.BatchNorm1d(output_size))
        
        if last_activation == 'leaky':
            post_layers.append(nn.LeakyReLU())
        elif last_activation == 'tanh':
            post_layers.append(nn.Tanh())
        
        self.post_net = nn.Sequential(*post_layers)
        
    def forward(self, x, h0=None):
        # x shape: (batch_size, seq_len, input_size)
        rnn_out, hn = self.rnn(x, h0)
        batch_size, seq_len, hidden_size = rnn_out.shape
        
        # Process each time step through post_net
        processed = self.post_net(rnn_out.contiguous().view(-1, hidden_size))
        processed = processed.view(batch_size, seq_len, -1)
        
        return processed, hn

    def apply_heterogeneous_weights(self, x, weights, sigmoid=False, tanh=False):
        # Modified to handle 3D inputs (batch, seq, features)
        w = weights["weight"]
        b = weights["bias"]
        
        # x shape: (batch, seq, features_in)
        # w shape: (batch, features_in, features_out)
        y = torch.bmm(x, w) + b.unsqueeze(1)
        
        if sigmoid:
            y = torch.sigmoid(y)
        if tanh:
            y = torch.tanh(y)
        return y

def make_rnn_net(params: list[int], 
             last_activation: str = "leaky", 
             dropout_rate: float = -1,
             enable_batch_norm: bool = False, 
             enable_spectral_norm: bool = False,
             num_rnn_layers: int = 0  # New parameter for RNN support
            ) -> nn.Module:
    
    if num_rnn_layers > 0:
        return RNNWrapper(
            params=params,
            num_rnn_layers=num_rnn_layers,
            dropout_rate=dropout_rate,
            enable_batch_norm=enable_batch_norm,
            enable_spectral_norm=enable_spectral_norm,
            last_activation=last_activation
        )
    else:
        make_net(
            params = params, 
            last_activation= last_activation,
            dropout_rate= dropout_rate,
            enable_batch_norm= enable_batch_norm,
            enable_spectral_norm= enable_spectral_norm
        )