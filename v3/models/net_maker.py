import torch 
import torch.nn as nn 
import torch.functional as F 
import numpy as np 

def make_net(params: list[int], 
             last_activation = True, 
             dropout_rate = 0.1, 
             enable_batch_norm = False, 
             enable_spectral_norm = False,
    ) -> nn.Sequential:
    layers = []

    for i in range(len(params) - 1):
        linear_layer = nn.Linear(params[i], params[i + 1])
        nn.init.xavier_normal_(linear_layer.weight, 
                               gain=nn.init.calculate_gain('leaky_relu')
        )
        
        # Add spectral norm to deeper layers
        if enable_spectral_norm and i > 0:
            linear_layer = nn.utils.spectral_norm(linear_layer)
            nn.init.orthogonal_(linear_layer.weight)

        layers.append(linear_layer)

        if last_activation and i == len(params) - 1: 
            layers.append(nn.LeakyReLU())
        elif i < len(params) - 1: 
            layers.append(nn.LeakyReLU())


        if enable_batch_norm:
            layers.append(nn.BatchNorm1d(params[i + 1]))


        if dropout_rate > 0: 
            layers.append(nn.Dropout(dropout_rate))
    
    return nn.Sequential(*layers)

def apply_heterogeneous_weights(x, weights, sigmoid = True):
    w = weights["weight"]
    b = weights["bias"]

    y = torch.bmm(w, torch.unsqueeze(x, 2))
    y = torch.squeeze(y, 2) + b
    if sigmoid:
        y = torch.sigmoid(y)

    return y

def expand_weights(batches, idx, weights):
    weight = weights[0][idx], weights[1][idx]
    w = torch.Tensor.expand(weight[0], (batches, weight[0].shape[0], weight[0].shape[1]))
    b = torch.Tensor.expand(weight[1], (batches, weight[1].shape[0]))
    return (w,b)