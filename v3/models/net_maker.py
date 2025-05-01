import torch 
import torch.nn as nn 
import torch.functional as F 
import numpy as np 

def make_net(params: list[int]) -> nn.Sequential:
    layers = []

    for i in range(len(params) - 1):
        linear_layer = nn.Linear(params[i], params[i + 1])
        nn.init.xavier_normal_(linear_layer.weight, gain = nn.init.calculate_gain("leaky_relu"))
        layers.append(linear_layer)
        
        layers.append(nn.BatchNorm1d(params[i + 1]))

        layers.append(nn.LeakyReLU())
        layers.append(nn.Dropout())
    
    return nn.Sequential(*layers)

def apply_heterogeneous_weights(x, weights, sigmoid = False ):
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