import torch 
import torch.nn as nn 
import torch.functional as F 
import numpy as np 

def make_net(params : list[int]) : 
    layers = []

    for i in range(0, len(params) - 1): 
        layers.append(nn.Linear(params[i], params[i + 1]))
        if i < len(params) - 1:
            layers.append(nn.ReLU())

    return nn.Sequential(*layers)

def apply_heterogeneous_weights(x, weights):
    w, b = weights
    y = torch.bmm(w, torch.unsqueeze(x, 2))
    y = torch.squeeze(y, 2) + b

    return y

def expand_weights(batches, idx, weights):
    weight = weights[0][idx], weights[1][idx]
    w = torch.Tensor.expand(weight[0], (batches, weight[0].shape[0], weight[0].shape[1]))
    b = torch.Tensor.expand(weight[1], (batches, weight[1].shape[0]))
    return (w,b)