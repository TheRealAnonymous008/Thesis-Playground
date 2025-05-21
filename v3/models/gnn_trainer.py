from .losses import * 
from .model import * 
from .base_env import *
from tests.eval import *

import torch 
from torch.utils.tensorboard import SummaryWriter
from .param_settings import TrainingParameters

def train_gnn(model: SACModel | PPOModel, env: BaseEnv, exp: TensorDict, params: TrainingParameters, writer: SummaryWriter = None):
    total_loss = torch.tensor(0.0, device=params.device)
    Mij = exp["M_ij"]
    Mji = exp["M_ji"]

    # First, the MSE between Mij and Mji


    # Then the MSE between ze and zd
    z_e = exp["z_e"]
    z_d = exp["z_d"]
    

    return total_loss