from .losses import * 
from .model import * 
from .base_env import *
from tests.eval import *

import torch 
from torch.utils.tensorboard import SummaryWriter
from .param_settings import TrainingParameters

def train_filter(model: SACModel | PPOModel, env: BaseEnv, exp: TensorDict, params: TrainingParameters, writer: SummaryWriter = None):
    total_loss = torch.tensor(0.0, device=params.device)
    

    return total_loss