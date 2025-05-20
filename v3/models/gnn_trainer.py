from .losses import * 
from .model import * 
from .base_env import *
from tests.eval import *

import torch 
from torch.utils.tensorboard import SummaryWriter
from .param_settings import TrainingParameters
from .ppo_trainer import *
from .sac_trainer import *

def train_gnn(model: SACModel | PPOModel, env: BaseEnv, exp: TensorDict, params: TrainingParameters, writer: SummaryWriter = None):
    total_loss = torch.tensor(0.0, device=params.device)
    
    # Compute the JSD loss between each agent pair. 

    return total_loss