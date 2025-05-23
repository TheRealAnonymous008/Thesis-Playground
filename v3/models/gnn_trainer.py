from .losses import * 
from .model import * 
from .base_env import *
from tests.eval import *

import torch 
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from .param_settings import TrainingParameters

def train_gnn(model: SACModel | PPOModel, env: BaseEnv, exp: TensorDict, params: TrainingParameters, writer: SummaryWriter = None):
    total_loss = torch.tensor(0.0, device=params.device)
    Mij = exp["M_ij"]
    Mji = exp["M_ji"]

    # First, the MSE between Mij and Mji
    mse_m = F.mse_loss(Mij, Mji)
    total_loss += mse_m

    # Then the MSE between ze and zd
    z_e = exp["z_e"]
    z_d = exp["z_d"]
    mse_z = F.mse_loss(z_e, z_d)
    total_loss += mse_z

    # Log losses to TensorBoard
    if writer is not None:
        writer.add_scalar('Decoder/Relations', mse_m.item(), params.global_steps)
        writer.add_scalar('Decoder/Beliefs', mse_z.item(), params.global_steps)

    return total_loss