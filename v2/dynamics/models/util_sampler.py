

import numpy as np 

from core.agent import Agent
from core.env_params import *


TOTAL_FEATURES = PRODUCT_TYPES + RESOURCE_TYPES
class UtilitySampler:
    def __init__(self, dims = TOTAL_FEATURES ):
        self.dims = dims 
    
    def forward(self, agent : Agent): 
        pass 