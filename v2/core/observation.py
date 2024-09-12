from __future__ import annotations
from dataclasses import dataclass

from enum import Enum
import numpy as np


@dataclass
class LocalObservation:
    """
    Contains attributes relevant to local observation of an agent 

    `nearby_agents`: A list of all nearby agents and their location relative to the current agent 

    `resources`: A list of the resourcee types available in an area
    """
    nearby_agents : np.ndarray
    resource_types : np.ndarray 
    
    