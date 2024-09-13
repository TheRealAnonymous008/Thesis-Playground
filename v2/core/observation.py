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
    
    @property
    def neighbors(self) -> np.ndarray[int]:
        """
        Returns a list of id's of all communicable agents 
        """
        center_x, center_y = self.nearby_agents.shape[0] // 2, self.nearby_agents.shape[1] // 2
        surrounding = self.nearby_agents[center_x-1:center_x+2, center_y-1:center_y+2]
        return surrounding[surrounding != 0].flatten().tolist()