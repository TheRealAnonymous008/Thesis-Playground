from __future__ import annotations
from dataclasses import dataclass

from enum import Enum
import numpy as np


@dataclass
class LocalObservation:
    """
    Contains attributes relevant to local observation of an agent 
    Please note that the contents of the observation space should not be modified in any way! 

    :param nearby_agents: A list of all nearby agents and their location relative to the current agent 


    """
    nearby_agents : np.ndarray
    
    def neighbors(self, id : int) -> list[int]:
        """
        Returns a list of id's of all communicable agents (defined as being adjacent to the agent)
        """
        center_x, center_y = (s // 2 for s in self.nearby_agents.shape)
        surrounding = self.nearby_agents[center_x-1:center_x+2, center_y-1:center_y+2].ravel()
        return [val for val in surrounding if val != 0 and val != id]