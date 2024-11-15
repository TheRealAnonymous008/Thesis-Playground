from __future__ import annotations
from typing import TYPE_CHECKING
from core.agent import *
from .sar_agent import *
from sar.sar_env_params import *

class SARTraitSampler:
    """
    Utility class to generate a heterogeneous swarm of agents with complete traits 
    """

    def __init__(self): 
        pass 

    def generate(self, n_agents : int, device : str = "cpu") -> list[SARAgent]:
        """
        Generate the specified amount of agents. Any hyperparameters such as target population distribution 
        should be specified as part of this class' specification
        """
        agents : list[SARAgent] = []
        for _ in range(n_agents):
            agent = SARAgent()
            traits = SARAgentTraits()

            # Sample traits here
            visibility = int(np.random.uniform(1, MAX_VISIBILITY))
            energy_capacity = np.clip(np.random.normal(100, 20), a_min=10, a_max = MAX_ENERGY)
            max_slope = np.clip(np.random.normal(1, 0.5), a_min = 0.01, a_max = None)

            traits._tensor = torch.tensor([3, energy_capacity, max_slope], dtype = torch.float32)
            agent._traits = traits 

            agent.to(device)

            agents.append(agent)

        return agents