import numpy as np
from core.agent import Agent
from core.models import *


class EnergyModel(BaseDynamicsModel):
    def __init__(self):
        pass 

    def forward(self, world : BaseWorld) -> float: 
        """
        Compute the energy consumption of an egent and update its current state.
        """
        for agent in world.agents: 
            action = agent.action
            total_energy_consumption = 0
            
            if action.movement != None and agent.has_moved: 
                e = np.random.normal(0.5, 0.25)
                total_energy_consumption += max(0.1, e)

            agent._current_state.current_energy -= total_energy_consumption
            agent._current_state.current_energy = max(0, agent._current_state.current_energy)
    