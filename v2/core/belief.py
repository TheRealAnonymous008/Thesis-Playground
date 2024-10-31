from __future__ import annotations
from .agent import Agent
from abc import ABC, abstractmethod

class BaseBeliefInitializer(ABC) :
    """
    Class for seeding all agent belief states. 
    """
    def __init__(self):
        pass 

    @abstractmethod
    def initialize_belief(self, agent : Agent):
        pass 

    def initialize_beliefs(self, agents : list[Agent]):
        for agent in agents: 
            self.initialize_belief(agent)
        return agents 