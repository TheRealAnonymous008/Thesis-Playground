from __future__ import annotations
from abc import ABC 
from .agent import Agent
from .env_params import * 

import numpy as np 

class BaseSkillInitializer(ABC):
    def __init__(self, num_skills = PRODUCT_TYPES): 
        self._num_skills =  num_skills
    
    def forward(self, agent : Agent ):
        """
        Initialize an agent's skills.
        """
        pass 

class DefaultSkillInitializer(BaseSkillInitializer):
    def __init__(self, num_skills = 4): 
        super().__init__(num_skills)

        self.distributions : list[callable] = []
        self._initialize_distributions()

    def _initialize_distributions(self):
        for _ in range(self._num_skills):
            self.distributions.append(lambda : np.clip(np.random.uniform(0.25, 0.75), 0.25, 0.75))

    def forward(self, agent: Agent):
        skill  = []
        for i in range(self._num_skills):
            skill.append(self.distributions[i]())
        
        agent._current_state.skills = np.array(skill)