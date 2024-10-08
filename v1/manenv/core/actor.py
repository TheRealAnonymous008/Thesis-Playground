from abc import ABC, abstractmethod

from manenv.core.idpool import IDPool


class Actor(ABC):
    def __init__(self):
        # ID has to be a string because of pettingzoo / stablebaselines3 specs. 
        self._id = str(IDPool.get())
        
    @abstractmethod
    def set_action(self, action_code : int):
        pass

    @abstractmethod
    def get_observation(self):
        pass 
    
    @abstractmethod
    def get_observation_space(self):
        pass

    def unplace(self):
        IDPool.pop(self._id)