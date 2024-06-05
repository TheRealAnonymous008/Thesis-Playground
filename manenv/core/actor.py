from abc import ABC, abstractmethod


class Actor(ABC):
    @abstractmethod
    def set_action(self, action_code : int):
        pass

    @abstractmethod
    def get_observation(self):
        pass 