from world import World
from dataclasses import dataclass


from abc import abstractmethod, ABC

@dataclass
class FactoryMetrics:
    throughput : float
    # utilization: float
    # inventory: float
    # cycle_time: int  
    # lead_time: int
    # customer_service : float
    # quality : float



class FactoryMonitor(ABC): 
    def __init__(self):
        """
        The factory monitor measures the performance of the current world 
        """

        self._world : World = None 

    def bind(self, world : World):
        self._world = world 

    @abstractmethod
    def observe(self) -> FactoryMetrics:
        # Observe returns a dictionary
        assert(self._world != None)
        pass 


class DefaultFactoryMonitor(FactoryMonitor):
    def observe(self) -> FactoryMetrics:
        super().observe()
        pass 

    def _calculate_throughput(self): 
        """
        Throughput is defined as the number of outputs per time unit.
        """

        