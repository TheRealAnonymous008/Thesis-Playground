
from __future__ import annotations
from dataclasses import dataclass, field

from abc import abstractmethod, ABC
from typing import TYPE_CHECKING

import numpy as np
from manenv.components.assembler import Assembler
from manenv.core.demand import Order
from manenv.core.effector import Effector
from manenv.core.product import Product

if TYPE_CHECKING:
    from world import World


@dataclass
class FactoryMetrics:
    throughput :dict[int, float] = field(default_factory=lambda: {})
    utilization : dict[int, float] = field(default_factory=lambda : {})
    inventory: dict[int, float] = field(default_factory=lambda : {})
    cycle_time: dict[int, float] = field(default_factory=lambda : {})  
    lead_time: int = 0
    customer_service : float = 0
    quality : float = 0

@dataclass
class MonitorState: 
    assembler_output_buffer : dict[int, list[Product]] =  field(default_factory=lambda: {})

class FactoryMonitor(ABC): 
    def __init__(self):
        """
        The factory monitor measures the performance of the current world 
        """

        self._world : World = None 
        self._state : MonitorState = None 

    def reset(self):
        assert(self._world != None)
        self._state = MonitorState()

        for assembler in self._world.get_all_assemblers():
            self._state.assembler_output_buffer[assembler._id] = []


    def bind(self, world : World):
        self._world = world 
        self.reset()

    @abstractmethod
    def observe(self) -> FactoryMetrics:
        # Observe returns a dictionary
        assert(self._world != None)
        self.reset()
        pass 

    
    def _probe_assembler(self, assembler: Assembler):
        """
        The following method simply checks if the assembler is recognized in the state object 
        and if not adds it to the state. 
        """
        if not assembler._id in self._state.assembler_output_buffer:
            self._state.assembler_output_buffer[assembler._id] = [] 


class DefaultFactoryMonitor(FactoryMonitor):
    def __init__(self):
        super().__init__()

    def observe(self) -> FactoryMetrics:
        super().observe()
        throughput = {}
        utilization = {}
        inventory = {}
        cycle_time = {}
        lead_time = 0 
        customer_service = 0
        quality = 0

        for assembler in self._world.get_all_assemblers():
            id = assembler._id
            throughput[id] = self._process_assembler_throughput(assembler)
            utilization[id] = self._process_assembler_utilization(assembler)
            inventory[id] = self._process_assembler_inventory_cost(assembler)
            cycle_time[id] = self._process_assembler_cycle_time(assembler)
        
        for (k, order) in self._world._demand._orders.items():
            lead_time += self._process_lead_time_score(order) 
            customer_service += self._process_service_level(order)
            quality += self._process_quality_level(order)
        
        if len(self._world._demand._orders) > 0:
            num_orders = len(self._world._demand._orders)
            lead_time /= num_orders
            customer_service /= num_orders
            quality /= num_orders

        return FactoryMetrics(
            throughput=throughput,
            utilization=utilization,
            inventory=inventory,
            cycle_time=cycle_time,
            lead_time=lead_time,
            customer_service=customer_service,
            quality=quality
        )

    def _process_assembler_throughput(self, assembler: Assembler) -> float: 
        """
        Throughput is defined as the number of outputs per time unit.
        """
        products = assembler._product_outputs
        
        return max(0, len(products) - len(self._state.assembler_output_buffer[assembler._id]))
    
    def _process_assembler_utilization(self, assembler: Assembler) -> float:
        """
        Utilization is defined as the fraction of time the station is busy. 

        The busyness of a station is defined as the proportion of its effectors that did not perform the Idle Action
        """

        effectors : list[Effector] = assembler._effectors
        ctr : int = 0

        for eff in effectors:
            ctr += (not eff._is_idle)
        
        return 1.0 * ctr / len(effectors)
    
    def _process_assembler_inventory_cost(self, assembler: Assembler) -> float:
        """
        Inventory for an assembler is defined as the number of products in its staging area at the current moment 
        """
        inventory_cost = 0
        if len(assembler._staging_area) == 0:
            inventory_cost += 2
        else: 
            inventory_cost = 1.0 / len(assembler._staging_area) 

        dims = assembler._workspace_size
        nonempty_cells = np.count_nonzero(assembler._workspace != 0) / (dims[0] * dims[1])
        inventory_cost -= nonempty_cells

        return inventory_cost 
    
    
    def _process_assembler_cycle_time(self, assembler: Assembler) -> float: 
        """
        Cycle time is defined as the amount of time it took to finish a particular job. 

        The cycle time reward is the reciprocal of Cycle Time. If an order was not completed, its cycle time reward is treated as zero.
        """
        orders : list[Order] = assembler._completed_order_buffer
        cycle_score = 0

        for order in orders: 
            cycle_time = order.get_cycle_time()
            if cycle_time > 0:
                cycle_score += 1.0 / cycle_time

        return cycle_score

    def _process_lead_time_score(self, order : Order) -> float:
        """
        Lead time is defined as the time quoted to the customer. For each order in the demand module, we penalize for high lead times
        """
        return 1.0 / order._due_date
    
    def _process_service_level(self, order : Order) -> float: 
        """
        Service is defined as the number of products delivered on time. Here, we also quantify the lateness.
        """
        if order.is_satisfied():
            return order._due_date - order._satisfied_time

        return 0

    def _process_quality_level(self, order : Order) -> float: 
        """
        Quality is defined as how well the product given matches the product ordered. Here, distance is defined using something analogous to
        Jaccard Index
        """ 
        if not order.is_satisfied():
            return 0
        
        return Product.compare(order._product, order._satisfied_product)