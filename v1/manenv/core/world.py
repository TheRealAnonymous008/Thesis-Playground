from __future__ import annotations
import copy
import numpy as np 
from typing import Tuple

from typing import TYPE_CHECKING

from manenv.components.assembler import Assembler
from .service import *
from .demand import * 
from .inventory import * 
from .monitor import *

if TYPE_CHECKING: 
    from .component import *
    from .product import *
    from .monitor import * 

from ..utils.vector import * 

class WorldCell: 
    """
    The basic unit within the factory. The world cell may contain:
    - a factory component
    - a product
    - a robot 
    """

    def __init__(self, position : Vector, capacity : int = 5):
        """
        `position` - the position associated with this cell
        `capacity` - the max number of products that can be in this cell
        """

        self._factory_component : FactoryComponent | None = None 
        self._products : dict[int, Product] = {}
        # is_placed stores a series of booleans corresponding to whether or not a product is at the edge
        # of this cell or not 

        self._dirty_set : set[int] = set()
        self._position : Vector = position
        self._capacity : int = capacity
        self.reset()

    def reset(self):
        if self._factory_component != None: 
            self._factory_component.reset()
        
        self.clear_products()
        self._dirty_set.clear()

    def is_full(self):
        return len(self._products) >= self._capacity

    def place_component(self, cmp : FactoryComponent):
        if self._factory_component == cmp: 
            return
        self._factory_component = cmp
        cmp.place(self)

    def place_product(self, product: Product, position : Vector | None = make_vector(0, 0)):
        self._products[product._id] = product
        if position is None:
            return 
        
        if position[0] != 0 or position[1] != 0:
            self._dirty_set.add(product._id)


    def is_product_placed(self, product : Product) -> bool:
        return not (product._id in self._dirty_set)
    
    def update_place_status(self, product : Product):
        if product._id in self._dirty_set: 
            self._dirty_set.remove(product._id) 

    def remove_product(self, product : Product):
        """
        Remove product with specified `id` from the products on this cell
        """
        p = self._products[product._id]
        if self.is_product_placed(p):
            self._products.pop(p._id)

    def clear_products(self):
        for i in self._products.values():
            IDPool.pop(i._id)

        self._products.clear()

    def get_product_list(self):
        return copy.copy(self._products).values()
        
class World: 
    """
    Contains information about the smart factory environment 
    """
    def __init__(self,
                shape : Tuple, 
                 demand: DemandSimulator = DefaultDemandSimulator(), 
                 inventory : Inventory = DefaultInventorySystem(),
                 service: ServiceModule = DefaultServiceModule(),
                 monitor : FactoryMonitor = DefaultFactoryMonitor()
        ): 
        """
        `shape` - the dimensions of the environment in (width, height) format 
        `demand` - the simulator for demand
        `inventory` - the inventory module
        `service` - the service module
        `monitor` - factory monitor that measures performance metrics per time step 
        """
        self._shape : Tuple = shape 
        self._map : list[list[WorldCell]] = [[WorldCell(position=make_vector(x, y)) for x in range(shape[0])] for y in range(shape[0])]
        self._demand : DemandSimulator = demand
        self._inventory : Inventory = inventory

        self._service_module : ServiceModule = service
        self._service_module.bind(self)

        self._monitor: FactoryMonitor = monitor 
        self._monitor.bind(self)

        self._time_step : int  = 0 

    def _width(self):
        return self._shape[0]
    
    def _height(self):
        return self._shape[1]
    
    def reset(self):
        self._time_step = 0

        for M in self._map: 
            for r in M: 
                r.reset()
        
        self._demand.reset()
        self._inventory.reset()
        self._monitor.reset()

    def update(self): 
        self._time_step += 1
        # Update the current demand system
        self._demand.update()
        self._service_module.update()

        # All dirty components are pushed to the cell's center
        for x in range(self._shape[0]):
            for y in range(self._shape[1]):
                cell = self._map[x][y]
                for prod in cell._products.values():
                    if not cell.is_product_placed(prod):
                        cell.update_place_status(prod)

        # Update all components
        for x in range(self._shape[0]):
            for y in range(self._shape[1]):
                if self._map[x][y]._factory_component != None: 
                    self._map[x][y]._factory_component.update()
    
    def get_cell(self, v : Vector) -> WorldCell | None:
        # Vector written as (x, y) 
        if v[0] < 0 or v[1] < 0 or v[0] >= self._shape[0] or v[1] >= self._shape[1]:
            return None 
        
        return self._map[v[1]][v[0]]
    
    def get_all_effectors(self) -> list[Effector]: 
        effectors : list[Effector] = []
        for x in range(self._shape[0]):
            for y in range(self._shape[1]):
                cell = self._map[x][y] 
                if isinstance(cell._factory_component, Assembler):
                    assembler : Assembler = cell._factory_component
                    effectors.extend(assembler._effectors)

        return effectors
    
    def get_all_assemblers(self) -> list[Assembler]:
        assemblers : list[Assembler] = []
        for x in range(self._shape[0]):
            for y in range(self._shape[1]):
                cell = self._map[x][y]
                if isinstance(cell._factory_component, Assembler):
                    assembler : Assembler = cell._factory_component
                    assemblers.append(assembler)

        return assemblers
    
    def place_component(self, pos: Vector, cmp : FactoryComponent):
        cmp.bind(self)
        self.get_cell(pos).place_component(cmp)
        
    def place_product(self, pos: Vector, product : Product):
        self.get_cell(pos).place_product(product)
    
    def log_world_status(self, verbose = False):
        print("====== Demand ======")
        print("Total Orders: ", len(self._demand._orders))
        if verbose:
            for order in self._demand._orders:
                print(order)

        print("====================")

        print("====== Inventory ===== ")
        print("Total Cost: ", self._inventory.compute_storage_cost())
        print("Total Items: ", len(self._inventory._product_inventory))

        if verbose: 
            for prod in self._inventory._product_inventory:
                print(prod)