from __future__ import annotations
from typing import Tuple

from abc import abstractmethod, ABC
from .vector import *
from .asset_paths import AssetPath
import numpy as np

from typing import TYPE_CHECKING
from .product_utils import * 
from .product import Product
if TYPE_CHECKING: 
    from .world import World, WorldCell
    from .effector import Effector

class FactoryComponent(ABC): 
    """
    Abstract class for factory components
    """
    def __init__(self, asset : str = ""):
        """
        `world` - the world instance that holds this factory component 

        `asset` - a path to the image associated with this asset. If empty, defaults to no asset. 
        """
        self._cell : WorldCell | None = None 
        self._world : World = None
        self._asset : str = asset

    def bind(self, world):
        self._world = world 

    def place(self, cell: WorldCell):
        self._check_is_bound()
        if self._cell == cell:
            return 
        
        if self._cell != None: 
            self._cell.place_component(None)

        self._cell = cell 
        self._cell.place_component(self)

        self._on_place()

    def _on_place(self):
        pass 

    @abstractmethod
    def update(self):
        self._check_is_bound()
        pass 

    def _check_is_bound(self):
        if self._world == None: 
            raise Exception("World is not initialized for this component ")

class Spawner(FactoryComponent):
    """
    Spawns product objects based on a provided template product  
    """
    def __init__(self, product : Product):
        """
        `product`: The product that this spawner will spawn 
        """
        super().__init__(AssetPath.SPAWNER)
        self._product = product
    
    def update(self):
        super().update()
        if len(self._cell._products) == 0:
            self._cell.place_product(self._product.copy())


class Conveyor(FactoryComponent): 
    """
    System for transporting products across the factory. 
    """
    def __init__(self, weights : np.ndarray):
        """
        `weights`: a 3x3 transition matrix that represent how a product will flow through this cell. 
        The middle entry is ignored 

        An integer n means that n of that product will be allowed first from that side (if applicable)
        Positive n means the product is moved away from the cell
        Negative n means the product is moved towards the cell. 
        """
        assert(weights.shape == (3, 3))
        self._weights = weights

        """
        curr in and curr out follow numpad order
        1 2 3
        4 5 6
        7 8 9
        """

        super().__init__("")

    def _on_place(self):
        self._initialize_in_out_ports()

    def _initialize_in_out_ports(self):
        self._curr_in = -1
        self._curr_out = -1
        self._inports = []
        self._outports = []

        for y in range(0, 3):
            for x in range(0, 3):
                idx = 3 * y + x
                offset = make_vector(x - 1, y - 1)
                if self._world.get_cell(offset + self._cell._position) == None:
                    continue 

                if self._weights[x][y] < 0:
                    self._inports.append(idx)
                elif self._weights[x][y] > 0:
                    self._outports.append(idx)

        np.random.shuffle(self._inports)
        np.random.shuffle(self._outports)

    def update(self): 
        super().update() 
        if len(self._inports) > 0: 
            self._curr_in = (self._curr_in + 1) % len(self._inports)
            idx = self._inports[self._curr_in]
            input_offset = make_vector(idx % 3 - 1, idx // 3 - 1)
            input_vec = self._cell._position + input_offset
            src_cell = self._world.get_cell(input_vec)

            if src_cell != None:
                for product in src_cell._products:
                    if self._cell.is_product_placed(product):
                        src_cell.remove_product(product)
                        self._cell.place_product(product, input_offset)

        if len(self._outports) > 0:
            self._curr_out = (self._curr_out + 1) % len(self._outports)
            idx = self._outports[self._curr_out]
            output_offset = make_vector(idx % 3 - 1, idx // 3 - 1)
            output_vec = self._cell._position + output_offset
            dest_cell = self._world.get_cell(output_vec)
            
            # Move current product out of this cell 
            if dest_cell != None: 
                P = self._cell._products
                for product in P:
                    if self._cell.is_product_placed(product):
                        self._cell.remove_product(product)
                        dest_cell.place_product(product, -output_offset)

class Assembler(FactoryComponent):
    """
    Component for converting the provided input products into an output product

    `workspace_size`: the size of the workspace supported by this product

    `effectors`: the list of effectors that can be used by this assembler unit. 
    """
    def __init__(self, workspace_size : Vector, effectors : list[Effector] = []):
        super().__init__(AssetPath.ASSEMBLER)
        
        self._workspace_size = workspace_size
        self._workspace = np.zeros(self._workspace_size, dtype=int)
        self._effectors = effectors

        self._product_mask = np.zeros(self._workspace_size, dtype = int)

        for e in effectors:
            e.bind(self)
    
    def update(self):
        pass

    def get_product_inventory(self) -> list[Product]:
        return self._cell._products
    
    def place_in_inventory(self, product : Product):
        self._cell._products.append(product)

    def place_in_workspace(self, product: Product, position : Vector):
        if not check_bounds(position, self._workspace_size - product._structure.shape):
            return 

        new_workspace = place_structure(product._structure, self._workspace.copy(), position)
        mask = ((new_workspace - self._workspace) != 0).astype(int) 
        self._product_mask = (self._product_mask + mask * product._id).astype(int)

        self._workspace = new_workspace

    def get_product_in_workspace(self, position: Vector, remove = True ) -> Product:
        if not check_bounds(position, self._workspace_size):
            raise Exception("Not in bounds")
        
        id = self._product_mask[position[0]][position[1]]
        if id == 0:
            return None 
        
        mask = (self._product_mask == id).astype(int)
        product = Product((self._workspace.copy() * mask), id = id)

        if remove: 
            self._workspace = (1 - mask) * self._workspace

        return product


        
