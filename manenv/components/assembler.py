from manenv.asset_paths import AssetPath
from manenv.component import FactoryComponent
from manenv.effector import Effector
from manenv.product import Product
from manenv.product_utils import *
from manenv.vector import *


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
        self._product_list : dict[int, Product] = {}

        self._product_outputs = []

        for e in effectors:
            e.bind(self)
    
    def update(self):
        for eff in self._effectors:
            eff._preupdate()

        for eff in self._effectors:
            eff._update()

        for eff in self._effectors:
            eff._postupdate()

        self.update_masks()

    def get_product_inventory(self) -> list[Product]:
        return self._cell._products
    
    def place_in_inventory(self, product : Product):
        self._cell._products.append(product)

    def place_in_workspace(self, product: Product, position : Vector):
        if not check_bounds(position, self._workspace_size - product._structure.shape):
            return 
        if not is_region_zeros(product._structure, self._workspace, position):
            return 

        self._product_list[product._id] = product

    def update_masks(self):
        # Flush the workspace and the product mask 
        self._workspace *= 0 
        self._product_mask *= 0

        for product in self._product_list.values():
            cpy = self._workspace.copy()
            new_workspace = place_structure(product._structure, cpy, product._transform_pos)
            mask = ((new_workspace - self._workspace) != 0).astype(int) 
            self._product_mask = (self._product_mask + mask * product._id).astype(int)
            self._workspace = new_workspace

            del cpy

    def get_product_in_workspace(self, position: Vector) -> Product:
        if not check_bounds(position, self._workspace_size):
            raise Exception("Not in bounds")
        
        id = self._product_mask[position[0]][position[1]]
        if id == 0:
            return None 
        
        product = self._product_list[id]
        return product
    
    def delete_product_in_workspace(self, position: Vector) -> Product:
        if not check_bounds(position, self._workspace_size):
            raise Exception("Not in bounds")
        
        id = self._product_mask[position[0]][position[1]]
        if id == 0:
            return None 
        
        product = self._product_list[id]
        self._product_list.pop(id)

        product.delete()
        return product
    
    def release_product_in_workspace(self, position: Vector) -> Product:
        if not check_bounds(position, self._workspace_size):
            raise Exception("Not in bounds")
        
        id = self._product_mask[position[0]][position[1]]
        if id == 0:
            return None 
        
        product = self._product_list[id]
        self._product_list.pop(id)

        product._is_dirty = True 
        self._product_outputs.append(product)
        return product