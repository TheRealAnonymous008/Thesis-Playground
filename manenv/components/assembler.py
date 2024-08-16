from manenv.asset_paths import AssetPath
from manenv.core.component import FactoryComponent
from manenv.core.effector import Effector
from manenv.core.product import Product
from manenv.utils.product_utils import *
from manenv.utils.vector import *
from manenv.core.demand import Order

class Assembler(FactoryComponent):
    """
    Component for converting the provided input products into an output product

    `workspace_size`: the size of the workspace supported by this product

    `effectors`: the list of effectors that can be used by this assembler unit. 

    `staging_size`: the number of products that can be in the staging area at most
    """
    def __init__(self, 
                 workspace_size : Vector, 
                 effectors : list[Effector] = [],
                 staging_size : int = -1):
        super().__init__(AssetPath.ASSEMBLER)
        
        self._workspace_size = workspace_size.copy()
        self._effectors = effectors
        self._staging_size = staging_size
        self._completed_order_buffer : list[Order] = []

        self.reset()

        for e in effectors:
            e.bind(self)
    
    def update(self):
        self._completed_order_buffer = []
        for product in self._product_list.values():
            product._transform_pos = np.clip(product._transform_pos, (0, 0), self._workspace_size  - VectorBuiltin.ONE_VECTOR)

        for eff in self._effectors:
            eff._preupdate()

        for eff in self._effectors:
            eff._update()

        for eff in self._effectors:
            eff._postupdate()

        self.update_staging()

    def reset(self):
        self._workspace = np.zeros(self._workspace_size, dtype=int)
        self._product_mask = np.zeros(self._workspace_size, dtype = int)
        self._product_list : dict[int, Product] = {}

        # The staging area is where we first push our products.
        # We then push it onto the product outputs array to be delivered outside
        # The assembler. This way, we can match orders in the queue with staged products
        self._inventory : list[Product] = []
        self._staging_area : list[Product]= []
        self._product_outputs : list[Product] = []

        self._job_queue : list[Order] = []

        # Each effector in this assembler must be reset
        for eff in self._effectors:
            eff.reset()

    def add_order_to_queue(self, order : Order):
        self._job_queue.append(order)
    
    def place_in_inventory(self, product : Product):
        self._cell.place_product(product, None)
        self._inventory.append(product)

    def get_from_inventory(self) -> Product | None:
        if len(self._inventory) > 0:
            prod = self._inventory.pop(0)
            self._cell.remove_product(prod)
            return prod
        return None

    def place_in_workspace(self, product: Product, position : Vector):
        if not check_bounds(position, self._workspace_size - product._structure.shape):
            return 
        if not is_region_zeros(product._structure, self._workspace, position):
            return 
        
        structure = (product._structure != 0) * product._id 
        self._product_mask = place_structure(structure, self._product_mask, position)
        self._product_list[product._id] = product

        self._workspace = place_structure(product._structure, self._workspace, position)
        product._transform_pos = position

    def get_product_in_workspace(self, position: Vector) -> Product | None:
        if not check_bounds(position, self._workspace_size):
            return None
        
        id = self._product_mask[position[0]][position[1]]
        if id == 0:
            return None 
        
        return self._product_list[id]
    
    def _pop_from_workspace(self, position: Vector) -> Product:
        product = self.get_product_in_workspace(position)

        if product == None:
            return 

        self._product_list.pop(product._id)    
        self._workspace = (self._product_mask != product._id) * self._workspace
        self._product_mask = (self._product_mask != product._id) * self._product_mask
         
        return product


    def delete_product_in_workspace(self, position: Vector) -> Product:
        product = self._pop_from_workspace(position)
        if product == None: 
            return 
        
        product.delete()
        return product
    
    def release_product_in_workspace(self, position: Vector) -> Product:
        product = self.get_product_in_workspace(position)
        if product == None: 
            return 
        if len(self._staging_area) >= self._staging_size and self._staging_size > 0: 
            return 
        
        self._pop_from_workspace(position)
        product._is_dirty = True 
        
        self._staging_area.append(product)
        return product
    
    def update_staging(self):
        for i in range(len(self._staging_area)):
            p = self._staging_area[i] 
            idx = self._assess_product_with_queue(p)
            self._product_outputs.append(p)
            
            if idx < 0:
                continue 
            
            order = self._job_queue.pop(idx)
            self._completed_order_buffer.append(order)
            self._staging_area.pop(i)
            

    def _assess_product_with_queue(self, product : Product) -> int:
        # TODO: Implement this during model implementation phase. 
        # It is recommended that any function for this should take into account
        # The product's match to a specified order on the list
        # The current status of product inventory
        # The current status of the staging area (is it being filled)

        """
        This determines if the product can be matched to a job in the queue.
        """
        if len(self._job_queue) == 0:
            return -1

        return 0