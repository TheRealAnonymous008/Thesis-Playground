
from manenv.asset_paths import AssetPath
from manenv.component import FactoryComponent
from manenv.product import Product


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
