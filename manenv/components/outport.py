
from manenv.asset_paths import AssetPath
from manenv.core.component import FactoryComponent
from manenv.core.product import Product


class Outport(FactoryComponent):
    """
    Recives products and updates the factory's stock based on what it receives
    """
    def __init__(self):
        super().__init__(AssetPath.OUTPORT)
    
    def update(self):
        super().update()

