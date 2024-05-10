
from enum import Enum

from manenv.asset_paths import AssetPath
from manenv.effector import Effector
from manenv.product import Product
from manenv.vector import *

class DiscardActions(Enum):
    DISCARD = 1

class Discard(Effector):
    def __init__(self, position : Vector = None):
        super().__init__(DiscardActions, AssetPath.DISCARD, position)
        self._grabbed_product : Product = None

    def _preupdate(self):
        super()._preupdate()
    
    def _update(self):
        match(self._current_action):
            case _: 
                self._assembler.remove_product_in_workspace(self._position)

    def _postupdate(self):
        super()._postupdate()

    