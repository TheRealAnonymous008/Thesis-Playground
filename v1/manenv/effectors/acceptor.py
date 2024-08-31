
from enum import Enum

from manenv.asset_paths import AssetPath
from manenv.core.effector import Effector
from manenv.core.product import Product
from manenv.utils.vector import *

class AcceptorActions(Enum):
    ACCEPT = 1

class Acceptor(Effector):
    def __init__(self, position : Vector = None):
        super().__init__(AcceptorActions, AssetPath.ACCEPTOR, position)
        self._grabbed_product : Product = None

    def _preupdate(self):
        super()._preupdate()
    
    def _update(self):
        match(self._current_action):
            case _: 
                self._assembler.release_product_in_workspace(self._position)

    def _postupdate(self):
        super()._postupdate()

    def reset(self):
        pass

    