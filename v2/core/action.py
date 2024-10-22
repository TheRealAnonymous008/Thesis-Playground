from __future__ import annotations
from dataclasses import dataclass

from enum import Enum
import numpy as np


from .direction import Direction
from .resource import _ResourceType


@dataclass
class ActionInformation:
    """
    Contains attributes and misc. information about an agent's actions.

    Base Classes should override this 
    """

    def reset(self):
        pass 