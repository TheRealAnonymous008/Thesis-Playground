from __future__ import annotations
from dataclasses import dataclass

from enum import Enum
import numpy as np

class ResourceGenerator:
    def __init__(self):
        pass

    def generate(self, dims : tuple[int, int]) -> np.ndarray:
        resource_map = np.zeros(dims)

        return resource_map