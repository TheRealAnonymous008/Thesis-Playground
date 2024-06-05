from typing import Any, SupportsFloat
import gymnasium as gym

from .core.world import World

# TODO: Implement this
class MARLFactoryEnvironmen(gym.Env):
    def __init__(self, world : World): 
        self._world = world 

    def step(self, action: Any) -> tuple[Any, SupportsFloat, bool, bool, dict[str, Any]]:
        self._world.update()
        return super().step(action)
    
    def reset(self, *, seed: int | None = None, options: dict[str, Any] | None = None) -> tuple[Any, dict[str, Any]]:
        return super().reset(seed=seed, options=options)
    
    def render(self):
        return None
    
    def close(self):
        return super().close()
    