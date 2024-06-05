from typing import Any, SupportsFloat
import gymnasium as gym
from gymnasium.spaces import Dict,Discrete

from manenv.core.actor import Actor
from manenv.core.effector import Effector

from .core.world import World

# TODO: Implement this
class MARLFactoryEnvironment(gym.Env):
    def __init__(self, world : World): 
        """
        A gym wrapper for the Factory environment.

        In particular, it takes in a configured `World` instance and sets it up to be usable in RL training.
        """
        self._world = world 
        self.action_space, self.actor_space = self.build_action_space()

    def build_action_space(self): 
        # All effectors contribute to the action set of the gym wrapper. 
        actor_space : dict[int, Actor] = {}
        action_space : dict = {}
        effectors : list[Effector] = self._world.get_all_effectors()
        for i, eff in enumerate(effectors):
            actor_space[i] = eff
            action_space[i] = Discrete(len(eff._action_space))

        return Dict(action_space), actor_space
    
    def build_observation_space(self):
        pass 


    def step(self, actions: dict[int, int]) -> tuple[Any, SupportsFloat, bool, bool, dict[str, Any]]:
        """
        Actions are expected to align with the actor_space specified
        """
        for (actor, action) in actions.items():
            self.actor_space[actor].set_action(action)

        self._world.update()

        return (1)
    
    def reset(self, *, seed: int | None = None, options: dict[str, Any] | None = None) -> tuple[Any, dict[str, Any]]:
        return super().reset(seed=seed, options=options)
    
    def render(self):
        return None
    
    def close(self):
        return super().close()