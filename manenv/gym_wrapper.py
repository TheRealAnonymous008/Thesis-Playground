import functools
from typing import Any, SupportsFloat
import gymnasium as gym
from gymnasium.spaces import Dict,Discrete

from manenv.core.actor import Actor
from manenv.core.effector import Effector
from manenv.core.monitor import FactoryMetrics
from manenv.solution.assembler_order_selector import AssemblerSelectorModel

from .core.world import World

import numpy as np

from pettingzoo import ParallelEnv, AECEnv


class MARLFactoryEnvironment(ParallelEnv):
    MAX_GAME_STEPS = 1500

    def __init__(self, world : World): 
        """
        A gym wrapper for the Factory environment.

        In particular, it takes in a configured `World` instance and sets it up to be usable in RL training.
        """
        super().__init__()
        self._world : World = world 

        aux_models : list = []
        # Build additional models
        for assembler in self._world.get_all_assemblers():
            aux_models.append(AssemblerSelectorModel(assembler))
    

        self._action_space, self.actor_space = self.build_action_space()
        self.metadata = {
            "name" : "factory"
        }
        self.render_mode = None

        self.reset()


    def build_action_space(self): 
        # All effectors contribute to the action set of the gym wrapper. 
        actor_space : dict[str, Actor] = {}
        action_space : dict = {}
        effectors : list[Effector] = self._world.get_all_effectors()
        for eff in effectors:
            actor_space[eff._id] = eff
            action_space[eff._id] = Discrete(len(eff._action_space))

        return Dict(action_space), actor_space

    def step(self, actions: dict[int, int]) -> tuple[Any, SupportsFloat, bool, bool, dict[str, Any]]:
        """
        Actions are expected to align with the actor_space specified
        """

        self.steps += 1
        if self.steps >= MARLFactoryEnvironment.MAX_GAME_STEPS:
            self.reset()

        for (actor, action) in actions.items():
            self.actor_space[actor].set_action(action)
            
        self._world.update()

        """
        Observations are obtained per actor
        """
        observations = self.get_observation()
        rewards = self._clean_rewards(self._world._monitor.observe())
        trunc = {}
        term = {}
        info = {}

        for agent in self.agents: 
            trunc[agent] = self.steps >= MARLFactoryEnvironment.MAX_GAME_STEPS 
            term[agent] = self.steps >= MARLFactoryEnvironment.MAX_GAME_STEPS
            info[agent] = {}

        return observations, rewards, trunc, term, info 
    
    def _clean_rewards(self, metrics : FactoryMetrics):
        rew : dict[int, int] = {}

        print(metrics)

        for (key, actor) in self.actor_space.items():
            if isinstance(actor, Effector):
                eff : Effector = actor
                rew[key] = 5 * metrics.throughput[eff._assembler._id] + \
                    metrics.inventory[eff._assembler._id] + \
                    metrics.cycle_time[eff._assembler._id] + \
                    metrics.utilization[eff._id] + \
                    4 * metrics.quality + \
                    metrics.customer_service
            else:
                rew[key] = 0

        return rew

    def reset(self, *, seed: int | None = None, options: dict[int, Any] | None = None) -> tuple[Any, dict[str, Any]]:
        np.random.seed(seed)
        self._world.reset()
        self.agents = [x for x in self.actor_space.keys()]
        self.possible_agents = [x for x in self.actor_space.keys()]

        obs = self.get_observation()
        infos = {}
        for ag in self.agents:
            infos[ag] = 1

        self.steps = 0
        return obs, infos

    
    def get_observation(self):
        observations = {}
        for (key, actor) in self.actor_space.items():
            observations[key] = actor.get_observation()

        return observations
    
    @functools.lru_cache(maxsize=None)
    def observation_space(self, agentId : int):
        space = self.actor_space[str(agentId)].get_observation_space()
        return space
    
    @functools.lru_cache(maxsize=None)
    def action_space(self, agentId : int):
        return self._action_space[str(agentId)]

    def render(self):
        return None
    
    def close(self):
        return super().close()