import functools
from typing import Any, SupportsFloat
import gymnasium as gym
from gymnasium.spaces import Dict,Discrete

from manenv.core.actor import Actor
from manenv.core.effector import Effector
from manenv.core.monitor import FactoryMetrics

from .core.world import World

import numpy as np

from pettingzoo import ParallelEnv, AECEnv


class MARLFactoryEnvironment(ParallelEnv):
    MAX_GAME_STEPS = 10000

    def __init__(self, world : World): 
        """
        A gym wrapper for the Factory environment.

        In particular, it takes in a configured `World` instance and sets it up to be usable in RL training.
        """
        super().__init__()
        self._world : World = world 
        self._action_space, self.actor_space = self.build_action_space()
        self.metadata = {
            "name" : "factory"
        }
        self.render_mode = None

    def build_action_space(self): 
        # All effectors contribute to the action set of the gym wrapper. 
        actor_space : dict[int, Actor] = {}
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
            info[agent] = False

        return observations, rewards, trunc, term, info 
    
    def _clean_rewards(self, metrics : FactoryMetrics):
        rew : dict[int, int] = {}

        for (key, actor) in self.actor_space.items():
            # TODO: Insert calculations for single reward here
            if actor is Effector:
                eff : Effector = actor
                rew[key] = metrics.throughput[eff._assembler._id]

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
        space = self.actor_space[agentId].get_observation_space()
        return space
    
    @functools.lru_cache(maxsize=None)
    def action_space(self, agentId : int):
        return self._action_space[agentId]

    def render(self):
        return None
    
    def close(self):
        return super().close()