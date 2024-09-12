from __future__ import annotations

import numpy as np
from enum import Enum

from .observation import LocalObservation

from .action import *

@dataclass
class AgentSate:
    """
    Contains attributes and misc. information about an agent's internal state.

    `current_energy` - current energy of the agent
    """
    current_energy : float = 0

    def can_move(self):
        return self.current_energy > 0

class Agent:
    def __init__(self):
        """
        Initializes a simple agent 
        """
        self._id = -1

        self._previous_position : np.ndarray[int] | None = None 
        self._current_position : np.ndarray[int] | None =  None 

        self._current_observation : LocalObservation = None
        self._current_action : ActionInformation = ActionInformation()
        self._current_state :  AgentSate = AgentSate()

        # Attributes of the agent. 
        self._visibility_range : int = 3
        self._energy_capacity : float = 100.0
        self._carrying_capacity : float = 100.0

        self.reset()
        
    def reset(self):
        """
        Reset the agent
        """
        self._previous_position = None 
        self._current_state.current_energy = self._energy_capacity

    def move(self, dir : Direction | int):
        """
        Moves an agent along a specified direction. 
        The direction is either a Direction instance or an integer associated with a Direction value.
        """
        if not self._current_state.can_move():
            return 

        if type(dir) is Direction: 
            val = dir.value
        else: 
            val = dir 

        match(val):
            case Direction.NORTH.value: 
                self._current_action.movement = Direction.NORTH

            case Direction.SOUTH.value: 
                self._current_action.movement = Direction.SOUTH

            case Direction.EAST.value: 
                self._current_action.movement = Direction.EAST
            
            case Direction.WEST.value:
                self._current_action.movement = Direction.WEST

            case _: 
                raise Exception(f"Invalid direction specified {val}")

    def pick_up(self, dir : Direction | int ): 
        """
        Pick up a resource adjacent to this agent 
        """
        if not self._current_state.can_move():
            return 

        if type(dir) is Direction: 
            val = dir.value
        else: 
            val = dir 

        match(val):
            case Direction.NORTH.value: 
                self._current_action.pick_up = Direction.NORTH

            case Direction.SOUTH.value: 
                self._current_action.pick_up = Direction.SOUTH

            case Direction.EAST.value: 
                self._current_action.pick_up = Direction.EAST
            
            case Direction.WEST.value:
                self._current_action.pick_up = Direction.WEST

            case _: 
                raise Exception(f"Invalid direction specified {val}")

    def put_down(self, dir : Direction | int ): 
        """
        Put down the held resource to somewhere adjacent to the agent.
        """
        if not self._current_state.can_move():
            return 

        if type(dir) is Direction: 
            val = dir.value
        else: 
            val = dir 

        match(val):
            case Direction.NORTH.value: 
                self._current_action.pick_up = Direction.NORTH

            case Direction.SOUTH.value: 
                self._current_action.pick_up = Direction.SOUTH

            case Direction.EAST.value: 
                self._current_action.pick_up = Direction.EAST
            
            case Direction.WEST.value:
                self._current_action.pick_up = Direction.WEST

            case _: 
                raise Exception(f"Invalid direction specified {val}")
            
    def reset_for_next_action(self):
        """
        Resets the agent for a new action
        """
        self._current_action.reset()

    def set_observation(self, observation : LocalObservation):
        self._current_observation = observation

    def get_observation(self) -> LocalObservation:
        return self._current_observation

    def bind_to_world(self, world_id : int):
        """
        Set the agent's ID to be that of the ID assigned to it by the world
        """
        self._id = world_id

    def get_id(self) -> int:
        """
        Return the ID of this agent in the world
        """
        return self._id 

    def set_position(self, position : np.array):
        """
        Set the agent's position to `position`
        """
        self._previous_position = self._current_position
        self._current_position = position

    def get_previous_position(self) -> np.ndarray | None :
        """
        Get a copy of the agent's previous position if it is defined
        """
        if self._previous_position == None:
            return None 
        return self._previous_position.copy()

    def get_current_position(self) -> np.ndarray:
        """
        Get a copy of the agent's current position position
        """
        return self._current_position.copy()
    
    def get_action(self) -> ActionInformation:
        """
        Return the current action of the agent 
        """
        return self._current_action
    
    def has_moved(self) -> bool:
        if self._previous_position is None: 
            return False 
        return  np.all(self._previous_position == self._current_position)
        
        