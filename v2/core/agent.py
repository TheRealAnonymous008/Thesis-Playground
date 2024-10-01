from __future__ import annotations

import numpy as np


from .observation import LocalObservation

from .action import *
from .resource_map import Resource
from .message import * 

from .resource import Resource, _QuantityType, _ResourceType
from .utility import UtilityFunction
from .agent_state import AgentState
from .direction import Direction

_IdType = int

class Agent:
    def __init__(self):
        """
        Initializes a simple agent 
        """
        self._id : _IdType = "-1"

        self._previous_position : np.ndarray[int] | None = None 
        self._current_position : np.ndarray[int] | None =  None 

        self._current_observation : LocalObservation = None
        self._current_action : ActionInformation = ActionInformation()
        self._current_state :  AgentState = AgentState()

        # Attributes of the agent. 
        self._visibility_range : int = 3
        self._energy_capacity : float = 100.0
        self._carrying_capacity : float = 100.0
        self._utility_function : UtilityFunction = None 

        self.reset()
        
    def reset(self):
        """
        Reset the agent
        """
        self._previous_position = None 
        self._current_state.reset()
        self._current_state.current_energy = self._energy_capacity

    def update(self):
        """
        Update the agent's state
        """
        pass 

    def move(self, dir : Direction | int):
        """
        Moves an agent along a specified direction. 
        The direction is either a Direction instance or an integer associated with a Direction value.
        """
        if not self._current_state.can_move:
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
        if not self._current_state.can_move:
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
        if not self._current_state.can_move:
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

    def talk_to(self, agent : Agent):
        """
        Communicate with the agent. Construct a message to send to this agent 
        """
        self.send_message(agent, Message(self._id, 1))
    
    def send_message(self, agent : Agent, message : Message):
        """
        Send a message to the target agent
        """
        agent._current_state.msgs.append(message)

    def get_messages(self) -> list[Message]:
        """
        Return all messages 
        """
        return self._current_state.msgs

    def clear_messages(self):
        """
        Clear all messages 
        """
        self._current_state.clear_messages() 
    
    def add_relation(self, agent_id : _IdType, weight : float):
        """
        Add a social relation between this agent and the one specified
        """
        self._current_state.set_relation(agent_id, weight  + self._current_state.get_relation(agent_id))

    def make(self, prod : _ResourceType): 
        """
        Choose to make a product 
        """
        self._current_action.production_job = prod 

    def add_to_inventory(self, res : Resource) -> float:
        """
        Add resource to inventory, assuming it can be carried. Return excess mass.
        """
        current_carried = self._current_state.current_mass_carried
        excess_mass = max(0, current_carried + res.quantity - self._carrying_capacity)
        res.quantity = res.quantity - excess_mass

        self._current_state.add_to_inventory(res)
        return excess_mass
    

    def get_from_inventory(self, type : _ResourceType, qty : _QuantityType) -> _QuantityType:
        """
        Get a product from the inventory. Update the inventory.
        """
        return self._current_state.get_from_inventory(type, qty)
    
    def has_in_inventory(self, type : _ResourceType, qty : _QuantityType) -> bool: 
        """
        Check if the agent has the specified product in the specified amount
        """
        return self._current_state.has_in_inventory(type, qty)


    def reset_for_next_action(self):
        """
        Resets the agent for a new action
        """
        self._current_action.reset()

    def set_observation(self, observation : LocalObservation):
        """
        Set local observation
        """
        self._current_observation = observation

    def set_utility(self, utility : UtilityFunction) :
        """
        Set utility functionn
        """
        self._utility_function = utility

    def calculate_utility(self):
        """
        Evaluate this agent's utility
        """
        self._utility_function.forward(self._current_state)

    @property
    def agents_in_range(self) -> list[_IdType]:
        """
        Returns a list of id's of all visible agents.
        """
        return self._current_observation.neighbors(self.id)

    @property
    def local_observation(self) -> LocalObservation:
        """
        Get local observation
        """
        return self._current_observation

    def bind_to_world(self, world_id : int):
        """
        Set the agent's ID to be that of the ID assigned to it by the world
        """
        self._id = _IdType(world_id)

    @property
    def id(self) -> _IdType:
        """
        Return the ID of this agent in the world
        """
        return self._id 

    def __hash__(self) -> int:
        return int(self._id)

    def set_position(self, position : np.array):
        """
        Set the agent's position to `position`
        """
        self._previous_position = self._current_position
        self._current_position = position
    
    @property
    def previous_position(self) -> np.ndarray | None :
        """
        Get a copy of the agent's previous position if it is defined
        """
        if self._previous_position == None:
            return None 
        return self._previous_position.copy()

    @property
    def current_position(self) -> np.ndarray:
        """
        Get the agent's current position
        """
        return self._current_position.copy()
    
    @property
    def current_position_const(self) -> np.ndarray:
        """
        Get the agent's current position. Does not return a copy! Never mutate the output of this function
        """
        return self._current_position

    @property
    def action(self) -> ActionInformation:
        """
        Return the current action of the agent 
        """
        return self._current_action
    
    @property
    def has_moved(self) -> bool:
        """
        Return if the agent has been displaced
        """
        if self._previous_position is None: 
            return False 
        return  np.all(self._previous_position == self._current_position)
        
    
    @property
    def utility(self) : 
        return self._current_state.current_utility