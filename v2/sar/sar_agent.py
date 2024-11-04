
from __future__ import annotations
from core.agent import *
from core.direction import * 
from core.observation import *
from core.belief import *

_UtilityType = float 

@dataclass
class SARActionInformation(ActionInformation):
    """
    Contains attributes and misc. information about an agent's actions.

    If the value is None, then that action was not taken 

    :param movement: Action correpsonding to motion on the world
    """
    movement : Direction | None = None

    def reset(self):
        self.movement = None


@dataclass 
class SARAgentTraits(AgentTraits):
    """
    Data class containing fixed parameters / traits of the agent relevant to the simulation
    """
    _visibility_range : int = 3                 # How far can the agent see 
    _energy_capacity : float = 100.0            # How far can the agent move
    _max_slope : float = 1                      # How high can the agent traverse


@dataclass
class SARAgentState(AgentState):
    """
    Contains attributes and misc. information about an agent's internal state.

    :param current_energy: Current energy of the agent
    :param current_utility:  Currently calculated utility. If None, utility was not calculated. 
    :param relations:  Dictionary mapping agent ids to agent relations
    :param msgs:  The current message buffer
    :param skill: The vector representing the skill of the agent. Must be initialized.
    """

    current_energy : float = 0
    current_utility : _UtilityType | None = None
    relations : dict[int, int] = field(default_factory= lambda : {})
    msgs: list[Message] = field(default_factory=lambda : [])
    skills : np.ndarray | None = None 
    victims_rescued : int = 0
    
    def reset(self, traits : SARAgentTraits):
        """
        Reset the state
        """
        self.current_energy = traits._energy_capacity
        self.current_utility = None
        self.relations.clear()
        self.msgs.clear()
        self.skills  = None 
        self.victims_rescued = 0

    @property
    def can_move(self):
        return self.current_energy > 0
 
    def add_message(self, message : Message) : 
        """
        Add a message to the message buffer
        """
        self.msgs.append(message)

    def clear_messages(self):
        """
        Clear all messages
        """
        self.msgs.clear()

    def set_relation(self, agent : int, weight : float): 
        """
        Set the relation between this agent and the specified `agent`  to `weight`
        """
        self.relations[agent] = weight

    def get_relation(self, agent : int) -> float:
        """
        Get the relation between this agent and the specified `agent`. Defaults to 0
        """
        if agent in self.relations:
            return self.relations[agent]
        return 0

    def remove_relatioon(self, agent : int) : 
        """
        Remove a relation 
        """
        self.relations.pop(agent)

class SARUtilityFunction(UtilityFunction):
    def __init__(self):
        super().__init__()

    def forward(self, state : SARAgentState):
        return state.victims_rescued
    
    def update(self):
        pass

@dataclass
class SARObservation(LocalObservation):
    victim_map : np.ndarray = None 

class SARBeliefInitializer(BaseBeliefInitializer):
    def __init__(self, belief_dims : int = 1):
        self.belief_dims = belief_dims

    def initialize_belief(self, agent : SARAgent):
        agent._current_belief = np.zeros((self.belief_dims))

class SARAgent(Agent):
    """
    SAR Agent Class. Override or augment this class as needed.

    This is meant to be a wrapper that allows interfacing between actions and states.
    """

    def __init__(self):
        """
        Initializes a simple agent 
        """
        super().__init__()

    def _initializer(self, *args, **kwargs):
        super()._initializer()
        self._previous_position : np.ndarray[int] | None = None 
        self._current_position : np.ndarray[int] | None =  None 
        self._traits : SARAgentTraits = SARAgentTraits()
        self._current_action : SARActionInformation = SARActionInformation()
        self._current_state :  SARAgentState = SARAgentState()
        self._utility_function : SARUtilityFunction = SARUtilityFunction()

    def _reset(self):
        self._previous_position = None 
        self._current_action = SARActionInformation()

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
            
    def rescue(self): 
        self._current_state.victims_rescued += 1

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