
from __future__ import annotations

from abc import ABC
from typing import TYPE_CHECKING
from dataclasses import dataclass
if TYPE_CHECKING: 
    from core.agent import * 
    from core.world import * 

import numpy as np 
@dataclass
class Message: 
    sender : Agent = None 
    message : int = 0             # Override the message here

class BaseCommunicationProtocol(ABC):
    """
    Base Class for streamlining comms 

    Child classes should only override the following methods and no more
    -> _choose_target
    -> _formulate_message_contents
    -> _interpret_message_contents

    """
    def __init__(self):
        pass 

    def send_messages(self, world : World): 
        """
        Protocol for sending messages. 
        """
        for agent in world.agents :
            choice = self._choose_target(agent)
            if choice != None: 
                tgt_agent = world.get_agent(choice)
                message_contents = self._formulate_message_contents(agent, tgt_agent)
                message = Message(agent, message_contents)
                agent.send_message(tgt_agent, message)

    def _choose_target(self, sender : Agent) -> Agent | None:
        """
        Chooses an agent to send the message to. Returns either an Agent or None (should not return the sender).
        """
        neighbors = sender.agents_in_range 
        # Send agents a message
        if len (neighbors) > 0:
            return (np.random.choice(neighbors))
        else: 
            return  None 

    def _formulate_message_contents(self, sender : Agent, receiver : Agent):
        """
        Formulates a message between sender and receiver. Returns the message contents which can be any object as long as it is interpretable 
        """
        return 1


    def receive_messages(self, world : World):
        """
        Protocol for receiving messages.
        """    
        # Comms protocol (receiving)
        for agent in world.agents: 
            received_messages = agent.get_messages()
            for msg in received_messages: 
                self._interpret_message_contents(agent, msg)
            agent.clear_messages()

    def _interpret_message_contents(self, receiver : Agent, message : Message): 
        """
        Protocol for interpreting any received messages. Does not return anything.
        """
        pass 
