from __future__ import annotations

from core.message import * 

@dataclass 
class SARMessagePacket: 
    location : np.ndarray = None 

class SARCommunicationProtocol(BaseCommunicationProtocol):
    def __init__(self):
        super().__init__()

    def _choose_target(self, sender : Agent) -> Agent :
        return super()._choose_target(sender)
    
    def _formulate_message_contents(self, sender : Agent , receiver : Agent) -> Message:
        contents = SARMessagePacket( 
            sender.current_position
        )
        return contents
    
    def _interpret_message_contents(self, agent : Agent, message : Message):
        super()._interpret_message_contents(agent, message)
        agent.add_relation(message.sender.id ,1)
    
    