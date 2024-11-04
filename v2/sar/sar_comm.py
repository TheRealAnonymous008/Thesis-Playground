from __future__ import annotations

from core.message import * 

import torch.nn as nn

@dataclass 
class SARMessagePacket: 
    location : np.ndarray = None 

class SARCommunicationProtocol(BaseCommunicationProtocol):
    def __init__(self, encoder_decoder : nn.Module):
        super().__init__()
        self._encoder = encoder_decoder

    def start(self, world : BaseWorld):
        self._encoder.encoder_forward_batch(world.agents)

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
    
    