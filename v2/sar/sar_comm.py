from __future__ import annotations

from core.message import * 

import torch
import torch.nn as nn

@dataclass 
class SARMessagePacket: 
    location : torch.Tensor = None 

class SARCommunicationProtocol(BaseCommunicationProtocol):
    def __init__(self, encoder : nn.Module, decoder : nn.Module):
        super().__init__()
        self._encoder = encoder 
        self._decoder = decoder 

        self._embeddings = None

    def start(self, world : BaseWorld):
        self._embeddings = self._encoder.encoder_forward_batch(world.agents)

    def _choose_targets(self, sender : Agent) -> Agent :
        return super()._choose_targets(sender)
    
    def _formulate_message_contents(self, sender : Agent , receiver : Agent) -> Message:
        contents = SARMessagePacket( 
            torch.tensor(sender.current_position, device = sender._device, dtype = torch.float32)
        )
        return contents
    
    def _interpret_message_contents(self, agent : Agent, message : Message):
        agent._current_belief = self._decoder.decoder_forward(agent._current_belief, message.message, self._embeddings[message.sender.id - 1])
        agent.add_relation(message.sender.id ,1)
    
    