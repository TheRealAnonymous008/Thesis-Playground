from __future__ import annotations

from core.message import * 

import torch
import torch.nn as nn

@dataclass 
class SARMessagePacket: 
    location : np.ndarray = None 

class SARCommunicationProtocol(BaseCommunicationProtocol):
    def __init__(self, encoder : nn.Module, decoder : nn.Module):
        super().__init__()
        self._encoder = encoder 
        self._decoder = decoder 

        self._embeddings = None

    def start(self, world : BaseWorld):
        self._embeddings = self._encoder.encoder_forward_batch(world.agents)

    def _choose_target(self, sender : Agent) -> Agent :
        return super()._choose_target(sender)
    
    def _formulate_message_contents(self, sender : Agent , receiver : Agent) -> Message:
        contents = SARMessagePacket( 
            sender.current_position
        )
        return contents
    
    def _interpret_message_contents(self, agent : Agent, message : Message):
        belief : torch.Tensor = self._decoder.decoder_forward(message.message, self._embeddings[message.sender.id - 1])
        agent._current_belief += belief.numpy()
        agent.add_relation(message.sender.id ,1)
    
    