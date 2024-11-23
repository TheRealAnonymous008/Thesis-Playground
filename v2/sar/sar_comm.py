from __future__ import annotations

from core.message import * 

import torch
import torch.nn as nn

from .sar_agent import SARObservation

@dataclass 
class SARMessagePacket: 
    victims : torch.Tensor = None
    exploration : torch.Tensor = None 

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
        obs : SARObservation = sender._current_observation
        vision_grid = obs.victim_map
        exploration_grid = obs.exploration_map

        vision_tensor = torch.from_numpy(vision_grid).float().unsqueeze(0).to(sender._device)
        exploration_tensor = torch.from_numpy(exploration_grid).float().unsqueeze(0).to(sender._device)
        contents = SARMessagePacket( 
            vision_tensor, 
            exploration_tensor
        )
        return contents
    
    def _interpret_message_contents(self, agent : Agent, message : Message):
        agent._current_belief = self._decoder.decoder_forward(agent._current_belief, message.message, self._embeddings[message.sender.id - 1])
        agent.add_relation(message.sender.id ,1)
    
    