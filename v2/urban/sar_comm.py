from __future__ import annotations

from core.message import * 

class SARCommunicationProtocol(BaseCommunicationProtocol):
    def __init__(self):
        super().__init__()

    def _choose_target(self, sender : Agent) -> Agent :
        return super()._choose_target(sender)
    
    def _formulate_message(self, sender : Agent , receiver : Agent) -> Message:
        return super()._formulate_message(sender, receiver)
    
    def _interpret_message(self, agent : Agent, message : Message):
        super()._interpret_message(agent, message)
    