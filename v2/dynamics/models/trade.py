from __future__ import annotations

from core.message import * 
from core.resource import * 
from enum import Enum 

class TradeMessagePacketHeader(Enum):
    REQ = 1         # Trade Request
    ACC = 2         # Accept
    REF = 3         # Refuse

class TradeMesasgeOfferType(Enum):
    RESOURCE = 1 
    MONEY = 2 
    INFORMATION  =3
    NONE = 4 

@dataclass 
class OfferPacket:
    offer_type : TradeMesasgeOfferType = TradeMesasgeOfferType.RESOURCE                                    
    offer : Resource | float | np.ndarray | None = None              

@dataclass 
class TradeMessagePacket: 
    # The message header which dictates the kind of message 
    header : TradeMessagePacketHeader = TradeMessagePacketHeader.REQ         
    # What is being offered. It is either a resource, money, or information
    offer_sender : OfferPacket = None           # I give you . . . 
    offer_receiver : OfferPacket = None         # You give me . . . 

class TradeCommunicationProtocol(BaseCommunicationProtocol):
    def __init__(self):
        super().__init__()

    def _choose_target(self, sender : Agent) -> Agent :
        return super()._choose_target(sender)
    
    def _formulate_message_contents(self, sender : Agent , receiver : Agent) -> Message:
        contents = TradeMessagePacket() 
        return contents
    
    def _interpret_message_contents(self, receiver : Agent, message : Message):
        packet : TradeMessagePacket = message.message
        sender = message.sender

        match(packet.header):
            case TradeMessagePacketHeader.REQ: 
                # We can either accept, refuse, or push another request.  
                pass 
            
            case TradeMessagePacketHeader.ACC: 
                # Accept the deal and perform the transaction
                i_get = packet.offer_sender
                u_get = packet.offer_receiver

                self._perform_transaction(sender, receiver, i_get)
                self._perform_transaction(receiver, sender,  u_get)

            
            case TradeMessagePacketHeader.REF: 
                # Close the deal 
                pass

    def _perform_transaction(self, sender : Agent, receiver : Agent,  offer : OfferPacket ):
        match(offer.offer_type):
            case TradeMesasgeOfferType.RESOURCE: 
                pass 

            case TradeMesasgeOfferType.MONEY: 
                pass 
            
            case TradeMesasgeOfferType.INFORMATION: 
                pass  
            
            case TradeMesasgeOfferType.NONE: 
                return 
    