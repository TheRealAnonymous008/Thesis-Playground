from __future__ import annotations

from core.message import * 
from core.resource import *
from core.resource import _ResourceType

from enum import Enum 
from dynamics.agents.trade_agent import *

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
    offer : _ResourceType | float | np.ndarray | None = None              

@dataclass 
class TradeMessagePacket: 
    # The message header which dictates the kind of message 
    header : TradeMessagePacketHeader = TradeMessagePacketHeader.REQ         
    # What is being offered. It is either a resource, money, or information
    offer_sender : OfferPacket = None           # I give you . . . 
    offer_receiver : OfferPacket = None         # You give me . . . 

class TradeCommunicationProtocol(BaseCommunicationProtocol):
    def __init__(self, request_logic : TradeRequester):
        """
        :param evaluator: Mechanism for evaluating requests 
        :param requestor: Mechanism for creating requests
        """
        super().__init__()
        self._request_logic = request_logic

    def _choose_target(self, sender : TradeAgent) -> TradeAgent :
        return super()._choose_target(sender)
    
    def _formulate_message_contents(self, sender : TradeAgent , receiver : TradeAgent) -> Message:
        return self._request_logic.create_request(sender, receiver)
    
    def _interpret_message_contents(self, receiver : TradeAgent, message : Message):
        packet : TradeMessagePacket = message.message
        sender = message.sender

        match(packet.header):
            case TradeMessagePacketHeader.REQ: 
                # TODO; Refactor this
                # We can either accept, refuse, or push another request.  
                # When pushing another request, we swap the roles of sender and receiver
                verdict = self._request_logic.evaluate(sender, packet, receiver)
                if verdict == TradeMessagePacketHeader.REQ: 
                    counter_offer = self._request_logic.create_request(sender, packet, receiver)
                else: 
                    counter_offer = packet.offer_receiver

                reply = TradeMessagePacket(
                    header= verdict, 
                    offer_sender = counter_offer, 
                    offer_receiver = packet.offer_sender,
                )
                message = Message(receiver, reply)

                receiver.send_message(sender, message)
            
            case TradeMessagePacketHeader.ACC: 
                # Accept the deal and perform the transaction
                self._perform_transaction(sender, receiver, packet.offer_sender)
                self._perform_transaction(receiver, sender,  packet.offer_receiver)

            case TradeMessagePacketHeader.REF: 
                # Close the deal 
                pass

    def _perform_transaction(self, sender : TradeAgent, receiver : TradeAgent,  offer : OfferPacket ):
        """
        Sender gives receiver the offer
        """
        match(offer.offer_type):
            case TradeMesasgeOfferType.RESOURCE: 
                qty = sender.get_from_inventory(offer.offer, 1)
                receiver.add_to_inventory(offer.offer, qty)

            case TradeMesasgeOfferType.MONEY: 
                qty = sender.get_money(offer.offer)
                receiver.add_money(qty)
            
            case TradeMesasgeOfferType.INFORMATION: 
                # TODO: Implement information passing mechanism.
                pass   
            
            case TradeMesasgeOfferType.NONE: 
                return 
    

class TradeRequester(ABC):
    def __init__(self): 
        pass 

    @abstractmethod
    def create_request(self, sender : TradeAgent, receiver : TradeAgent) -> TradeMessagePacket:
        """
        Creates a request from sender to receiver. 
        """
        pass 

    @abstractmethod
    def evaluate(self, sender : TradeAgent, request : TradeMessagePacket,  receiver : TradeAgent) -> TradeMessagePacketHeader:
        """
        Evaluate the main response for a request.
        """
        pass 