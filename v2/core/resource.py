from dataclasses import dataclass

_ResourceType = int
_QuantityType = int

@dataclass
class Resource: 
    """
    Dataclass for resources. A resource holds a `type` and `quantity`
    """
    type : _ResourceType 
    quantity : _QuantityType
