import random


class IDPool: 
    """
    A helper class to assign unique IDs to anything
    """
    _IDs : set = set()

    def get() -> int:
        _id = random.getrandbits(31)
        IDPool._IDs.add(_id)
        return _id 
    
    def pop(_id : int): 
        if _id in IDPool._IDs: 
            IDPool._IDs.remove(_id)