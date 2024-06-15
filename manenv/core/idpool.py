import random
import threading

class IDPool:
    """
    A helper class to assign unique IDs to anything
    """
    _IDs: set = set()
    _counter: int = 0
    _lock = threading.Lock()  # To make the ID generation thread-safe

    @classmethod
    def get(cls) -> int:
        with cls._lock:
            _id = cls._counter
            cls._counter += 1
            cls._IDs.add(_id)
        return _id 

    @classmethod
    def pop(cls, _id: int): 
        with cls._lock:
            if _id in cls._IDs:
                cls._IDs.remove(_id)