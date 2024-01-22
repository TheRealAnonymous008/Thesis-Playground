import numpy as np 
from .world import World

class Entity: 
    """
    Arguments: 

    id - the id of this entity. 

    world - reference to the world this entity is inside
    """
    def __init__(self, id : int, world : World):
        self.id = id
        self.world = world
        self.reset()

    def update(self):
        # Sample from the need vector
        need = np.random.choice(a=self.world.market._industry_count, p=self.need_vector)
        self.world.market.post_need(need)

    def report(self):
        return {
            "money": 0,
            "skills": self._skill_vector,
            "needs": self.need_vector,
        }

    def reset(self):
        self._skill_vector = self.world.market.sample_skills()
        self.need_vector = self.world.market.sample_needs()



