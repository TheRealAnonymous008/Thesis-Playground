import numpy as np 

import environment as env

class Entity: 
    """
    Arguments: 

    id - the id of this entity. 

    world - reference to the world this entity is inside
    """
    def __init__(self, id : int, world):
        self.id = id
        self.world : env.world.World = world
        self.reset()

    def update(self):
        # Sample from the need vector
        need = np.random.choice(a=self.world.market._industry_count, p=self._need_vector)
        self.world.market.post_need(need)

        # Attempt to produce a product 
        # TODO Current selection of product is based on the skill vector but this will need to be changed 
        industry = np.random.choice(a=self.world.market._industry_count, p = self._occupation_vector)
        self.world.market.sell_product(industry, product=env.product.Product(
            quantity=1, quality=[self._skill_vector], price=1))

    def report(self):
        return {
            "money": 0,
            "skills": self._skill_vector,
            "needs": self._need_vector,
        }

    def reset(self):
        self._skill_vector = self.world.market.sample_skills()
        self._need_vector = self.world.market.sample_needs()

        self._occupation_vector = self.world.market.sample_skills()
        self._occupation_vector /= sum(self._occupation_vector)



