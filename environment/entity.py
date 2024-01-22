import numpy as np 
from .world import World

class Entity: 
    """
    Arguments: 

    id - the id of this entity. 

    world - reference to the world this entity is inside

    _skills - the initial number of skills this entity can have inside its skill vector. The skill behavior 
    models productivity on various items 
 
    """
    def __init__(self, id : int, world : World, _skills : int = 10):
        self.id = id
        self.n_skills = _skills

        self.reset()

    def update(self):
        pass 

    def report(self):
        return {
            "money": 0,
            "skills": self._skill_vector
        }

    def reset(self):
        # Assume that the skillfulness of an agent in a particular worker is obtained by sampling from a
        # Normal distribution. 
        # Rationale: Law of large numbers.

        self._skill_vector = sample_truncated_normal(size=self.n_skills)



# Util functions
def sample_truncated_normal(mean=0.5, sd=0.1, low=0, high=1, size=1):
    samples = []
    while len(samples) < size:
        sample = np.random.normal(mean, sd)
        if low <= sample <= high:
            samples.append(sample)
    return np.array(samples) if size > 1 else samples[0]