import numpy as np 

from .industry import Industry 

class Market:
    """
    Arguments:

    _industries - the number of initial industries. This also corresponds to the number 
    of skills recognized in the market (1 skill per industry)
    """

    def __init__(self,industries = 10): 
        self.industry_count = industries
        self.industries : list[Industry] = []

        self.reset()

    def reset(self):
        self._industry_count = self.industry_count 
        self.industries.clear()

        for i in range(0, self._industry_count):
            self.industries.append(Industry(i))

    def post_need(self, industry : int):
        self.industries[industry].demand += 1

    def sample_skills(self) -> np.ndarray:
        # Assume that the skillfulness of an agent in a particular worker is obtained by sampling from a
        # Normal distribution. 
        # Rationale: Law of large numbers.
        return sample_truncated_normal(size=self._industry_count)
    
    def sample_needs(self) -> np.ndarray:
        need = sample_truncated_normal(size=self._industry_count)
        return need / sum(need)

    def report(self):
        report = {}
        
        for industry in self.industries:
            report[industry.id] = industry.report()

        return report 

# Util functions
def sample_truncated_normal(mean=0.5, sd=0.1, low=0, high=1, size=1):
    samples = []
    while len(samples) < size:
        sample = np.random.normal(mean, sd)
        if low <= sample <= high:
            samples.append(sample)
    return np.array(samples) if size > 1 else samples[0]