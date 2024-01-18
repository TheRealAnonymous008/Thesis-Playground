class Entity: 
    def __init__(self, id):
        self.id = id 

        self.money = 10
        self.reset()

    def update(self):
        pass 

    def report(self):
        print(f"{self.id}: income={self._money}")

    def reset(self):
        self._money = self.money 
