class Firm: 
    def __init__(self, id):
        self.id = id 

        self.quantity = 10 
        self.price = 10
        self.productivity = 1

        self.reset()

    def update(self):
        pass 

    def report(self):
        print(f"{self.id}: qty={self._quantity} price={self._price}")

    def reset(self):
        self._quantity = self.quantity
        self._price = self.price

