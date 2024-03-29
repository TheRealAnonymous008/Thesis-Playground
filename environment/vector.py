class Vector: 
    def __init__(self, x, y):
        self.x = x 
        self.y = y

    def __add__(self, other):
        return Vector(self.x + other.x, self.y + other.y)
    
    def __sub__(self, other):
        return Vector(self.x - other.x, self.y - other.y)

    def __eq__(self, other):
        return self.x == other.x and self.y == other.y 
    
    def __mul__(self, scalar):
        return Vector(self.x * scalar, self.y * scalar )
    
    def __hash__(self):
        return hash((self.x, self.y))
    
    def __str__(self):
        return (str(self.x) + ", " + str(self.y)) 
    
ZERO_VECTOR = Vector(0, 0)


def is_in_bounds(v : Vector, lb : Vector , ub : Vector):
    return v.x >= lb.x and v.y >= lb.y and v.x < ub.x and v.y < ub.y