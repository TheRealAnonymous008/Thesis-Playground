class Vector: 
    def __init__(self, x, y):
        self.x = x 
        self.y = y

    def add(self, other):
        return Vector(self.x + other.x, self.y + other.y)
    
    def is_equal(self, other):
        return self.x == other.x and self.y == other.y 
    
    def mult(self, scalar):
        return Vector(self.x * scalar, self.y * scalar )
    
ZERO_VECTOR = Vector(0, 0)


def is_in_bounds(v : Vector, lb : Vector , ub : Vector):
    return v.x >= lb.x and v.y >= lb.y and v.x < ub.x and v.y < ub.y