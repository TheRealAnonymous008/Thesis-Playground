from pygame.surface import Surface
from .vector import Vector, is_in_bounds, ZERO_VECTOR
from .constants import DEFAULT_RECT
from .tiles import Sprite, AssetProfiles
from .direction import Direction, DirectionVectors

class WorldTile: 
    def __init__(self, world, position : Vector, sprite : Sprite = None ):
        self.position : Vector = position
        self.sprite : Sprite = sprite 
        self.is_passable = True

        self.place(world, self.position)

    def draw(self, surface):
        self.sprite.draw(surface)

    
    def update(self, world):
        pass 

    def place(self, world, position : Vector):
        if not is_in_bounds(position, ZERO_VECTOR, world.bounds):
            return 
        self.position = position 


    
    def move(self, world, offset : Vector):
        if not world.is_passable(self.position + offset):
            return False
        new_loc = self.position + offset
        self.place(world, new_loc)
        return True 
        
    def move_direction(self, world, direction : Direction):
        match(direction):
            case Direction.NORTH:
                offset = DirectionVectors.NORTH
            case Direction.SOUTH:
                offset = DirectionVectors.SOUTH
            case Direction.EAST:
                offset= DirectionVectors.EAST 
            case Direction.WEST:
                offset = DirectionVectors.WEST
        
        return self.move(world, offset)


    def draw(self, surface : Surface):
        self.sprite.set_coordinate(self.position)
        self.sprite.draw(surface)

class EmptyTile(WorldTile):
    def __init__(self, world, position : Vector, should_render = True):
        super().__init__(position= position,
                         world = world,
                         sprite= Sprite(AssetProfiles.EMPTY, DEFAULT_RECT)
                         )
        
class WallTile(WorldTile):
    def __init__(self, world, position: Vector, should_render = True):
        super().__init__(position=position,
                         world = world,
                         sprite= Sprite(AssetProfiles.WALL, DEFAULT_RECT))
        self.is_passable = False 