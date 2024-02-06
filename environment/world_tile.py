from pygame.surface import Surface
from .vector import Vector, is_in_bounds, ZERO_VECTOR
from .constants import DEFAULT_RECT
from .tiles import Sprite, AssetProfiles
from .direction import Direction, DirectionVectors

class WorldTile: 
    def __init__(self, world, position : Vector, should_render = True, sprite : Sprite = None ):
        self.position : Vector = position
        self.sprite : Sprite = sprite 
        self.is_passable = True
        
        if self.sprite is None:
            self.should_render = False 
        else: 
            self.should_render = should_render

        self.place(world, self.position)

    def draw(self, surface):
        if self.should_render:
            self.sprite.draw(surface)

    
    def place(self, world, position : Vector):
        if not is_in_bounds(position, ZERO_VECTOR, world.bounds):
            return 
        self.position = position 

        if self.should_render:
            self.sprite.set_coordinate(position)

    
    def move(self, world, offset : Vector):
        if not world.is_passable(self.position.add(offset)):
            return 
        
        self.place(world, self.position.add(offset))
        
    def move_direction(self, world, direction : Direction):
        match(direction):
            case Direction.NORTH:
                self.move(world, DirectionVectors.NORTH)
            case Direction.SOUTH:
                self.move(world, DirectionVectors.SOUTH)
            case Direction.EAST:
                self.move(world, DirectionVectors.EAST)
            case Direction.WEST:
                self.move(world, DirectionVectors.WEST)

    def render(self, surface : Surface):
        if self.should_render:
            self.sprite.draw(surface)

class EmptyTile(WorldTile):
    def __init__(self, world, position : Vector, should_render = True):
        super().__init__(position= position,
                         world = world,
                         should_render=should_render,
                         sprite= Sprite(AssetProfiles.EMPTY, DEFAULT_RECT)
                         )
        
class WallTile(WorldTile):
    def __init__(self, world, position: Vector, should_render = True):
        super().__init__(position=position,
                         world = world,
                         should_render=should_render,
                         sprite= Sprite(AssetProfiles.WALL, DEFAULT_RECT))
        self.is_passable = False 