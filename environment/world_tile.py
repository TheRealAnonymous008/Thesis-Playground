from pygame.surface import Surface
from .vector import Vector  
from .constants import is_in_bounds, DEFAULT_RECT
from .tiles import Sprite, AssetProfiles
from .direction import Direction, DirectionVectors

class WorldTile: 
    def __init__(self, position : Vector, should_render = True, sprite : Sprite = None ):
        self.position : Vector = position
        self.sprite : Sprite = sprite 
        self.is_passable = True
        
        if self.sprite is None:
            self.should_render = False 
        else: 
            self.should_render = should_render

        self.place(self.position)

    def draw(self, surface):
        if self.should_render:
            self.sprite.draw(surface)

    
    def place(self, position : Vector):
        if not is_in_bounds(position):
            return 
        self.position = position 

        if self.should_render:
            self.sprite.set_coordinate(position)

    
    def move(self, offset : Vector, world):
        if not world.is_passable(self.position.add(offset)):
            return 
        
        self.place(self.position.add(offset))
        
    def move_direction(self, direction : Direction, world):
        match(direction):
            case Direction.NORTH:
                self.move(DirectionVectors.NORTH, world)
            case Direction.SOUTH:
                self.move(DirectionVectors.SOUTH, world)
            case Direction.EAST:
                self.move(DirectionVectors.EAST, world)
            case Direction.WEST:
                self.move(DirectionVectors.WEST, world)

    def render(self, surface : Surface):
        if self.should_render:
            self.sprite.draw(surface)

class EmptyTile(WorldTile):
    def __init__(self, position : Vector, should_render = True):
        super().__init__(position= position,
                         should_render=should_render,
                         sprite= Sprite(AssetProfiles.EMPTY, DEFAULT_RECT)
                         )
        
class WallTile(WorldTile):
    def __init__(self, position: Vector, should_render = True):
        super().__init__(position=position,
                         should_render=should_render,
                         sprite= Sprite(AssetProfiles.WALL, DEFAULT_RECT))
        self.is_passable = False 