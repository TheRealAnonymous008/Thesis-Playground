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
        if not world.is_passable(self.position.add(offset)):
            return False
        
        self.place(world, self.position.add(offset))
        
    def move_direction(self, world, direction : Direction):
        # FIrst get all the resources in that 
        current = self.position 
        offset = None 
        match(direction):
            case Direction.NORTH:
                offset = DirectionVectors.NORTH
            case Direction.SOUTH:
                offset = DirectionVectors.SOUTH
            case Direction.EAST:
                offset= DirectionVectors.EAST 
            case Direction.WEST:
                offset = DirectionVectors.WEST

        resources_to_update = []
        while (world.has_resource(current)):
            resources_to_update.append(world.get_resource(current))
            current = current.add(offset)

        for rsrc in resources_to_update:
            rsrc : WorldTile = rsrc 
            rsrc.move(world, offset)


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