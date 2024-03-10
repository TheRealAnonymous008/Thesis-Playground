import pygame
from pygame.locals import *
from environment.constants import BLOCK_SIZE, BOUNDS
from environment.world import World
from environment.factory import Factory
from environment.components import *

class FactorySimulation():
    def __init__(self):
        super().__init__()
        self.DISPLAY_WIDTH = 800
        self.DISPLAY_HEIGHT = 600
        
        self.running = True 
        
        pygame.init()
        pygame.display.set_mode((self.DISPLAY_WIDTH, self.DISPLAY_HEIGHT))

        self.world = World(BOUNDS.x, BOUNDS.y, BLOCK_SIZE)

    def run(self):
        assembler : Assembler  = self.world.factory.assemblers[3][4]

        while self.running:
            for event in pygame.event.get():
                if event.type == QUIT:
                    self.running = False 
                if event.type == KEYDOWN: 
                    if event.key == K_w: 
                        assembler.move_direction(self.world, Direction.NORTH)
                    if event.key == K_s: 
                        assembler.move_direction(self.world, Direction.SOUTH)
                    if event.key == K_d: 
                        assembler.move_direction(self.world, Direction.EAST)
                    if event.key == K_a:
                        assembler.move_direction(self.world, Direction.WEST)
                    if event.key == K_q:
                        assembler.rotate_ccw()
                    if event.key == K_e:
                        assembler.rotate_cw()
                    if event.key == K_SPACE:
                        assembler.switch_mode()
                    self.update()
                        
            self.draw()
        
        pygame.quit()


    def update(self):
        pygame.display.update()
        self.world.update()

    def draw(self):
        self.world.draw(pygame.display.get_surface())
        pygame.display.flip()
