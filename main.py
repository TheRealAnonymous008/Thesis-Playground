import pygame
from pygame.locals import *
from environment.constants import BLOCK_SIZE
from environment.world import World
from environment.factory import Factory

class FactorySimulation():
    def __init__(self):
        super().__init__()
        self.DISPLAY_WIDTH = 800
        self.DISPLAY_HEIGHT = 600
        
        self.running = True 

        self.world = World(10, 10, BLOCK_SIZE)

    def run(self):
        pygame.init()
        pygame.display.set_mode((self.DISPLAY_WIDTH, self.DISPLAY_HEIGHT))

        while self.running:
            for event in pygame.event.get():
                if event.type == QUIT:
                    self.running = False 
                    break

            self.update()
            self.draw()
        
        
        pygame.quit()


    def update(self):
        pygame.display.update()
        self.world.update()

    def draw(self):
        self.world.draw(pygame.display.get_surface())
        pygame.display.flip()

if __name__ == "__main__":
    FactorySimulation().run()