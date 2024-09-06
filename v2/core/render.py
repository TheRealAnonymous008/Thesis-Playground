import pygame
import numpy as np

from typing import Callable
from .world import World

def render_world(world: World, screen_size: tuple[int, int] = (600, 600), update_fn : Callable | None= None ):
    pygame.init()

    screen = pygame.display.set_mode(screen_size)
    clock = pygame.time.Clock()
    cell_size = (screen_size[0] // world._dims[0], screen_size[1] // world._dims[1])

    def draw_grid():
        for x in range(0, screen_size[0], cell_size[0]):
            pygame.draw.line(screen, (200, 200, 200), (x, 0), (x, screen_size[1]))
        for y in range(0, screen_size[1], cell_size[1]):
            pygame.draw.line(screen, (200, 200, 200), (0, y), (screen_size[0], y))

    def draw_agents():
        for agent in world._agents:
            pos = agent.get_position()
            rect = pygame.Rect(pos[0] * cell_size[0], pos[1] * cell_size[1], cell_size[0], cell_size[1])
            pygame.draw.rect(screen, (0, 255, 0), rect)

    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        if update_fn != None: 
            update_fn()

        screen.fill((0, 0, 0))
        # draw_grid()
        draw_agents()

        pygame.display.flip()
        clock.tick(30)

    pygame.quit()
