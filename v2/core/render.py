import pygame
import numpy as np

from typing import Callable
from .world import World

def render_world(world: World, screen_size: tuple[int, int] = (600, 600), update_fn : Callable | None= None, delay_s : float = 1 ):
    pygame.init()

    screen = pygame.display.set_mode(screen_size)
    clock = pygame.time.Clock()
    cell_size = (screen_size[0] // world._dims[0], screen_size[1] // world._dims[1])
    delay = int (delay_s * 1000)
    def draw_grid():
        for x in range(0, screen_size[0], cell_size[0]):
            pygame.draw.line(screen, (200, 200, 200), (x, 0), (x, screen_size[1]))
        for y in range(0, screen_size[1], cell_size[1]):
            pygame.draw.line(screen, (200, 200, 200), (0, y), (screen_size[0], y))

    def draw_agents():
        for agent in world.get_agents():
            pos = agent.get_position()
            center = (pos[0] * cell_size[0] + cell_size[0] // 2, pos[1] * cell_size[1] + cell_size[1] // 2)
            radius = min(cell_size) // 3
            pygame.draw.circle(screen, (0, 255, 0), center, radius)
            
    def draw_resources():
        resource_map = world.get_resource_map()
        for x in range(resource_map.shape[0]):
            for y in range(resource_map.shape[1]):
                resource_type = resource_map[x, y]
                if resource_type > 0:
                    color = (resource_type * 40, 100, 100)  # Assign color based on resource type
                    rect = pygame.Rect(x * cell_size[0], y * cell_size[1], cell_size[0], cell_size[1])
                    pygame.draw.rect(screen, color, rect)


    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        if update_fn != None: 
            update_fn()

        screen.fill((0, 0, 0))
        draw_resources()
        draw_agents()
        
        pygame.time.delay(delay)
        pygame.display.flip()
        clock.tick(30)

    pygame.quit()
