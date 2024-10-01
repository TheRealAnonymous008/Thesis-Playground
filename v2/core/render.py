import pygame
import numpy as np

from typing import Callable
from .world import World

def render_world(world: World, screen_size: tuple[int, int] = (600, 600), update_fn: Callable | None = None, delay_s: float = 1):
    """
    Renders a given scenario
    """
    pygame.init()

    screen = pygame.display.set_mode(screen_size)
    clock = pygame.time.Clock()
    cell_size = (screen_size[0] // world._dims[0], screen_size[1] // world._dims[1])
    delay = int(delay_s * 1000)

    # Initialize font for displaying FPS
    pygame.font.init()
    font = pygame.font.SysFont("Arial", 18)

    # Height map rendering toggle state
    render_height_map = False

    def draw_grid():
        for x in range(0, screen_size[0], cell_size[0]):
            pygame.draw.line(screen, (200, 200, 200), (x, 0), (x, screen_size[1]))
        for y in range(0, screen_size[1], cell_size[1]):
            pygame.draw.line(screen, (200, 200, 200), (0, y), (screen_size[0], y))

    def draw_agents():
        for agent in world.agents:
            pos_const = agent.current_position_const
            center = (pos_const[0] * cell_size[0] + cell_size[0] // 2, pos_const[1] * cell_size[1] + cell_size[1] // 2)
            radius = min(cell_size) // 3
            pygame.draw.circle(screen, (0, 255, 0), center, radius)

    def draw_resources():
        resource_map = world.resource_map
        x0, y0 = resource_map.shape
        for x in range(0, x0):
            for y in range(0, y0):
                resource_type, _ = resource_map.get((x, y))
                if resource_type > 0:
                    color = (resource_type * 40, 100, 100)  # Assign color based on resource type
                    rect = pygame.Rect(x * cell_size[0], y * cell_size[1], cell_size[0], cell_size[1])
                    pygame.draw.rect(screen, color, rect)

    def draw_height_map():
        terrain_map = world.terrain_map
        min_height = world._terrain_generator.min_height
        max_height = world._terrain_generator.max_height
        x0, y0 = terrain_map.shape
        for i in range(0, x0):
            for j in range(0, y0):
                height_value = terrain_map.get_height((i, j))
                brightness = int(((height_value - min_height) /(max_height - min_height)) * 255)  
                color = (brightness, brightness, brightness)
                print((i, j), brightness)
                rect = pygame.Rect(i * cell_size[0], j * cell_size[1], cell_size[0], cell_size[1])
                pygame.draw.rect(screen, color, rect)


    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_m:
                    render_height_map = not render_height_map  # Toggle the height map view

        if update_fn is not None:
            update_fn()

        screen.fill((0, 0, 0))

        # Draw the height map if toggled on
        if render_height_map:
            draw_height_map()
        else:
            draw_resources()
            draw_agents()

        # Display the FPS
        fps = str(int(clock.get_fps()))
        fps_text = font.render(f"FPS: {fps}", True, pygame.Color("white"))
        screen.blit(fps_text, (screen_size[0] - 100, 10))  # Display in the upper right

        pygame.time.delay(delay)
        pygame.display.flip()
        clock.tick(180)

    pygame.quit()
