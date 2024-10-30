import pygame
import numpy as np

from enum import Enum
from typing import Callable
from .world import BaseWorld
from sar.terrain_map import *

def render_world(world: BaseWorld, screen_size: tuple[int, int] = (600, 600), update_fn: Callable | None = None, delay_s: float = 1):
    """
    Renders a provided world on screen.

    Note that for the framework itself, the render world function is assumed to be defined based on the user's specific need. This
    code is to be treated as an example of how one might implement such a function.
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
    class RenderMode(Enum):
        DEFAULT = 1
        HEIGHT_MAP = 2
        POPULATION_MAP = 3
        VICTIM_MAP = 4  
        BLENDED_MAP = 5  # Blended height and population density

    render_mode: RenderMode = RenderMode.DEFAULT

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

    def draw_height_map():
        terrain_map: TerrainMap = world.get_map("Terrain")
        min_height = terrain_map._min_height
        max_height = terrain_map._max_height
        x0, y0 = terrain_map.shape
        for i in range(0, x0):
            for j in range(0, y0):
                height_value = terrain_map.get((i, j))
                brightness = int(((height_value - min_height) / (max_height - min_height)) * 255)
                color = (brightness, brightness, brightness)
                rect = pygame.Rect(i * cell_size[0], j * cell_size[1], cell_size[0], cell_size[1])
                pygame.draw.rect(screen, color, rect)

    def draw_population_map():
        pop_map: BaseMap = world.get_map("Population")
        min_val = 0
        max_val = 1
        x0, y0 = pop_map.shape
        for i in range(0, x0):
            for j in range(0, y0):
                val = pop_map.get((i, j))
                brightness = int(((val - min_val) / (max_val - min_val)) * 255)
                color = (brightness, brightness, brightness)
                rect = pygame.Rect(i * cell_size[0], j * cell_size[1], cell_size[0], cell_size[1])
                pygame.draw.rect(screen, color, rect)

    def draw_victim_map():
        """Draws the cells that have victims on them."""
        victim_map: BaseMap = world.get_map("Victims")  # Assuming the victim map is stored under "Victims"
        x0, y0 = victim_map.shape
        for i in range(0, x0):
            for j in range(0, y0):
                if victim_map.get((i, j)) == 1:  # Victim present at this location
                    rect = pygame.Rect(i * cell_size[0], j * cell_size[1], cell_size[0], cell_size[1])
                    pygame.draw.rect(screen, (255, 0, 0), rect)  # Draw victims in red

    def draw_blended_map():
        """Blends both height map and population density on top of each other."""
        terrain_map: TerrainMap = world.get_map("Terrain")
        pop_map: BaseMap = world.get_map("Population")
        min_height = terrain_map._min_height
        max_height = terrain_map._max_height
        min_density = 0
        max_density = 1

        x0, y0 = terrain_map.shape
        for i in range(0, x0):
            for j in range(0, y0):
                # Get height and population density
                height_value = terrain_map.get((i, j))
                density_value = pop_map.get((i, j))

                # Calculate brightness for height and density
                height_brightness = int(((height_value - min_height) / (max_height - min_height)) * 255)
                density_brightness = int(((density_value - min_density) / (max_density - min_density)) * 255)

                # Blend the two values (50% height, 50% density, modify alpha as desired)
                blended_r = int(height_brightness * 0.5 + density_brightness * 0.5)
                blended_g = int(height_brightness * 0.5)  # Height will have more greenish tint
                blended_b = int(density_brightness * 0.5)  # Density will add bluish tint

                color = (blended_r, blended_g, blended_b)
                rect = pygame.Rect(i * cell_size[0], j * cell_size[1], cell_size[0], cell_size[1])
                pygame.draw.rect(screen, color, rect)

    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_d:
                    render_mode = RenderMode.DEFAULT
                if event.key == pygame.K_h:
                    render_mode = RenderMode.HEIGHT_MAP  # Toggle the height map view
                if event.key == pygame.K_p:
                    render_mode = RenderMode.POPULATION_MAP
                if event.key == pygame.K_v:  # New key to toggle the victim map view
                    render_mode = RenderMode.VICTIM_MAP
                if event.key == pygame.K_b:  # New key to toggle the blended map
                    render_mode = RenderMode.BLENDED_MAP

        if update_fn is not None:
            update_fn()

        screen.fill((0, 0, 0))

        # Draw according to the selected render mode
        match render_mode:
            case RenderMode.DEFAULT:
                draw_agents()

            case RenderMode.HEIGHT_MAP:
                draw_height_map()

            case RenderMode.POPULATION_MAP:
                draw_population_map()

            case RenderMode.VICTIM_MAP:
                draw_victim_map()  # Draw cells with victims

            case RenderMode.BLENDED_MAP:
                draw_blended_map()  # Draw blended map

        # Display the FPS
        fps = str(int(clock.get_fps()))
        fps_text = font.render(f"FPS: {fps}", True, pygame.Color("white"))
        screen.blit(fps_text, (screen_size[0] - 100, 10))  # Display in the upper right

        pygame.time.delay(delay)
        pygame.display.flip()
        clock.tick(180)

    pygame.quit()
