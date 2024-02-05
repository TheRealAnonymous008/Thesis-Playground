import pygame

class World:
    def __init__(self, width, height, block_size):
        self.width = width
        self.height = height
        self.block_size = block_size
        self.grid = [[None for _ in range(height)] for _ in range(width)]
        
    def draw(self, surface):
        for x in range(self.width):
            for y in range(self.height):
                rect = pygame.Rect(x * self.block_size, y * self.block_size, self.block_size, self.block_size)
                pygame.draw.rect(surface, (255, 255, 255), rect, 1)

    def fill_cell(self, x, y, color):
        if 0 <= x < self.width and 0 <= y < self.height:
            self.grid[x][y] = color
        else:
            raise IndexError("Cell ({}, {}) is outside the grid bounds.".format(x, y))

    def clear_cell(self, x, y):
        if 0 <= x < self.width and 0 <= y < self.height:
            self.grid[x][y] = None
        else:
            raise IndexError("Cell ({}, {}) is outside the grid bounds.".format(x, y))

    def get_cell_color(self, x, y):
        if 0 <= x < self.width and 0 <= y < self.height:
            return self.grid[x][y]
        else:
            raise IndexError("Cell ({}, {}) is outside the grid bounds.".format(x, y))
