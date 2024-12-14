import numpy as np 

def bresenham_line(mask: np.ndarray, start: tuple[int, int], end: tuple[int, int], thickness: int = 0):
    """Draw a line on the mask using Bresenham's line algorithm, with a specified thickness."""
    x1, y1 = start
    x2, y2 = end

    dx = abs(x2 - x1)
    dy = abs(y2 - y1)
    sx = 1 if x1 < x2 else -1
    sy = 1 if y1 < y2 else -1
    err = dx - dy

    def draw_thick_point(x, y, thickness):
        """Draw a thick point by marking all the pixels within the thickness radius."""
        for i in range(-thickness, thickness + 1):
            for j in range(-thickness, thickness + 1):
                if 0 <= x + i < mask.shape[0] and 0 <= y + j < mask.shape[1]:
                    if np.sqrt(i**2 + j**2) <= thickness:  # Check within the radius for circular thickness
                        mask[x + i, y + j] = 1

    while True:
        draw_thick_point(x1, y1, thickness)  # Draw a thick point at each step

        if x1 == x2 and y1 == y2:
            break
        err2 = err * 2
        if err2 > -dy:
            err -= dy
            x1 += sx
        if err2 < dx:
            err += dx
            y1 += sy