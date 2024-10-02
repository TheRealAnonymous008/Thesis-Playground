import numpy as np 

def bresenham_line(mask: np.ndarray, start : tuple[int, int], end : tuple[int, int]):
    """Draw a line on the mask using Bresenham's line algorithm."""
    x1, y1 = start 
    x2, y2 = end

    dx = abs(x2 - x1)
    dy = abs(y2 - y1)
    sx = 1 if x1 < x2 else -1
    sy = 1 if y1 < y2 else -1
    err = dx - dy

    while True:
        mask[x1, y1] = 1  # Set road presence

        if x1 == x2 and y1 == y2:
            break
        err2 = err * 2
        if err2 > -dy:
            err -= dy
            x1 += sx
        if err2 < dx:
            err += dx
            y1 += sy