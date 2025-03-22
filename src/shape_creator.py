import numpy as np

def generate_circle_grid(N, L):
    """
    Creates a circle grid, with the radius equal to L/2. 

    Parameters: 
        N (int) : Number of discretization steps.  
        L (int) : Diameter of the circle. 

    Returns:
        - x (ndarray): X-coordinates of the grid.
        - y (ndarray): Y-coordinates of the grid.
        - in_square (ndarray): Boolean mask indicating points within the circle.
        - h (float): Grid spacing.
    """
    R = L / 2
    h = L / N
    y, x = np.meshgrid(np.linspace(-R, R, N), np.linspace(-R, R, N))
    in_circle = x**2 + y**2 <= R**2

    return x, y, in_circle, h 


def generate_square_grid(N, L):
    """
    Creates a square grid, with the side of the square equal to L. 

    Parameters: 
        N (int) : Number of discretization steps.  
        L (int) : Length of the side of the square. 

    Returns:
        - x (ndarray): X-coordinates of the grid.
        - y (ndarray): Y-coordinates of the grid.
        - in_square (ndarray): Boolean mask indicating points within the square.
        - h (float): Grid spacing.
    """
    h = L / N
    y, x = np.meshgrid(np.linspace(0, L, N), np.linspace(0, L, N))
    in_square = np.ones((N, N), dtype=bool)
    in_square[0] = False
    in_square[-1] = False
    in_square[:, 0] = False
    in_square[:, -1] = False

    return x, y, in_square, h


def generate_rectangle_grid(N, L):
    """
    Creates a rectangle grid, with the sides equal to L and 2L. 

    Parameters: 
        N (int) : Number of discretization steps.  
        L (int) : Length of the side of the shorter side of the rectangle. 

    Returns:
        - x (ndarray): X-coordinates of the grid.
        - y (ndarray): Y-coordinates of the grid.
        - in_square (ndarray): Boolean mask indicating points within the rectangle.
        - h (float): Grid spacing.
    """
    h = L / N  
    y, x = np.meshgrid(np.linspace(0, L, N), np.linspace(0, L, N))

    in_rectangle = np.zeros((N, N), dtype=bool)

    x_min, x_max = L/N, L - (L/N)  # Rectangle spans 25% to 75% in x
    y_min, y_max = L * 0.25, L * 0.75  # Rectangle spans 25% to 75% in y

    in_rectangle[(x >= x_min) & (x <= x_max) & (y >= y_min) & (y <= y_max)] = True

    return x, y, in_rectangle, h
