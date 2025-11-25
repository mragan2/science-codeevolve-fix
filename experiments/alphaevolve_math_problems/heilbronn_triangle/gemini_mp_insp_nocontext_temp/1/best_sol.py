# EVOLVE-BLOCK-START
import numpy as np
import itertools
from scipy.optimize import differential_evolution

# --- Geometric Helper Functions & Constants ---

# Define the equilateral triangle vertices
V1 = np.array([0.0, 0.0])
V2 = np.array([1.0, 0.0])
V3 = np.array([0.5, np.sqrt(3) / 2])

# Pre-calculate values for the barycentric coordinate check to speed up
# the point-in-triangle test, which is called frequently.
_v0 = V2 - V1
_v1 = V3 - V1
_d00 = np.dot(_v0, _v0)
_d01 = np.dot(_v0, _v1)
_d11 = np.dot(_v1, _v1)
# Add a small epsilon to prevent division by zero in case of degenerate pre-calculation
_INV_DENOM = 1.0 / (_d00 * _d11 - _d01 * _d01 + 1e-9)

def _is_inside_triangle(p: np.ndarray) -> bool:
    """
    Checks if a point p is inside the predefined equilateral triangle using
    fast barycentric coordinate calculations.
    """
    _v2 = p - V1
    _d20 = np.dot(_v2, _v0)
    _d21 = np.dot(_v2, _v1)
    u = (_d11 * _d20 - _d01 * _d21) * _INV_DENOM
    v = (_d00 * _d21 - _d01 * _d20) * _INV_DENOM
    # Add a small tolerance to handle floating point inaccuracies for points on the boundary
    return (u >= -1e-9) and (v >= -1e-9) and (u + v <= 1.0 + 1e-9)

def _triangle_area(p1: np.ndarray, p2: np.ndarray, p3: np.ndarray) -> float:
    """Calculates the area of a triangle defined by three points."""
    return 0.5 * np.abs(p1[0] * (p2[1] - p3[1]) + p2[0] * (p3[1] - p1[1]) + p3[0] * (p1[1] - p2[1]))

def _calculate_min_area(points: np.ndarray) -> float:
    """
    Calculates the minimum area of any triangle formed by a set of points.
    """
    min_area = float('inf')
    # Iterate through all C(n,3) combinations of points
    for p1, p2, p3 in itertools.combinations(points, 3):
        area = _triangle_area(p1, p2, p3)
        # A small threshold helps ignore degenerate triangles from floating point issues.
        if area > 1e-12:
            min_area = min(min_area, area)
            
    return min_area if min_area != float('inf') else 0.0

def _objective_func(flat_points: np.ndarray, n_points: int) -> float:
    """
    Objective function for the optimizer. It aims to be minimized.
    - Reshapes the flat input array into a list of 2D points.
    - Applies a large penalty if any point is outside the triangle.
    - Returns the negative of the minimum triangle area, as the goal is to maximize it.
    """
    points = flat_points.reshape(n_points, 2)
    
    # Constraint handling: apply a large penalty for points outside the triangle.
    for p in points:
        if not _is_inside_triangle(p):
            # Return a large positive value to penalize invalid configurations.
            # This is much larger than any possible valid objective value.
            return 1.0 

    min_area = _calculate_min_area(points)
    
    # The optimizer minimizes, so we return the negative of the value we want to maximize.
    return -min_area

def heilbronn_triangle11() -> np.ndarray:
    """
    Constructs an optimal arrangement of 11 points within an equilateral triangle
    to maximize the minimum area of any triangle formed by three of these points.

    This implementation uses Differential Evolution, a global optimization metaheuristic,
    to search the 22-dimensional space of point coordinates.

    Returns:
        np.ndarray: An array of shape (11, 2) with the (x, y) coordinates of the optimized points.
    """
    n = 11
    
    # Define search space bounds. The points are inside a bounding box containing the triangle.
    # The exact triangle constraint is handled by a penalty in the objective function.
    bounds = [(0, 1), (0, np.sqrt(3) / 2)] * n
    
    # Use Differential Evolution to find the optimal point arrangement.
    # A fixed seed ensures the result is deterministic and reproducible.
    result = differential_evolution(
        func=_objective_func,
        bounds=bounds,
        args=(n,),
        strategy='best1bin',
        maxiter=1000,       # Increased iterations for a more thorough search
        popsize=30,         # Increased population size for the 22-dim space
        tol=1e-7,
        mutation=(0.5, 1.0),
        recombination=0.7,
        seed=42,
        disp=False,
        workers=-1          # Use all available CPU cores for parallelization
    )
    
    optimal_points = result.x.reshape(n, 2)
    
    # Final safeguard: project any points slightly outside due to float precision back onto the boundary.
    for i, p in enumerate(optimal_points):
       if not _is_inside_triangle(p):
           p[1] = np.clip(p[1], 0, V3[1])
           max_x_at_y = 1 - p[1] / np.sqrt(3)
           min_x_at_y = p[1] / np.sqrt(3)
           p[0] = np.clip(p[0], min_x_at_y, max_x_at_y)
           optimal_points[i] = p

    return optimal_points

# EVOLVE-BLOCK-END
