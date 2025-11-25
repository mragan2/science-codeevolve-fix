# EVOLVE-BLOCK-START
import numpy as np
import itertools
from scipy.optimize import dual_annealing
import time

# Constants for the Heilbronn triangle problem
N_POINTS = 11
DOMAIN_VERTICES = np.array([[0.0, 0.0], [1.0, 0.0], [0.5, np.sqrt(3)/2.0]])
DOMAIN_AREA = np.sqrt(3)/4.0
EPSILON = 1e-7 # Small tolerance for floating point comparisons to handle boundary checks

def calculate_triangle_area(p1, p2, p3):
    """
    Calculates the area of a triangle given three 2D points.
    Uses the Shoelace formula (determinant form).
    """
    return 0.5 * abs(p1[0]*(p2[1]-p3[1]) + p2[0]*(p3[1]-p1[1]) + p3[0]*(p1[1]-p2[1]))

def is_inside_triangle(point, vertices):
    """
    Checks if a point (x, y) is inside or on the boundary of the equilateral triangle.
    The triangle vertices are A=(0,0), B=(1,0), C=(0.5, sqrt(3)/2).
    Conditions for a point (x,y) to be inside:
    1. y >= 0
    2. y <= sqrt(3) * x (Line AC)
    3. y <= -sqrt(3) * (x - 1) (Line BC)
    We use a small epsilon for boundary cases to account for floating point inaccuracies.
    """
    x, y = point
    
    # Line AB: y = 0. Point must be above or on this line.
    if y < -EPSILON:
        return False
    
    # Line AC: y = sqrt(3) * x. Point must be below or on this line.
    # Rearranged: sqrt(3) * x - y >= 0
    if np.sqrt(3) * x - y < -EPSILON:
        return False
    
    # Line BC: y = -sqrt(3) * (x - 1). Point must be below or on this line.
    # Rearranged: sqrt(3) * x + y - sqrt(3) <= 0
    if np.sqrt(3) * x + y - np.sqrt(3) > EPSILON:
        return False
        
    return True

def objective_function(points_flat):
    """
    Objective function for the Heilbronn problem.
    Takes flattened point coordinates, reshapes them, checks domain constraints,
    and returns the negative of the minimum triangle area to convert to a minimization problem.
    """
    points = points_flat.reshape((N_POINTS, 2))
    
    # Constraint handling: penalize points outside the triangle
    # This check is crucial as dual_annealing's bounds only cover the bounding box.
    for p in points:
        if not is_inside_triangle(p, DOMAIN_VERTICES):
            return 1e10 # Large penalty for infeasible solutions
            
    min_area = float('inf')
    
    # Calculate all N_POINTS choose 3 unique triangle areas
    for i, j, k in itertools.combinations(range(N_POINTS), 3):
        p1, p2, p3 = points[i], points[j], points[k]
        area = calculate_triangle_area(p1, p2, p3)
        # Handle collinear points or very small areas
        if area < min_area:
            min_area = area
            
    # If all points are collinear, min_area could be 0.
    # We want to maximize min_area, so return its negative for minimization.
    return -min_area

def generate_initial_points_in_triangle(n_points, vertices, seed=123):
    """
    Generates n_points uniformly within an equilateral triangle using barycentric coordinates.
    A fixed seed ensures reproducibility of initial point generation.
    """
    points = np.zeros((n_points, 2))
    A, B, C = vertices[0], vertices[1], vertices[2]
    
    rng = np.random.default_rng(seed=seed)
    
    for i in range(n_points):
        r1, r2 = rng.random(2)
        
        # If r1 + r2 > 1, reflect the point to be within the triangle
        if r1 + r2 > 1:
            r1 = 1 - r1
            r2 = 1 - r2
        
        # Barycentric coordinates: P = (1-r1-r2)*A + r1*B + r2*C
        # Since A=(0,0), this simplifies to P = r1*B + r2*C
        points[i] = r1 * B + r2 * C
        
    return points


def heilbronn_triangle11() -> np.ndarray:
    """
    Construct an arrangement of exactly 11 points on or inside a unit equilateral triangle
    to maximize the minimum triangle area formed by any three points.

    Returns:
        points: np.ndarray of shape (11,2) with the x,y coordinates of the optimized points.
    """
    
    # Bounds for the coordinates of each point: x in [0,1], y in [0, sqrt(3)/2]
    bounds = [(0.0, 1.0), (0.0, np.sqrt(3)/2.0)] * N_POINTS
    
    # Generate initial points uniformly within the triangle
    initial_points = generate_initial_points_in_triangle(N_POINTS, DOMAIN_VERTICES)
    
    # Flatten the initial points array for the optimizer
    x0 = initial_points.flatten()
    
    # Set a fixed random seed for the dual_annealing algorithm for reproducibility
    optimization_seed = 42 
    
    # Perform global optimization using dual_annealing
    # maxiter is increased from default (2000) to 5000 to allow for a more thorough search
    # and potentially achieve a higher quality solution, prioritizing quality over speed.
    # initial_temp is set to a reasonable value to start the annealing process.
    # no_local_search=False ensures a local search is performed after the annealing phase
    # to refine the point positions, which is typically beneficial for precision.
    result = dual_annealing(
        func=objective_function, 
        bounds=bounds, 
        x0=x0, 
        seed=optimization_seed, 
        maxiter=5000, 
        initial_temp=5e3, 
        no_local_search=False 
    )
    
    # Reshape the optimized flattened coordinates back into (N_POINTS, 2)
    optimized_points = result.x.reshape((N_POINTS, 2))
            
    return optimized_points

# EVOLVE-BLOCK-END
