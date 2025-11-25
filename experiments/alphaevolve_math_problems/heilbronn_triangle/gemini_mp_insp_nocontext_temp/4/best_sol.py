# EVOLVE-BLOCK-START
import numpy as np
import itertools
from scipy.optimize import differential_evolution
import time

# Constants for the bounding equilateral triangle
SQRT3 = np.sqrt(3)
TRIANGLE_VERTICES = np.array([[0.0, 0.0], [1.0, 0.0], [0.5, SQRT3 / 2]])
BOUNDING_TRIANGLE_AREA = SQRT3 / 4 # Area of the unit equilateral triangle

def calculate_triangle_area(p1, p2, p3):
    """Calculates the area of a triangle given three points using the determinant formula."""
    return 0.5 * np.abs(p1[0]*(p2[1]-p3[1]) + p2[0]*(p3[1]-p1[1]) + p3[0]*(p1[1]-p2[1]))

def is_point_inside_triangle(point, vertices, epsilon=1e-9):
    """Checks if a point is inside or on the boundary of the equilateral triangle.
    The triangle is defined by vertices (0,0), (1,0), and (0.5, sqrt(3)/2).
    A point (x, y) is inside if it satisfies three linear inequalities:
    1. y >= 0 (above or on the base)
    2. y <= sqrt(3) * x (below or on the left edge)
    3. y <= -sqrt(3) * (x - 1) (below or on the right edge)
    Epsilon is used for robust floating-point comparisons to include boundary points.
    """
    x, y = point
    # Condition 1: y >= 0
    if y < -epsilon: return False
    
    # Condition 2: y - sqrt(3) * x <= 0 (line from (0,0) to (0.5, sqrt(3)/2))
    if y - SQRT3 * x > epsilon: return False
    
    # Condition 3: y + sqrt(3) * (x - 1) <= 0 (line from (1,0) to (0.5, sqrt(3)/2))
    if y + SQRT3 * (x - 1) > epsilon: return False
    
    return True

def get_min_triangle_area(points):
    """Calculates the minimum area among all possible triangles formed by combinations of three points."""
    min_area = float('inf')
    num_points = points.shape[0]

    # For n=11, num_points will always be 11, so this check is mostly defensive.
    if num_points < 3:
        return 0.0

    # Iterate through all unique combinations of three points (C(n, 3) triplets)
    for i, j, k in itertools.combinations(range(num_points), 3):
        area = calculate_triangle_area(points[i], points[j], points[k])
        # Update minimum area found
        if area < min_area:
            min_area = area
    return min_area

def objective_function(points_flat, n_points, bounding_area, triangle_vertices):
    """
    Objective function for the Heilbronn triangle problem, designed for minimization.
    It returns the negative normalized minimum triangle area, incorporating penalties
    for invalid point placements or degenerate (near-zero area) triangles.
    """
    # Reshape the flattened 1D array of coordinates back into a 2D array of (x, y) points
    points = points_flat.reshape((n_points, 2))

    # --- Constraint Handling: Penalty for points outside the bounding equilateral triangle ---
    penalty = 0.0
    for p in points:
        if not is_point_inside_triangle(p, triangle_vertices):
            # Apply a large fixed penalty for each point found outside the valid region.
            # This strongly discourages the optimizer from exploring invalid regions,
            # guiding it towards solutions within the specified equilateral triangle.
            penalty += bounding_area * 1000 

    if penalty > 0:
        # If any point is outside, return a very high objective value.
        # This makes such solutions very "bad" for the minimizer.
        # Adding a small constant ensures this value is strictly worse than any valid solution,
        # including those with zero minimum area (which are also penalized).
        return penalty + 1.0 

    # --- Objective Calculation: Find the minimum triangle area ---
    min_area = get_min_triangle_area(points)

    # --- Constraint Handling: Penalty for degenerate triangles (near-zero area) ---
    # Solutions with collinear points or very close points result in near-zero areas.
    # These are highly undesirable for the Heilbronn problem, which seeks to MAXIMIZE
    # the minimum area. Thus, they are heavily penalized.
    if min_area < 1e-12: # Using a small threshold for numerical stability
        return 1.0 # Return a high objective value (bad solution)

    # --- Final Objective Value ---
    # The Heilbronn problem aims to MAXIMIZE the minimum normalized area.
    # Since optimizers typically MINIMIZE, we return the NEGATIVE of the normalized minimum area.
    return -min_area / bounding_area

def heilbronn_triangle11() -> np.ndarray:
    """
    Generates an optimal arrangement of exactly 11 points within or on the boundary of an
    equilateral triangle with vertices at (0,0), (1,0), and (0.5, sqrt(3)/2).
    The arrangement maximizes the minimum area of any triangle formed by three of these points.

    Returns:
        points: np.ndarray of shape (11,2) containing the (x, y) coordinates of the optimal points.
    """
    n = 11
    
    # Define bounds for the optimization. The `differential_evolution` optimizer expects
    # a list of (min, max) tuples, one for each dimension in the flattened parameter vector.
    # The parameter vector is structured as (x1, y1, x2, y2, ..., xN, yN).
    bounds = []
    for _ in range(n):
        bounds.append((0, 1))           # x-coordinate bounds (0 to 1, covering the base)
        bounds.append((0, SQRT3 / 2))   # y-coordinate bounds (0 to sqrt(3)/2, covering the height)

    # Run Differential Evolution for global optimization.
    # Differential Evolution is a robust metaheuristic well-suited for non-convex,
    # non-differentiable problems like the Heilbronn triangle problem.
    result = differential_evolution(
        func=objective_function,                      # The function to be minimized
        bounds=bounds,                                # Bounds for each parameter
        args=(n, BOUNDING_TRIANGLE_AREA, TRIANGLE_VERTICES), # Extra arguments for the objective function
        strategy='best1bin',                          # Optimization strategy (often a good default)
        popsize=30,                                   # Population size (number of candidate solutions per generation).
                                                      # Increased from default (15) for better exploration.
        maxiter=2000,                                 # Maximum number of generations.
                                                      # Increased from default (1000) to allow more convergence time.
        tol=0.01,                                     # Relative tolerance for convergence.
        seed=42,                                      # Fixed seed for reproducibility of results.
        workers=-1,                                   # Use all available CPU cores for parallel computation,
                                                      # which significantly speeds up evaluation.
        disp=False                                    # Suppress printing optimization progress to console.
    )

    # Check if the optimization converged successfully.
    if not result.success:
        # If the optimization did not converge or failed, print a warning.
        # In a production system, one might implement more sophisticated error handling,
        # such as returning a default known good configuration or raising an exception.
        print(f"Warning: Differential Evolution did not converge successfully: {result.message}")
        # For this problem, we still return the best solution found, even if not fully converged.
        
    # Reshape the optimal flattened coordinates back into a 2D array of (x, y) points.
    optimal_points_flat = result.x
    optimal_points = optimal_points_flat.reshape((n, 2))
    
    # Post-optimization clipping: ensure all points are strictly within the rectangular bounding box.
    # While the objective function's penalty term should largely keep points within the
    # triangular region, small numerical inaccuracies from the optimizer might push
    # points marginally outside the rectangular bounds. Clipping ensures strict adherence.
    for i in range(n):
        optimal_points[i, 0] = np.clip(optimal_points[i, 0], 0, 1)
        optimal_points[i, 1] = np.clip(optimal_points[i, 1], 0, SQRT3 / 2)
        # Note: This clipping only guarantees points are within the *rectangular* bounding box.
        # The `is_point_inside_triangle` check and penalty are crucial for the *triangular* boundary.

    return optimal_points

# EVOLVE-BLOCK-END
