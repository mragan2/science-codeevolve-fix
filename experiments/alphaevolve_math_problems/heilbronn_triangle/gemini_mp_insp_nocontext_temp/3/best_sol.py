# EVOLVE-BLOCK-START
import numpy as np
from numba import jit
from scipy.optimize import differential_evolution, NonlinearConstraint

# Define constants for the equilateral triangle
SQRT3 = np.sqrt(3.0)
TRIANGLE_VERTICES = np.array([[0.0, 0.0], [1.0, 0.0], [0.5, SQRT3 / 2]])
AREA_DOMAIN = SQRT3 / 4.0
N_POINTS = 11

@jit(nopython=True)
def calculate_triangle_area(p1, p2, p3):
    """Calculates the absolute area of a triangle given three 2D points."""
    return 0.5 * np.abs(p1[0]*(p2[1]-p3[1]) + p2[0]*(p3[1]-p1[1]) + p3[0]*(p1[1]-p2[1]))

# Removed _sign_helper and is_point_inside_triangle as they are no longer needed
# with the barycentric coordinate approach.

@jit(nopython=True)
def find_min_area(points):
    """
    Calculates the minimum triangle area formed by any three points in the set.
    Assumes points is a (N, 2) numpy array.
    """
    min_area = np.inf
    num_points = points.shape[0]

    # Iterate through all combinations of 3 points
    # Using nested loops for numba compatibility with combinations
    for i in range(num_points):
        for j in range(i + 1, num_points):
            for k in range(j + 1, num_points):
                area = calculate_triangle_area(points[i], points[j], points[k])
                if area < min_area:
                    min_area = area
    return min_area

@jit(nopython=True)
def barycentric_to_cartesian(bary_coords_flat, tri_verts):
    """
    Converts N points from barycentric coordinates (u, v) to Cartesian (x, y).
    bary_coords_flat: 1D array [u1, v1, u2, v2, ..., uN, vN]
    tri_verts: (3, 2) array of triangle vertices.
    """
    num_points = bary_coords_flat.shape[0] // 2
    cartesian_points = np.empty((num_points, 2), dtype=np.float64)

    V1 = tri_verts[0]
    V2 = tri_verts[1]
    V3 = tri_verts[2]

    for i in range(num_points):
        u = bary_coords_flat[2 * i]
        v = bary_coords_flat[2 * i + 1]
        w = 1.0 - u - v # w is implicitly >= 0 due to constraints

        # P = u*V1 + v*V2 + w*V3
        cartesian_points[i, 0] = u * V1[0] + v * V2[0] + w * V3[0]
        cartesian_points[i, 1] = u * V1[1] + v * V2[1] + w * V3[1]
        
    return cartesian_points

def objective_function(bary_coords_flat):
    """
    Objective function for differential_evolution, using barycentric coordinates.
    Takes a 1D array of (u, v) pairs for N points, converts to Cartesian,
    and returns the negative of the minimum triangle area (for maximization).
    The constraint u+v <= 1 is handled by NonlinearConstraint, so no penalty here.
    """
    points = barycentric_to_cartesian(bary_coords_flat, TRIANGLE_VERTICES)
    min_area = find_min_area(points)
    return -min_area # Minimize -min_area to maximize min_area

# Constraint function for differential_evolution
def barycentric_constraint(bary_coords_flat):
    """
    Constraint function for differential_evolution to ensure u_i + v_i <= 1 for all points.
    Returns an array of (1 - u_i - v_i) values for each point.
    These must be >= 0.
    """
    num_points = bary_coords_flat.shape[0] // 2
    constraints = np.empty(num_points, dtype=np.float64)
    for i in range(num_points):
        u = bary_coords_flat[2 * i]
        v = bary_coords_flat[2 * i + 1]
        constraints[i] = 1.0 - u - v
    return constraints

def heilbronn_triangle11() -> np.ndarray:
    """
    Construct an arrangement of 11 points on or inside an equilateral triangle
    to maximize the area of the smallest triangle formed by these points.

    Returns:
        points: np.ndarray of shape (11,2) with the x,y coordinates of the points.
    """
    # Define bounds for barycentric u and v coordinates: (0, 1) for each
    # Each point has (u, v), so 2*N_POINTS variables
    bounds = [(0, 1)] * (2 * N_POINTS)

    # Set a fixed random seed for reproducibility
    seed = 42

    # Define the nonlinear constraint for barycentric coordinates: u + v <= 1
    # The constraint function returns (1 - u - v) for each point.
    # We want 1 - u - v >= 0.
    constraints = NonlinearConstraint(barycentric_constraint, 0, np.inf)

    # Perform optimization using Differential Evolution
    result = differential_evolution(
        objective_function,
        bounds,
        constraints=constraints, # Pass the defined constraints
        strategy='best1bin',
        maxiter=4000,         # Increased maxiter for potentially harder search with constraints
        popsize=30,           # Increased popsize to explore more of the constrained space
        tol=0.001,            # Adjusted tolerance for convergence
        mutation=(0.5, 1),
        recombination=0.7,
        seed=seed,
        disp=False,
        workers=-1
    )

    if not result.success:
        print(f"Warning: Differential Evolution did not converge successfully. Message: {result.message}")
        print(f"However, returning the best solution found: {-result.fun} min_area.")

    # Convert the optimized barycentric coordinates back to Cartesian
    optimal_barycentric = result.x
    optimal_points = barycentric_to_cartesian(optimal_barycentric, TRIANGLE_VERTICES)

    # No need for final validation with is_point_inside_triangle, as barycentric coordinates
    # with the constraint inherently guarantee points are within the triangle.
    
    return optimal_points

# EVOLVE-BLOCK-END
