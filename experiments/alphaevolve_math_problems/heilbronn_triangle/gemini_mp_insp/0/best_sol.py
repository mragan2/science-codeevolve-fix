# EVOLVE-BLOCK-START
import numpy as np
# itertools is not used in the Numba-optimized loops, so it can be removed.
from scipy.optimize import dual_annealing, minimize, LinearConstraint
from numba import jit
from scipy.stats import qmc # Added for qmc.Sobol (from Inspiration 1 & 3)

# Define constants for the equilateral triangle
SQRT3 = np.sqrt(3.0)
TRIANGLE_VERTICES = np.array([
    [0.0, 0.0],
    [1.0, 0.0],
    [0.5, SQRT3 / 2.0]
], dtype=np.float64)
DOMAIN_AREA = SQRT3 / 4.0
EPSILON = 1e-9 # Global epsilon for robust geometric checks (from Inspiration 1 & 3)

# Numba-optimized function to calculate triangle area
@jit(nopython=True, cache=True)
def calculate_triangle_area(p1: np.ndarray, p2: np.ndarray, p3: np.ndarray) -> float:
    """Calculates the area of a triangle given three points."""
    area = 0.5 * np.abs(p1[0]*(p2[1] - p3[1]) + p2[0]*(p3[1] - p1[1]) + p3[0]*(p1[1] - p2[1]))
    return area

# Numba-optimized function to calculate boundary penalty (adapted from Inspiration 1 & 3)
@jit(nopython=True, cache=True)
def calculate_boundary_penalty(points: np.ndarray, penalty_multiplier: float, epsilon: float) -> float:
    """
    Calculates a penalty for points lying outside the equilateral triangle boundary.
    A strong penalty is applied to strongly guide points back into the valid region.
    Uses EPSILON for robust checks.
    """
    penalty = 0.0

    for i in range(points.shape[0]):
        x, y = points[i, 0], points[i, 1]

        # Constraint 1: y >= 0
        if y < -epsilon:
            penalty += np.abs(y) * penalty_multiplier

        # Constraint 2: y <= sqrt(3) * x  (or sqrt(3) * x - y >= 0)
        val_left_edge = SQRT3 * x - y
        if val_left_edge < -epsilon:
            penalty += np.abs(val_left_edge) * penalty_multiplier

        # Constraint 3: y <= -sqrt(3) * (x - 1) (or sqrt(3) * (1 - x) - y >= 0)
        val_right_edge = SQRT3 * (1 - x) - y
        if val_right_edge < -epsilon:
            penalty += np.abs(val_right_edge) * penalty_multiplier
            
        # Also ensure points are within rectangular x bounds [0,1] for robust penalty
        if x < -epsilon: penalty += np.abs(x) * penalty_multiplier
        if x > 1 + epsilon: penalty += np.abs(x - 1) * penalty_multiplier

    return penalty

# Numba-optimized function to find the minimum area among all triplets
@jit(nopython=True, cache=True)
def get_min_area(points: np.ndarray) -> float:
    """
    Calculates the minimum area among all possible triangles formed by triplets of points.
    Returns 0.0 if any three points are collinear or identical (area is numerically zero).
    """
    min_area = np.inf
    n = points.shape[0]

    for i in range(n):
        for j in range(i + 1, n):
            for k in range(j + 1, n):
                p1 = points[i]
                p2 = points[j]
                p3 = points[k]
                
                area = calculate_triangle_area(p1, p2, p3)
                
                if area < 1e-12: # Consistent epsilon for zero area check
                    return 0.0

                if area < min_area:
                    min_area = area

    return min_area if min_area != np.inf else 0.0

# Numba-optimized helper function for point containment (from Inspiration 1 & 3)
@jit(nopython=True, cache=True)
def _is_point_inside_equilateral_triangle_jit(point: np.ndarray, epsilon: float = EPSILON) -> bool:
    """Checks if a point is within or on the boundary of the unit equilateral triangle."""
    x, y = point
    if y < -epsilon: return False
    if SQRT3 * x - y < -epsilon: return False # Left edge: sqrt(3)*x - y >= 0
    if SQRT3 * (1 - x) - y < -epsilon: return False # Right edge: sqrt(3)*(1-x) - y >= 0
    return True

# Numba-optimized function for post-optimization projection (from Inspiration 1 & 3)
@jit(nopython=True, cache=True)
def _project_points_to_triangle_boundary_jit(points: np.ndarray, epsilon: float = EPSILON) -> np.ndarray:
    """
    Ensures all points are strictly within or on the boundary of the equilateral triangle
    by projecting any slightly outside points back. JIT compiled for performance.
    """
    projected_points = np.copy(points)
    for i in range(len(points)):
        x, y = projected_points[i]

        # Clamp y to be non-negative
        if y < -epsilon:
            y = 0.0
        elif y < 0: # If very close to 0, snap to 0
            y = 0.0
        
        # Clamp to left edge: y = sqrt(3) * x (line: sqrt(3)*x - y = 0)
        if SQRT3 * x - y < -epsilon:
            val = SQRT3 * x - y
            # Project onto the line using the formula for closest point on line ax+by+c=0 to (x0,y0)
            # x_proj = x0 - a*f(x0,y0)/(a^2+b^2), y_proj = y0 - b*f(x0,y0)/(a^2+b^2)
            # For SQRT3*x - y = 0, a=SQRT3, b=-1, c=0
            x_proj = x - SQRT3 * val / (SQRT3*SQRT3 + (-1)*(-1))
            y_proj = y - (-1) * val / (SQRT3*SQRT3 + (-1)*(-1))
            x, y = x_proj, y_proj
        
        # Clamp to right edge: y = -sqrt(3) * (x - 1) (line: sqrt(3)*x + y - sqrt(3) = 0)
        if SQRT3 * (1 - x) - y < -epsilon:
            val = SQRT3 * x + y - SQRT3 # This is the function for sqrt(3)*x + y - sqrt(3) = 0
            # For SQRT3*x + y - SQRT3 = 0, a=SQRT3, b=1, c=-SQRT3
            x_proj = x - SQRT3 * val / (SQRT3*SQRT3 + 1*1)
            y_proj = y - 1 * val / (SQRT3*SQRT3 + 1*1)
            x, y = x_proj, y_proj

        projected_points[i] = [x, y]
    return projected_points


def objective_function_for_optimizer(flat_points: np.ndarray, n_points: int) -> float:
    """
    Objective function for the optimizer.
    Takes a flattened array of points, reshapes it, calculates min_area and penalty,
    and returns -min_area + penalty (because optimizers minimize).
    """
    points = flat_points.reshape((n_points, 2))
    
    penalty_multiplier = 1e6 # Adjusted multiplier from inspirations
    min_area_threshold = 1e-8 # From inspirations

    penalty = calculate_boundary_penalty(points, penalty_multiplier, EPSILON)
    min_area = get_min_area(points)

    # Penalize near-zero areas heavily to prevent degenerate solutions.
    if min_area < min_area_threshold:
        penalty += 1e7 * (min_area_threshold - min_area) 

    # We want to maximize min_area, so we minimize -min_area.
    # Add penalty if constraints are violated.
    return -min_area + penalty

def objective_function_for_local_slsqp(flat_points: np.ndarray) -> float:
    """
    Objective function for the LOCAL optimizer (SLSQP).
    Takes a flattened array of points, reshapes it, calculates min_area.
    Boundary penalties are not needed, but a penalty for degenerate (near-zero area)
    triangles is added to guide the search away from bad local minima, inspired by Insp. 1.
    """
    n_points = len(flat_points) // 2
    points = flat_points.reshape((n_points, 2))
    min_area = get_min_area(points)
    
    min_area_threshold = 1e-8
    # If the area is near-zero, return a large penalty to create a steep gradient
    # pushing the optimizer away from this degenerate configuration.
    if min_area < min_area_threshold:
        return 1e7 * (min_area_threshold - min_area) - min_area

    # We want to maximize min_area, so we minimize -min_area.
    return -min_area


# Modified initial point generation using qmc.Sobol + Barycentric mapping (from Inspiration 1 & 3)
def _generate_initial_points_in_triangle(n, seed=42):
    """
    Generates n quasi-random points using a Sobol sequence and barycentric mapping
    to ensure they are strictly inside the unit equilateral triangle.
    """
    np.random.seed(seed) # For reproducibility of any internal random calls in qmc.Sobol
    
    sampler_x0 = qmc.Sobol(d=2, scramble=True, seed=seed)
    rand_pairs_x0 = sampler_x0.random(n=n) # Generate n 2D points from unit square

    # Map points from a unit square to a uniform distribution within the triangle
    # using a standard barycentric coordinate mapping.
    # P = u*V0 + v*V1 + w*V2, where u+v+w=1 and u,v,w >= 0.
    # V0=(0,0), V1=(1,0), V2=(0.5, sqrt(3)/2)
    sqrt_r1_x0 = np.sqrt(rand_pairs_x0[:, 0])
    u_x0 = 1 - sqrt_r1_x0
    v_x0 = rand_pairs_x0[:, 1] * sqrt_r1_x0
    w_x0 = 1 - u_x0 - v_x0
    
    # Convert from barycentric to Cartesian coordinates.
    # Since V0=(0,0) is the origin, P = v*TRIANGLE_VERTICES[1] + w*TRIANGLE_VERTICES[2].
    points_cartesian_x0 = (v_x0[:, np.newaxis] * TRIANGLE_VERTICES[1, :] +
                           w_x0[:, np.newaxis] * TRIANGLE_VERTICES[2, :])
    
    return points_cartesian_x0

def heilbronn_triangle11() -> np.ndarray:
    """
    Construct an arrangement of n points on or inside a convex region in order to maximize the area of the
    smallest triangle formed by these points. Here n = 11.

    Returns:
        points: np.ndarray of shape (11,2) with the x,y coordinates of the points.
    """
    n = 11
    
    # Set random seed for reproducibility
    np.random.seed(42)

    # Define rectangular bounds for each coordinate (x and y)
    max_y = SQRT3 / 2.0
    bounds = [(0.0, 1.0), (0.0, max_y)] * n
    
    # Define linear inequality constraints for the equilateral triangle (from Target and Inspirations)
    # 1. y_i >= 0
    # 2. SQRT3 * x_i - y_i >= 0
    # 3. SQRT3 * (1 - x_i) - y_i >= 0 (equivalent to -SQRT3 * x_i - y_i + SQRT3 >= 0)
    
    A = np.zeros((3 * n, 2 * n))
    lb = np.zeros(3 * n)
    ub = np.full(3 * n, np.inf)

    for i in range(n):
        # Constraint 1: y_i >= 0
        A[3*i + 0, 2*i + 1] = 1.0
        lb[3*i + 0] = 0.0

        # Constraint 2: SQRT3 * x_i - y_i >= 0
        A[3*i + 1, 2*i + 0] = SQRT3
        A[3*i + 1, 2*i + 1] = -1.0
        lb[3*i + 1] = 0.0

        # Constraint 3: SQRT3 * (1 - x_i) - y_i >= 0  => -SQRT3 * x_i - y_i >= -SQRT3
        A[3*i + 2, 2*i + 0] = -SQRT3
        A[3*i + 2, 2*i + 1] = -1.0
        lb[3*i + 2] = -SQRT3
        
    constraints = LinearConstraint(A, lb, ub)

    # Generate initial points using Sobol sequence and barycentric mapping (from inspirations)
    initial_points = _generate_initial_points_in_triangle(n, seed=42)
    x0 = initial_points.flatten()

    # Define minimizer kwargs for dual_annealing's internal local search (from Inspiration 1 & 3)
    minimizer_kwargs_global = {
        "method": "Nelder-Mead",
        "options": {
            "xatol": 1e-8, 
            "fatol": 1e-8,
            "adaptive": True,
        }
    }

    # Use Dual Annealing for global optimization (parameters from Inspiration 1 & 3)
    result_da = dual_annealing(
        func=lambda flat_points: objective_function_for_optimizer(flat_points, n), # Pass n
        bounds=bounds,
        x0=x0, # Start from Sobol initial points
        maxiter=18000, # Reduced maxiter for a better balance of quality and speed.
        initial_temp=5230.0, # Initial temp from inspirations
        seed=42,
        minimizer_kwargs=minimizer_kwargs_global, # Use Nelder-Mead for internal local search
    )

    if not result_da.success:
        print(f"Dual Annealing did not converge successfully: {result_da.message}")

    optimal_flat_points_da = result_da.x

    # Local refinement using scipy.optimize.minimize with SLSQP and explicit constraints.
    result_local = minimize(
        fun=objective_function_for_local_slsqp,
        x0=optimal_flat_points_da,
        bounds=bounds,
        method='SLSQP',
        constraints=constraints,
        options={'maxiter': 25000, 'ftol': 1e-10, 'gtol': 1e-7} # Reduced maxiter for a better balance of quality and speed.
    )
    
    if not result_local.success:
        print(f"Local optimization (SLSQP) did not converge successfully: {result_local.message}")

    optimal_flat_points = result_local.x
    optimal_points = optimal_flat_points.reshape((n, 2))
    
    # Perform a final projection to ensure all points are strictly within the triangle
    # Use the Numba-jitted projection function (from Inspiration 1 & 3)
    # Using a slightly larger epsilon for projection can be more robust.
    final_points = _project_points_to_triangle_boundary_jit(optimal_points, epsilon=1e-6)

    # Final validation check (optional, primarily for debugging)
    # Using the Numba-jitted point-in-triangle check
    for p in final_points:
        if not _is_point_inside_equilateral_triangle_jit(p, epsilon=EPSILON):
            print(f"Warning: Point {p} is outside the triangle after final projection.")

    return final_points

# EVOLVE-BLOCK-END
