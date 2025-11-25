# EVOLVE-BLOCK-START
import numpy as np
import itertools
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

_N_POINTS = 11 # Global constant for the number of points (from Inspiration 1 & 3)
# Pre-compute combination indices for fast, vectorized area calculation (from Inspiration 1 & 3)
_P_INDICES = np.array(list(itertools.combinations(range(_N_POINTS), 3)), dtype=np.intp)

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
def get_min_area(points: np.ndarray, p_indices: np.ndarray) -> float:
    """
    Calculates the minimum area among all possible triangles formed by triplets of points,
    using precomputed indices for efficiency with Numba.
    Returns 0.0 if any three points are collinear or identical (area is numerically zero).
    """
    min_area = np.inf
    n_combinations = p_indices.shape[0]

    if points.shape[0] < 3: # Handle case where not enough points for a triangle
        return 0.0

    # Iterate through all unique combinations of three points using precomputed indices
    for i in range(n_combinations):
        p1 = points[p_indices[i, 0]]
        p2 = points[p_indices[i, 1]]
        p3 = points[p_indices[i, 2]]
                
        area = calculate_triangle_area(p1, p2, p3)
                
        if area < 1e-12: # A small epsilon to account for floating point inaccuracies
            return 0.0 # Early exit, as 0 is the absolute minimum we want to avoid

        if area < min_area:
            min_area = area

    return min_area if min_area != np.inf else 0.0 # Should not happen for n>=3, but safe fallback

# Numba-optimized helper function for point containment (from Inspiration 1 & 3)
@jit(nopython=True, cache=True)
def _is_point_inside_equilateral_triangle_jit(point: np.ndarray, epsilon: float = EPSILON) -> bool:
    """Checks if a point is within or on the boundary of the unit equilateral triangle."""
    x, y = point
    if y < -epsilon: return False
    if SQRT3 * x - y < -epsilon: return False # Left edge: sqrt(3)*x - y >= 0
    if SQRT3 * (1 - x) - y < -epsilon: return False # Right edge: sqrt(3)*(1-x) - y >= 0
    return True

# Numba-optimized function for post-optimization projection (adapted from Inspiration 1 & 3)
@jit(nopython=True, cache=True)
def _project_points_to_triangle_boundary_jit(points: np.ndarray, epsilon: float = 1e-6) -> np.ndarray: # Changed default epsilon for robustness
    """
    Ensures all points are strictly within or on the boundary of the equilateral triangle
    by projecting any slightly outside points back. JIT compiled for performance.
    The projection method is robust and handles numerical inaccuracies by clamping
    coordinates to the nearest point on the boundary if slightly outside.
    """
    projected_points = np.copy(points)
    for i in range(len(points)):
        x, y = projected_points[i]

        # Clamp y to be non-negative
        if y < -epsilon:
            y = 0.0
        elif y < 0: # If very close to 0 but positive, snap to 0
            y = 0.0
        
        # Clamp to left edge: y = sqrt(3) * x (line: sqrt(3)*x - y = 0)
        if SQRT3 * x - y < -epsilon:
            val = SQRT3 * x - y
            # Project onto the line using the formula for closest point on line ax+by+c=0 to (x0,y0)
            # x_proj = x0 - a*f(x0,y0)/(a^2+b^2), y_proj = y0 - b*f(x0,y0)/(a^2+b^2)
            # For SQRT3*x - y = 0, a=SQRT3, b=-1, c=0. a^2+b^2 = 3 + 1 = 4.
            x_proj = x - SQRT3 * val / 4.0
            y_proj = y - (-1.0) * val / 4.0
            x, y = x_proj, y_proj
        
        # Clamp to right edge: y = -sqrt(3) * (x - 1) (line: sqrt(3)*x + y - sqrt(3) = 0)
        if SQRT3 * (1 - x) - y < -epsilon:
            val = SQRT3 * x + y - SQRT3 # This is the function for sqrt(3)*x + y - sqrt(3) = 0
            # For SQRT3*x + y - SQRT3 = 0, a=SQRT3, b=1, c=-SQRT3. a^2+b^2 = 3 + 1 = 4.
            x_proj = x - SQRT3 * val / 4.0
            y_proj = y - 1.0 * val / 4.0
            x, y = x_proj, y_proj

        # Final check to ensure x within [0,1] after projection,
        # in case projections moved points slightly outside these bounds. (from Inspiration 3)
        if x < -epsilon: x = 0.0
        if x > 1.0 + epsilon: x = 1.0

        projected_points[i] = [x, y]
    return projected_points


def _objective_function_factory(n: int, p_indices: np.ndarray): # Modified to accept p_indices and be a factory
    """
    Creates the objective function for global optimization (e.g., dual_annealing),
    including penalty for points outside the equilateral triangle.
    """
    penalty_multiplier = 1e6
    min_area_threshold = 1e-8

    def objective_fn(flat_points: np.ndarray) -> float:
        points = flat_points.reshape((n, 2))
        
        penalty = calculate_boundary_penalty(points, penalty_multiplier, EPSILON)
        min_area = get_min_area(points, p_indices) # Pass p_indices
        
        # Penalize near-zero areas heavily to prevent degenerate solutions.
        if min_area < min_area_threshold:
            penalty += 1e7 * (min_area_threshold - min_area) 

        # Maximize min_area is equivalent to minimizing -min_area
        return -min_area + penalty
    
    return objective_fn

def _local_objective_function(flat_points: np.ndarray, n: int, p_indices: np.ndarray) -> float:
    """
    Creates the objective function for local optimization (e.g., SLSQP).
    It minimizes -min_area, as boundary constraints are handled separately by SLSQP.
    Includes a proportional penalty for degenerate configurations to improve stability
    and provide a gradient away from zero-area solutions (adapted from Inspiration 2 & 3).
    """
    points = flat_points.reshape((n, 2))
    min_area = get_min_area(points, p_indices)
    
    # Use a proportional penalty for degenerate (near-zero area) triangles.
    # This creates a steep gradient away from degenerate solutions, which is more effective
    # than a fixed large penalty for gradient-based local optimizers like SLSQP.
    min_area_threshold = 1e-8 # Consistent threshold from global objective
    if min_area < min_area_threshold:
        return 1e7 * (min_area_threshold - min_area) - min_area # Minimize this value, so -min_area is added

    # Maximize min_area is equivalent to minimizing -min_area.
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
    # w_x0 is implicitly 1 - u_x0 - v_x0, used directly below (minor optimization from Inspiration 1)
    
    # Convert from barycentric to Cartesian coordinates.
    # P = u*V0 + v*V1 + w*V2. Since V0=(0,0) is the origin, P = v*V1 + w*V2.
    points_cartesian_x0 = (v_x0[:, np.newaxis] * TRIANGLE_VERTICES[1, :] +
                           (1 - u_x0 - v_x0)[:, np.newaxis] * TRIANGLE_VERTICES[2, :]) # Use (1-u-v) for w
    
    return points_cartesian_x0

# Define _get_linear_constraints (moved from inside heilbronn_triangle11 for better structure)
def _get_linear_constraints(n_points: int) -> LinearConstraint:
    """
    Defines the linear constraints for points to be within or on the boundary
    of the unit equilateral triangle using a LinearConstraint object for efficiency.
    """
    num_vars = 2 * n_points
    A = np.zeros((3 * n_points, num_vars))
    lb = np.zeros(3 * n_points)
    ub = np.full(3 * n_points, np.inf)

    for i in range(n_points):
        # Constraint 1: y_i >= 0  (1*y_i >= 0)
        A[3*i, 2*i + 1] = 1.0
        lb[3*i] = 0.0

        # Constraint 2: SQRT3 * x_i - y_i >= 0
        A[3*i + 1, 2*i] = SQRT3
        A[3*i + 1, 2*i + 1] = -1.0
        lb[3*i + 1] = 0.0

        # Constraint 3: SQRT3 * (1 - x_i) - y_i >= 0  => -SQRT3 * x_i - y_i >= -SQRT3
        A[3*i + 2, 2*i] = -SQRT3
        A[3*i + 2, 2*i + 1] = -1.0
        lb[3*i + 2] = -SQRT3
        
    return LinearConstraint(A, lb, ub)

def heilbronn_triangle11() -> np.ndarray:
    """
    Construct an arrangement of n points on or inside a convex region in order to maximize the area of the
    smallest triangle formed by these points. Here n = 11.

    Returns:
        points: np.ndarray of shape (11,2) with the x,y coordinates of the points.
    """
    n = _N_POINTS # Use global constant
    
    # Set random seed for reproducibility
    np.random.seed(42)

    # Define rectangular bounds for each coordinate (x and y)
    max_y = SQRT3 / 2.0
    bounds = [(0.0, 1.0), (0.0, max_y)] * n
        
    constraints = _get_linear_constraints(n) # Call the top-level helper function

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

    # Create the global objective function using the factory
    global_objective_fn = _objective_function_factory(n, _P_INDICES)

    # Use Dual Annealing for global optimization (parameters from Inspiration 1 & 3)
    result_da = dual_annealing(
        func=global_objective_fn, # Use the created objective function
        bounds=bounds,
        x0=x0, # Start from Sobol initial points
        maxiter=20000, # Increased maxiter for better global search (from Inspiration 2)
        initial_temp=5230.0, # Initial temp from inspirations
        seed=42,
        minimizer_kwargs=minimizer_kwargs_global, # Use Nelder-Mead for internal local search
    )

    if not result_da.success:
        print(f"Dual Annealing did not converge successfully: {result_da.message}")

    optimal_flat_points_da = result_da.x

    # Create the local objective function using the factory
    local_objective_fn = lambda flat_points: _local_objective_function(flat_points, n, _P_INDICES)

    # Local refinement using scipy.optimize.minimize with SLSQP and explicit constraints.
    result_local = minimize(
        fun=local_objective_fn, # Use the created objective function
        x0=optimal_flat_points_da,
        bounds=bounds,
        method='SLSQP',
        constraints=constraints,
        options={'maxiter': 25000, 'ftol': 1e-10, 'gtol': 1e-7} # Increased maxiter and stricter ftol (from Inspiration 2)
    )
    
    if not result_local.success:
        print(f"Local optimization (SLSQP) did not converge successfully: {result_local.message}")

    optimal_flat_points = result_local.x
    optimal_points = optimal_flat_points.reshape((n, 2))
    
    # Perform a final projection to ensure all points are strictly within the triangle.
    # Use a slightly larger epsilon for the final projection for robustness (from Inspiration 1 & 2).
    final_points = _project_points_to_triangle_boundary_jit(optimal_points, epsilon=1e-7)

    # Final validation check (optional, primarily for debugging):
    # Using the Numba-jitted point-in-triangle check with a slightly more forgiving epsilon
    # to account for minimal floating point inaccuracies.
    for p in final_points:
        if not _is_point_inside_equilateral_triangle_jit(p, epsilon=1e-6):
            print(f"Warning: Point {p} is outside the triangle after final projection.")
            # Optionally, re-project if a point is still outside, or raise an error.
            # For this problem, a warning is sufficient as the projection is robust.

    return final_points

# EVOLVE-BLOCK-END
