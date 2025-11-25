# EVOLVE-BLOCK-START
import numpy as np
import itertools
from scipy.optimize import dual_annealing, minimize, LinearConstraint # Added LinearConstraint
from numba import jit # Using numba for performance
from scipy.stats import qmc # For quasi-random initial points

# Define constants for the equilateral triangle
SQRT3 = np.sqrt(3.0)
TRIANGLE_VERTICES = np.array([
    [0.0, 0.0],
    [1.0, 0.0],
    [0.5, SQRT3 / 2.0]
], dtype=np.float64)
DOMAIN_AREA = SQRT3 / 4.0
EPSILON = 1e-9 # Tolerance for floating point comparisons in geometric checks

# Global constant for number of points
_N_POINTS = 11

# Numba-optimized function to calculate triangle area
@jit(nopython=True, cache=True)
def _calculate_triangle_area_jit(p1: np.ndarray, p2: np.ndarray, p3: np.ndarray) -> float:
    """Calculates the area of a triangle given three points."""
    # Using the determinant formula: 0.5 * |x1(y2 - y3) + x2(y3 - y1) + x3(y1 - y2)|
    area = 0.5 * np.abs(p1[0]*(p2[1] - p3[1]) + p2[0]*(p3[1] - p1[1]) + p3[0]*(p1[1] - p2[1]))
    return area

# Numba-optimized function to calculate boundary penalty
@jit(nopython=True, cache=True)
def _calculate_boundary_penalty_jit(points: np.ndarray, penalty_multiplier: float, epsilon: float = EPSILON) -> float:
    """
    Calculates a penalty for points lying outside the equilateral triangle boundary.
    The triangle vertices are (0,0), (1,0), (0.5, sqrt(3)/2).
    A strong penalty is applied to strongly guide points back into the valid region.
    """
    penalty = 0.0

    for i in range(points.shape[0]):
        x, y = points[i, 0], points[i, 1]

        # Constraint 1: y >= 0
        if y < -epsilon:
            penalty += np.abs(y) * penalty_multiplier

        # Constraint 2: y <= sqrt(3) * x  (or sqrt(3) * x - y >= 0)
        # This line passes through (0,0) and (0.5, sqrt(3)/2)
        violation_left_line = SQRT3 * x - y
        if violation_left_line < -epsilon:
            penalty += np.abs(violation_left_line) * penalty_multiplier

        # Constraint 3: y <= -sqrt(3) * (x - 1) (or sqrt(3) * (1 - x) - y >= 0)
        # This line passes through (1,0) and (0.5, sqrt(3)/2)
        violation_right_line = SQRT3 * (1 - x) - y
        if violation_right_line < -epsilon:
            penalty += np.abs(violation_right_line) * penalty_multiplier
            
        # Also ensure points are within rectangular x bounds [0,1] for robust penalty
        if x < -epsilon: penalty += np.abs(x) * penalty_multiplier
        if x > 1 + epsilon: penalty += np.abs(x - 1) * penalty_multiplier

    return penalty

# Numba-optimized function to find the minimum area among all triplets
@jit(nopython=True, cache=True)
def _get_min_triangle_area_jit(points: np.ndarray) -> float:
    """
    Calculates the minimum area among all possible triangles formed by triplets of points.
    Returns 0.0 if any three points are collinear or identical (area is numerically zero).
    """
    min_area = np.inf
    n = points.shape[0]

    if n < 3:
        return 0.0

    # Iterate through all unique combinations of three points using a direct triple loop,
    # which is efficient and simple for Numba to optimize.
    for i in range(n):
        for j in range(i + 1, n):
            for k in range(j + 1, n):
                p1 = points[i]
                p2 = points[j]
                p3 = points[k]
                
                area = _calculate_triangle_area_jit(p1, p2, p3)
                
                # If area is very small, treat it as zero. This ensures that
                # collinear or identical points result in a 0 min_area,
                # which is the worst possible outcome and correctly handled by the optimizer.
                if area < 1e-12: # A small epsilon to account for floating point inaccuracies
                    return 0.0 # Early exit, as 0 is the absolute minimum we want to avoid

                if area < min_area:
                    min_area = area

    return min_area if min_area != np.inf else 0.0 # Should not happen for n>=3, but safe fallback

# Numba-optimized helper function for point containment
@jit(nopython=True, cache=True)
def _is_point_inside_equilateral_triangle_jit(point: np.ndarray, epsilon: float = EPSILON) -> bool:
    """Checks if a point is within or on the boundary of the unit equilateral triangle."""
    x, y = point
    if y < -epsilon: return False
    if SQRT3 * x - y < -epsilon: return False # Left edge: sqrt(3)*x - y >= 0
    if SQRT3 * (1 - x) - y < -epsilon: return False # Right edge: sqrt(3)*(1-x) - y >= 0
    return True

@jit(nopython=True, cache=True)
def _project_points_to_triangle_boundary_jit(points: np.ndarray, epsilon: float = EPSILON) -> np.ndarray:
    """
    Ensures all points are strictly within or on the boundary of the equilateral triangle
    by projecting any slightly outside points back. JIT compiled for performance.
    """
    projected_points = np.copy(points)
    for i in range(len(points)):
        x, y = projected_points[i]

        # Project onto y=0 if below
        if y < -epsilon: y = 0.0
        elif y < 0: y = 0.0 # Snap to boundary if slightly out

        # Project onto left edge (y = sqrt(3)x)
        # Violation: y - sqrt(3)x > 0
        val_left = SQRT3 * x - y
        if val_left < -epsilon: # Point is above the left line
            # Vector normal to line sqrt(3)x - y = 0 is (sqrt(3), -1)
            # Project point (x,y) onto line:
            # New point (x', y') = (x,y) - (val_left / (sqrt(3)^2 + (-1)^2)) * (sqrt(3), -1)
            # Simplified: (x', y') = (x,y) - (val_left / 4) * (sqrt(3), -1)
            x_proj = x - SQRT3 * val_left / 4.0
            y_proj = y - (-1) * val_left / 4.0
            x, y = x_proj, y_proj

        # Project onto right edge (y = -sqrt(3)(x-1))
        # Violation: y - (-sqrt(3)(x-1)) > 0
        val_right = SQRT3 * (1 - x) - y
        if val_right < -epsilon: # Point is above the right line
            # Vector normal to line sqrt(3)(1-x) - y = 0 is (-sqrt(3), -1)
            # New point (x', y') = (x,y) - (val_right / ((-sqrt(3))^2 + (-1)^2)) * (-sqrt(3), -1)
            # Simplified: (x', y') = (x,y) - (val_right / 4) * (-sqrt(3), -1)
            x_proj = x - (-SQRT3) * val_right / 4.0
            y_proj = y - (-1) * val_right / 4.0
            x, y = x_proj, y_proj
        
        projected_points[i] = [x, y]
    return projected_points


def _objective_function_factory(n_points: int, penalty_multiplier: float = 1e6, min_area_threshold: float = 1e-8):
    """
    Creates the objective function for global optimization (e.g., dual_annealing),
    including penalty for points outside the equilateral triangle.
    """
    def objective_fn(flat_points: np.ndarray) -> float:
        points = flat_points.reshape((n_points, 2))
        
        penalty = _calculate_boundary_penalty_jit(points, penalty_multiplier, EPSILON)
        min_area = _get_min_triangle_area_jit(points)
        
        # Penalize near-zero areas heavily to prevent degenerate solutions.
        if min_area < min_area_threshold:
            penalty += 1e7 * (min_area_threshold - min_area) 

        # Maximize min_area is equivalent to minimizing -min_area
        return -min_area + penalty
    
    return objective_fn

def _local_objective_function(flat_points: np.ndarray, n_points: int) -> float:
    """
    Creates the objective function for local optimization (e.g., SLSQP).
    It only minimizes -min_area, as boundary constraints are handled separately by SLSQP.
    Includes a penalty for degenerate configurations to improve stability.
    """
    points = flat_points.reshape((n_points, 2))
    min_area = _get_min_triangle_area_jit(points)
    
    # Add a large penalty if the configuration is degenerate (collinear points)
    # This helps guide the local optimizer away from unstable regions.
    if min_area < 1e-9:
        return 1e7

    return -min_area

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

def _generate_initial_points_in_triangle(n: int, seed: int = 42) -> np.ndarray:
    """
    Generates n quasi-random points using a Sobol sequence and barycentric mapping
    to ensure they are strictly inside the unit equilateral triangle.
    """
    # Using scipy.stats.qmc.Sobol for quasi-random sequence
    sampler_x0 = qmc.Sobol(d=2, scramble=True, seed=seed)
    rand_pairs_x0 = sampler_x0.random(n=n)

    # Transform points from unit square to a right triangle (0,0), (1,0), (0,1)
    # Then transform from right triangle to equilateral triangle
    # This is a common method to get uniform points in a triangle
    sqrt_r1_x0 = np.sqrt(rand_pairs_x0[:, 0])
    u_x0 = 1 - sqrt_r1_x0
    v_x0 = rand_pairs_x0[:, 1] * sqrt_r1_x0
    
    # Barycentric coordinates (u_x0, v_x0, 1-u_x0-v_x0) for points in a triangle
    # Mapping to the specific equilateral triangle vertices
    points_cartesian_x0 = (u_x0[:, np.newaxis] * TRIANGLE_VERTICES[0, :] +
                           v_x0[:, np.newaxis] * TRIANGLE_VERTICES[1, :] +
                           (1 - u_x0 - v_x0)[:, np.newaxis] * TRIANGLE_VERTICES[2, :])
    
    return points_cartesian_x0


def heilbronn_triangle11() -> np.ndarray:
    """
    Construct an arrangement of n points on or inside a convex region in order to maximize the area of the
    smallest triangle formed by these points. Here n = 11.

    Returns:
        points: np.ndarray of shape (11,2) with the x,y coordinates of the points.
    """
    n = _N_POINTS # Use global constant
    np.random.seed(42) # Ensure numpy operations are reproducible

    # Define rectangular bounds for each coordinate (x and y)
    bounds = [(0.0, 1.0), (0.0, SQRT3 / 2.0)] * n # n pairs of (x_min, x_max), (y_min, y_max)
    
    # Generate initial points using a quasi-random sequence
    initial_points = _generate_initial_points_in_triangle(n, seed=42)
    x0_global = initial_points.flatten()

    # Create the global objective function with penalty
    global_objective_fn = _objective_function_factory(n)

    # Minimizer kwargs for the local step within dual_annealing (optional, but good for refinement)
    minimizer_kwargs_global = {
        "method": "Nelder-Mead", # Nelder-Mead is robust for non-smooth, non-gradient problems
        "options": { "xatol": 1e-8, "fatol": 1e-8, "adaptive": True }
    }

    # Use Dual Annealing for global optimization.
    result_global = dual_annealing(
        global_objective_fn,
        bounds,
        x0=x0_global,                 # Provide initial points from quasi-random generation
        maxiter=20000,                # Increased global search iterations for a final quality push
        initial_temp=5230.0,          # Set initial temperature (from inspiration)
        seed=42,                      # Seed for reproducibility
        minimizer_kwargs=minimizer_kwargs_global, # Use Nelder-Mead for local steps within DA
    )

    if not result_global.success:
        print(f"Dual Annealing did not converge successfully: {result_global.message}")

    optimal_flat_points_global = result_global.x

    # Create the local objective function (without penalty, as constraints are explicit)
    local_objective_fn = lambda flat_points: _local_objective_function(flat_points, n)
    
    # Define linear constraints for the equilateral triangle
    triangle_constraints = _get_linear_constraints(n)

    # Local refinement using scipy.optimize.minimize with SLSQP and explicit constraints
    result_local = minimize(
        fun=local_objective_fn,
        x0=optimal_flat_points_global, # Start from the best DA solution
        method='SLSQP',                # SLSQP handles linear constraints explicitly
        bounds=bounds,                 # Rectangular bounds still apply
        constraints=triangle_constraints, # Apply the specific triangle constraints
        options={'maxiter': 25000, 'ftol': 1e-10, 'gtol': 1e-7} # Increased local refinement iterations for a final quality push
    )
    
    if not result_local.success:
        print(f"Local optimization (SLSQP) did not converge successfully: {result_local.message}")

    optimal_flat_points = result_local.x # Use the locally refined points as the final result
    optimized_points = optimal_flat_points.reshape((n, 2))
    
    # Post-optimization projection to ensure strict boundary adherence
    final_points = _project_points_to_triangle_boundary_jit(optimized_points, epsilon=EPSILON)

    # Post-optimization check to verify boundary adherence.
    final_penalty = _calculate_boundary_penalty_jit(final_points, penalty_multiplier=1e6, epsilon=EPSILON)
    if final_penalty > 1e-6: # A small tolerance for floating point inaccuracies
        print(f"Warning: Final optimal points have a non-zero boundary penalty: {final_penalty}. "
              "This indicates a potential issue with constraint satisfaction even after projection.")

    return final_points

# EVOLVE-BLOCK-END
