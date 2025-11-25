# EVOLVE-BLOCK-START
import numpy as np
import itertools # Added for precomputing combinations
from scipy.optimize import dual_annealing, minimize, LinearConstraint
from numba import jit
from scipy.stats import qmc

# Define constants for the equilateral triangle
SQRT3 = np.sqrt(3.0)
TRIANGLE_VERTICES = np.array([
    [0.0, 0.0],
    [1.0, 0.0],
    [0.5, SQRT3 / 2.0]
], dtype=np.float64)
DOMAIN_AREA = SQRT3 / 4.0
EPSILON = 1e-9 # Global epsilon for robust geometric checks

# Global constants for Numba optimization (from Inspiration 2)
_N_POINTS = 11
_P_INDICES = np.array(list(itertools.combinations(range(_N_POINTS), 3)), dtype=np.intp)

# Numba-optimized function to calculate triangle area
@jit(nopython=True, cache=True)
def calculate_triangle_area(p1: np.ndarray, p2: np.ndarray, p3: np.ndarray) -> float:
    """Calculates the area of a triangle given three points."""
    area = 0.5 * np.abs(p1[0]*(p2[1] - p3[1]) + p2[0]*(p3[1] - p1[1]) + p3[0]*(p1[1] - p2[1]))
    return area

# Numba-optimized function to calculate boundary penalty (adapted from inspirations, removed redundant sqrt3_val)
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

# Numba-optimized function to find the minimum area among all triplets (adapted from Inspiration 2)
@jit(nopython=True, cache=True)
def get_min_area(points: np.ndarray, p_indices: np.ndarray) -> float:
    """
    Calculates the minimum area among all possible triangles formed by triplets of points,
    using precomputed indices for efficiency with Numba.
    Returns 0.0 if any three points are collinear or identical (area is numerically zero).
    """
    min_area = np.inf
    n_combinations = p_indices.shape[0]

    if points.shape[0] < 3: # Handle case with fewer than 3 points (no triangles)
        return 0.0

    # Iterate through all unique combinations of three points using precomputed indices
    for i in range(n_combinations):
        p1 = points[p_indices[i, 0]]
        p2 = points[p_indices[i, 1]]
        p3 = points[p_indices[i, 2]]
                
        area = calculate_triangle_area(p1, p2, p3)
                
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
def _is_point_inside_equilateral_triangle_jit(point: np.ndarray, epsilon: float) -> bool:
    """Checks if a point is within or on the boundary of the unit equilateral triangle."""
    x, y = point
    if y < -epsilon: return False
    if SQRT3 * x - y < -epsilon: return False # Left edge: sqrt(3)*x - y >= 0
    if SQRT3 * (1 - x) - y < -epsilon: return False # Right edge: sqrt(3)*(1-x) - y >= 0
    return True

# Numba-optimized function for post-optimization projection
@jit(nopython=True, cache=True)
def _project_points_to_triangle_boundary_jit(points: np.ndarray, epsilon: float) -> np.ndarray:
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
            x_proj = x - SQRT3 * val / (SQRT3*SQRT3 + (-1)*(-1))
            y_proj = y - (-1) * val / (SQRT3*SQRT3 + (-1)*(-1))
            x, y = x_proj, y_proj
        
        # Clamp to right edge: y = -sqrt(3) * (x - 1) (line: sqrt(3)*x + y - sqrt(3) = 0)
        if SQRT3 * (1 - x) - y < -epsilon:
            val = SQRT3 * x + y - SQRT3
            x_proj = x - SQRT3 * val / (SQRT3*SQRT3 + 1*1)
            y_proj = y - 1 * val / (SQRT3*SQRT3 + 1*1)
            x, y = x_proj, y_proj

        projected_points[i] = [x, y]
    return projected_points


def _objective_function_factory(n: int, p_indices: np.ndarray, penalty_multiplier: float, min_area_threshold: float, epsilon_val: float):
    """
    Creates the JIT-compiled objective function for global optimization (e.g., dual_annealing),
    including penalty for points outside the equilateral triangle.
    """
    @jit(nopython=True, cache=True)
    def objective_fn(flat_points: np.ndarray) -> float:
        points = flat_points.reshape((n, 2))
        
        penalty = calculate_boundary_penalty(points, penalty_multiplier, epsilon_val)
        min_area = get_min_area(points, p_indices)
        
        # Penalize near-zero areas heavily to prevent degenerate solutions.
        if min_area < min_area_threshold:
            penalty += 1e7 * (min_area_threshold - min_area) 

        # Maximize min_area is equivalent to minimizing -min_area
        return -min_area + penalty
    
    return objective_fn

def _local_objective_function_factory(n: int, p_indices: np.ndarray):
    """
    Creates the JIT-compiled objective function for local optimization (e.g., SLSQP).
    It only minimizes -min_area, as boundary constraints are handled separately by SLSQP.
    (Adapted from Inspiration Program 2 for performance)
    """
    @jit(nopython=True, cache=True)
    def local_objective_fn(flat_points: np.ndarray) -> float:
        points = flat_points.reshape((n, 2))
        min_area = get_min_area(points, p_indices)
        # Maximize min_area is equivalent to minimizing -min_area
        return -min_area
    return local_objective_fn

# Superior initialization using Sobol sequence and barycentric mapping.
def _generate_initial_points_in_triangle(n: int, seed: int) -> np.ndarray:
    """
    Generates n quasi-random points inside the unit equilateral triangle
    using a Sobol sequence and barycentric coordinate mapping.
    """
    np.random.seed(seed) # For reproducibility of any internal random calls in qmc.Sobol
    
    sampler = qmc.Sobol(d=2, scramble=True, seed=seed)
    # Generate n 2D points in the unit square [0,1)x[0,1)
    rand_pairs = sampler.random(n=n)

    # Map points from unit square to a uniform distribution within the triangle
    sqrt_r1 = np.sqrt(rand_pairs[:, 0])
    u = 1 - sqrt_r1
    v = rand_pairs[:, 1] * sqrt_r1
    
    # Convert from barycentric to Cartesian coordinates.
    points_cartesian = (v[:, np.newaxis] * TRIANGLE_VERTICES[1, :] +
                       (1 - u - v)[:, np.newaxis] * TRIANGLE_VERTICES[2, :])
    
    return points_cartesian


def heilbronn_triangle11() -> np.ndarray:
    """
    Constructs an optimal arrangement of 11 points in an equilateral triangle
    to maximize the minimum triangle area, combining strategies from top-performing solutions.
    """
    n = _N_POINTS # Use global constant
    np.random.seed(42) # Ensure reproducibility for any stochastic parts

    # Define rectangular bounds for the optimizer.
    bounds = [(0.0, 1.0), (0.0, SQRT3 / 2.0)] * n

    # --- Phase 1: Global Optimization with Dual Annealing ---

    # 1. Generate a high-quality initial guess using Sobol sequence + barycentric mapping.
    initial_points = _generate_initial_points_in_triangle(n, seed=42)
    x0 = initial_points.flatten()

    # 2. Configure dual_annealing's internal local searcher to Nelder-Mead.
    minimizer_kwargs_global = {
        "method": "Nelder-Mead",
        "options": {"xatol": 1e-8, "fatol": 1e-8, "adaptive": True}
    }

    # 3. Create the global objective function using the factory (JIT compiled).
    # Increased penalty_multiplier to 1e7 for stronger constraint enforcement (inspired by Program 2).
    global_objective_fn = _objective_function_factory(n, _P_INDICES, penalty_multiplier=1e7, min_area_threshold=1e-8, epsilon_val=EPSILON)

    # 4. Run the global optimizer. Parameters tuned based on inspirations.
    result_global = dual_annealing(
        global_objective_fn, # Use the factory-generated objective
        bounds,
        x0=x0,
        maxiter=15000, # High iteration count for thorough global search
        initial_temp=5230.0,
        seed=42,
        minimizer_kwargs=minimizer_kwargs_global,
    )
    x0_local = result_global.x

    # --- Phase 2: Local Refinement with SLSQP ---

    # 1. Create JIT-compiled local objective function using the new factory for performance.
    local_objective_fn = _local_objective_function_factory(n, _P_INDICES)
    
    # Define linear inequality constraints for the equilateral triangle
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
        
    triangle_constraints = LinearConstraint(A, lb, ub)

    # 2. Run the local optimizer from the best point found by the global search.
    # Tightened ftol for higher precision, which is feasible due to JIT-compiled objective.
    result_local = minimize(
        local_objective_fn,
        x0_local,
        method='SLSQP',
        bounds=bounds,
        constraints=triangle_constraints,
        options={'maxiter': 20000, 'ftol': 1e-12, 'gtol': 1e-8} # Tightened ftol, relaxed gtol
    )
    
    optimized_points = result_local.x.reshape((n, 2))

    # --- Phase 3: Final Projection ---
    final_points = _project_points_to_triangle_boundary_jit(optimized_points, epsilon=EPSILON)

    # Final validation check
    for p in final_points:
        if not _is_point_inside_equilateral_triangle_jit(p, epsilon=EPSILON):
            print(f"Warning: Point {p} is outside the triangle after final projection.")

    return final_points

# EVOLVE-BLOCK-END
