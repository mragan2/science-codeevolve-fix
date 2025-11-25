# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import dual_annealing
from itertools import combinations
from scipy.stats import qmc # New import for Sobol sequence
from scipy.optimize import minimize # Added import for local refinement
from numba import jit # New import for Numba

# --- Constants and Configuration ---
# These are defined at the module level for clarity and efficiency.
_N_POINTS = 11
_SQRT3 = np.sqrt(3.0) # Ensure float literal for Numba
_VERTICES = np.array([[0.0, 0.0], [1.0, 0.0], [0.5, _SQRT3 / 2]], dtype=np.float64) # Ensure float literal for Numba
_SEED = 42  # Seed for random number generation to ensure reproducibility.
_EPSILON = 1e-9 # Epsilon for floating point comparisons and degeneracy checks
_MIN_AREA_THRESHOLD = 1e-10 # Threshold for considering areas as degenerate
_BOUNDING_TRIANGLE_AREA = _SQRT3 / 4.0 # Area of the bounding equilateral triangle, for normalization

# Pre-compute combinations for vectorized area calculation. dtype=np.intp is for indexing.
_COMBINATIONS = np.array(list(combinations(range(_N_POINTS), 3)), dtype=np.intp)

# --- Helper Functions for Optimization ---

@jit(nopython=True, cache=True)
def _is_point_inside_triangle_jit(point: np.ndarray) -> bool:
    """
    Checks if a single point (x, y) is inside or on the boundary of the equilateral triangle
    with vertices (0,0), (1,0), and (0.5, sqrt(3)/2). JIT-compiled for speed.
    """
    x, y = point[0], point[1]
    
    # Check against the three lines forming the triangle boundaries.
    # Using _EPSILON for tolerance.
    # Line 1: y = 0
    if y < -_EPSILON: return False
    # Line 2: y = sqrt(3) * x  <=> sqrt(3) * x - y = 0
    if _SQRT3 * x - y < -_EPSILON: return False
    # Line 3: y = -sqrt(3) * (x - 1) <=> sqrt(3) * (1 - x) - y = 0
    if _SQRT3 * (1.0 - x) - y < -_EPSILON: return False
        
    return True

@jit(nopython=True, cache=True)
def _calculate_min_area_jit(points: np.ndarray, combinations_indices: np.ndarray) -> float:
    """
    Calculates the minimum area of a triangle formed by any three points in the set.
    This version is vectorized using NumPy and JIT-compiled for significant performance improvement.
    """
    if points.shape[0] < 3:
        return 0.0

    # Select points for all C(n,3) triangles at once using the pre-computed combinations.
    # Numba can handle this advanced indexing with `combinations_indices`.
    p1s = points[combinations_indices[:, 0]]
    p2s = points[combinations_indices[:, 1]]
    p3s = points[combinations_indices[:, 2]]

    # Vectorized Shoelace formula for area calculation
    # Area = 0.5 * |(x1(y2-y3) + x2(y3-y1) + x3(y1-y2))|
    areas = 0.5 * np.abs(p1s[:, 0] * (p2s[:, 1] - p3s[:, 1]) +
                         p2s[:, 0] * (p3s[:, 1] - p1s[:, 1]) +
                         p3s[:, 0] * (p1s[:, 1] - p2s[:, 1]))

    return np.min(areas) if areas.size > 0 else 0.0 # Handle case of no combinations (shouldn't happen for N_POINTS >= 3)


def _objective_function(x_flat: np.ndarray) -> float:
    """
    The objective function for the optimizer. It returns the negative of the
    minimum triangle area. A large penalty is applied if any point is outside
    the valid triangular region.
    """
    points = x_flat.reshape((_N_POINTS, 2))

    # Apply a large penalty if any point is outside the triangle.
    # This guides the optimizer to search within the valid domain.
    for i in range(_N_POINTS):
        if not _is_point_inside_triangle_jit(points[i]):
            return 1e10 # Increased penalty to strongly disincentivize invalid positions

    min_area = _calculate_min_area_jit(points, _COMBINATIONS)

    # Apply a continuous penalty if the minimum area is too small (degenerate).
    # This prevents the optimizer from collapsing points, which would yield a
    # minimum area close to zero. The problem asks to maximize the *minimum* area,
    # so a very small positive area is fundamentally a poor solution.
    degenerate_penalty = 0.0
    if min_area < _MIN_AREA_THRESHOLD:
        # Use a strong, continuous penalty that increases as min_area goes below the threshold.
        degenerate_penalty = (_MIN_AREA_THRESHOLD - min_area) * 1e12 # Strong continuous penalty

    # dual_annealing performs minimization, so we return the negative of the area
    # to achieve maximization of the minimum area, plus any continuous penalties.
    return -min_area + degenerate_penalty

def heilbronn_triangle11() -> np.ndarray:
    """
    Constructs an arrangement of 11 points within an equilateral triangle
    to maximize the minimum area of any triangle formed by three of these points.
    This implementation uses the dual_annealing optimization algorithm from SciPy,
    a stochastic global optimization method suitable for non-convex problems.
    
    Returns:
        np.ndarray: An array of shape (11, 2) containing the coordinates of the 11 points.
    """
    # Define the search space bounds (a bounding box for the triangle).
    # The triangular constraint is handled by the objective function's penalty.
    bounds = [(0, 1), (0, _SQRT3 / 2)] * _N_POINTS

    # Generate initial guess using a Sobol sequence for better space-filling properties
    # compared to pseudo-random numbers.
    sampler = qmc.Sobol(d=2, seed=_SEED)
    sobol_points = sampler.random(_N_POINTS) # Generates points in [0,1]^2

    # Transform Sobol points to be uniformly distributed within an equilateral triangle
    # using barycentric coordinates, similar to the original random generation.
    u = sobol_points[:, 0]
    v = sobol_points[:, 1]
    
    needs_flip = u + v > 1
    u[needs_flip] = 1 - u[needs_flip]
    v[needs_flip] = 1 - v[needs_flip]
    
    w = 1 - u - v
    
    # P = u*A + v*B + w*C, where A, B, C are the triangle vertices.
    initial_guess = (u[:, np.newaxis] * _VERTICES[0] +
                     v[:, np.newaxis] * _VERTICES[1] +
                     w[:, np.newaxis] * _VERTICES[2])

    # A single, intensive global optimization run is more effective than multi-start.
    # This focuses computational effort on deeply exploring the search space from a
    # high-quality Sobol starting point.
    result = dual_annealing(
        func=_objective_function,
        bounds=bounds,
        seed=_SEED,
        maxiter=75000, # Further increased iterations for even deeper global search, aiming for higher quality.
        initial_temp=1e4,
        minimizer_kwargs={"method": "Nelder-Mead"},
        x0=initial_guess.flatten()
    )

    # A final local refinement step to polish the solution found by dual_annealing.
    # This can often find a slightly better minimum in the local basin.
    result_local = minimize(
        fun=_objective_function,
        x0=result.x,
        method='Nelder-Mead',
        bounds=bounds,
        options={'maxiter': 5000, 'xatol': 1e-13, 'fatol': 1e-13} # Increased maxiter for local refinement for higher precision.
    )

    # It's possible for the local search to slightly worsen the solution if it gets
    # trapped or terminates prematurely. We explicitly choose the best result.
    if result_local.fun < result.fun:
        best_x = result_local.x
    else:
        best_x = result.x
    
    optimal_points = best_x.reshape((_N_POINTS, 2))

    return optimal_points

# EVOLVE-BLOCK-END
