# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import dual_annealing, minimize
from itertools import combinations
from scipy.stats import qmc
import numba # New import for Numba JIT compilation

# --- Constants and Configuration ---
# These are defined at the module level for clarity and efficiency.
_N_POINTS = 11
_SQRT3 = np.sqrt(3)
_VERTICES = np.array([[0, 0], [1, 0], [0.5, _SQRT3 / 2]])
_SEED = 42  # Seed for random number generation to ensure reproducibility.
# Pre-compute combinations for JIT-accelerated area calculation.
_COMBINATIONS_INDICES = np.array(list(combinations(range(_N_POINTS), 3)), dtype=np.intp)

# --- Helper Functions for Optimization ---

# This function is retained as-is for consistency with any external validation,
# but the optimization's penalty logic is handled directly in _objective_function_jit.
def _is_point_inside_triangle(points: np.ndarray) -> np.ndarray:
    """
    Checks if points are inside the predefined equilateral triangle using line equations.
    A small tolerance is added to handle floating-point inaccuracies on the boundaries.
    """
    x, y = points[:, 0], points[:, 1]
    # The three line equations for the triangle boundaries:
    # 1. y >= 0
    # 2. y <= sqrt(3) * x
    # 3. y <= -sqrt(3) * (x - 1)
    c1 = y >= -1e-9
    c2 = y <= _SQRT3 * x + 1e-9
    c3 = y <= -_SQRT3 * (x - 1) + 1e-9
    return np.all(np.vstack([c1, c2, c3]), axis=0)

@numba.jit(nopython=True, fastmath=True, cache=True)
def _calculate_min_area_jit(points: np.ndarray) -> float:
    """
    Calculates the minimum area of a triangle formed by any three points in the set.
    JIT-compiled for performance using pre-computed combinations.
    """
    if points.shape[0] < 3:
        return 0.0

    min_area = np.inf # Use np.inf for Numba compatibility
    num_combinations = _COMBINATIONS_INDICES.shape[0]

    for i in range(num_combinations):
        idx1, idx2, idx3 = _COMBINATIONS_INDICES[i, 0], _COMBINATIONS_INDICES[i, 1], _COMBINATIONS_INDICES[i, 2]
        p1, p2, p3 = points[idx1], points[idx2], points[idx3]
        
        # Area formula using the determinant (Shoelace formula)
        area = 0.5 * np.abs(p1[0] * (p2[1] - p3[1]) +
                            p2[0] * (p3[1] - p1[1]) +
                            p3[0] * (p1[1] - p2[1]))
        
        if area < min_area:
            min_area = area
            
    return min_area

@numba.jit(nopython=True, fastmath=True, cache=True)
def _objective_function_jit(x_flat: np.ndarray) -> float:
    """
    The objective function for the optimizer, JIT-compiled.
    It returns the negative of the minimum triangle area,
    and includes a proportional penalty for points outside the valid triangular region
    and for near-zero area triangles.
    """
    points = x_flat.reshape((_N_POINTS, 2))
    
    penalty = 0.0
    penalty_coeff = 2000.0 # A strong coefficient to discourage violations.

    # Proportional penalty for points violating triangle boundaries.
    # This replaces the discontinuous 'return 1e6' from the previous version.
    for i in range(_N_POINTS):
        p = points[i]
        x, y = p[0], p[1]
        
        # Constraint 1: y >= 0
        if y < 0:
            penalty += penalty_coeff * (-y)
        
        # Constraint 2: y <= _SQRT3 * x  (or _SQRT3 * x - y >= 0)
        violation2 = _SQRT3 * x - y
        if violation2 < 0:
            penalty += penalty_coeff * (-violation2)
            
        # Constraint 3: y <= -_SQRT3 * (x - 1)
        # The violation is y - (-_SQRT3 * (x - 1)) = y + _SQRT3 * (x - 1)
        violation3 = y + _SQRT3 * (x - 1)
        if violation3 > 0:
            penalty += penalty_coeff * violation3
            
    min_area = _calculate_min_area_jit(points)

    # Penalize degenerate configurations (collinear/duplicate points).
    # This is crucial for Heilbronn, where non-zero minimum area is the goal.
    # Adopted a stronger penalty from high-performing inspiration programs.
    if min_area < 1e-10: 
        penalty += 10000.0 # Significantly increased penalty for degeneracy.

    # dual_annealing performs minimization, so we return the negative of the area
    # to achieve maximization of the minimum area, plus any penalties.
    return -min_area + penalty

def heilbronn_triangle11() -> np.ndarray:
    """
    Constructs an arrangement of 11 points within an equilateral triangle
    to maximize the minimum area. This uses a two-stage hybrid optimization strategy
    inspired by high-performing solutions:
    1. A multi-start `dual_annealing` for robust global search.
    2. A `Powell` local search to refine the best solution from the global search.
    
    Returns:
        np.ndarray: An array of shape (11, 2) containing the coordinates of the 11 points.
    """
    bounds = [(0, 1), (0, _SQRT3 / 2)] * _N_POINTS
    
    # --- Stage 1: Global Search using Multi-Start Dual Annealing ---
    # A large computational budget is used to find a high-quality basin of attraction.
    num_runs = 15
    max_iter_per_run_da = 8000

    best_fun_val = np.inf
    best_x_from_da = None

    rng_da_seeds = np.random.default_rng(seed=_SEED)
    
    sobol_sampler = qmc.Sobol(d=2, seed=_SEED)
    all_sobol_points_flat = sobol_sampler.random(_N_POINTS * num_runs) 
    all_initial_guesses_sobol = all_sobol_points_flat.reshape(num_runs, _N_POINTS, 2)

    for run_idx in range(num_runs):
        sobol_points_for_run = all_initial_guesses_sobol[run_idx]
        
        u, v = sobol_points_for_run[:, 0], sobol_points_for_run[:, 1]
        needs_flip = u + v > 1
        u[needs_flip], v[needs_flip] = 1 - u[needs_flip], 1 - v[needs_flip]
        w = 1 - u - v
        initial_guess_flattened = (u[:, np.newaxis] * _VERTICES[0] +
                                   v[:, np.newaxis] * _VERTICES[1] +
                                   w[:, np.newaxis] * _VERTICES[2]).flatten()

        result_da = dual_annealing(
            func=_objective_function_jit,
            bounds=bounds,
            seed=rng_da_seeds.integers(1, 1_000_000_000),
            maxiter=max_iter_per_run_da,
            initial_temp=1e4,
            minimizer_kwargs={"method": "Nelder-Mead", "options": {"maxiter": 500}}, 
            x0=initial_guess_flattened
        )

        if result_da.fun < best_fun_val:
            best_fun_val = result_da.fun
            best_x_from_da = result_da.x

    if best_x_from_da is None:
        raise RuntimeError("Global optimization (dual_annealing) failed to find any valid solution.")

    # --- Stage 2: Local Refinement using Powell's Method ---
    # Use the best result from the global search as the starting point for a fine-tuned
    # local search, a strategy proven effective in inspiration programs.
    refinement_result = minimize(
        fun=_objective_function_jit,
        x0=best_x_from_da,
        method='Powell',
        options={'maxiter': 10000, 'disp': False, 'ftol': 1e-12}
    )

    # The final solution is the one found by the local refinement stage.
    optimal_points = refinement_result.x.reshape((_N_POINTS, 2))

    # Add final clamping to ensure points are strictly within the bounding box,
    # correcting for any minor floating point deviations from the optimizer.
    optimal_points[:, 0] = np.clip(optimal_points[:, 0], 0.0, 1.0)
    optimal_points[:, 1] = np.clip(optimal_points[:, 1], 0.0, _SQRT3 / 2.0)

    return optimal_points

# EVOLVE-BLOCK-END
