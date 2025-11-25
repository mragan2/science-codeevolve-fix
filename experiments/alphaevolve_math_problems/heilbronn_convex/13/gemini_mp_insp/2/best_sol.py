# EVOLVE-BLOCK-START
import numpy as np
import itertools
from scipy.spatial import ConvexHull
from scipy.optimize import differential_evolution, minimize
from scipy.stats import qmc
from numba import njit

# Fixed number of points for this problem
_N_POINTS = 13
# Constants for 3-fold symmetry rotation
_COS_120 = -0.5
_SIN_120 = np.sqrt(3) / 2.0
_ROT_MATRIX_120 = np.array([[_COS_120, -_SIN_120], [_SIN_120, _COS_120]], dtype=np.float64) # Explicit dtype for Numba
_ROT_MATRIX_240 = np.array([[_COS_120, _SIN_120], [-_SIN_120, _COS_120]], dtype=np.float64) # Explicit dtype for Numba
# Note: _TRIANGLE_INDICES global variable was removed as it was defined but not used
# in the Numba-jitted triangle area calculation.

@njit(fastmath=True, cache=True)
def _calculate_min_triangle_area_numba(points: np.ndarray) -> float:
    """
    Calculates the minimum triangle area among all combinations of 3 points using Numba.
    """
    n = points.shape[0]
    if n < 3:
        return 0.0
    
    min_triangle_area = np.inf
    
    for i in range(n):
        for j in range(i + 1, n):
            for k in range(j + 1, n):
                p1 = points[i]
                p2 = points[j]
                p3 = points[k]
                area = 0.5 * np.abs(p1[0]*(p2[1]-p3[1]) + p2[0]*(p3[1]-p1[1]) + p3[0]*(p1[1]-p2[1]))
                # Add a small epsilon to area to prevent exact zero areas and smooth the objective slightly.
                area += 1e-12 
                if area < min_triangle_area:
                    min_triangle_area = area
    return min_triangle_area

@njit(cache=True, fastmath=True) # JIT-compiled for performance
def _generate_symmetric_points_jit(params: np.ndarray) -> np.ndarray:
    """
    Generates 13 points with 3-fold rotational symmetry around a FIXED center point (0.5, 0.5).
    This reduces the search space to 8 dimensions, a highly effective heuristic for N=13.
    params: 1D array of (base1_x, base1_y, ..., base4_x, base4_y).
    """
    num_rotational_base_points = 4
    
    points = np.zeros((_N_POINTS, 2), dtype=np.float64)
    center_point = np.array([0.5, 0.5], dtype=np.float64) # Center is FIXED
    points[0] = center_point

    base_points = np.zeros((num_rotational_base_points, 2), dtype=np.float64)
    for i in range(num_rotational_base_points):
        base_points[i, 0] = params[i*2]
        base_points[i, 1] = params[i*2 + 1]
    
    points[1:5] = base_points

    # Manual loop for rotations to ensure Numba compatibility and efficiency
    # This is faster than matrix multiplication inside Numba for small matrices.
    cx, cy = 0.5, 0.5
    for i in range(num_rotational_base_points):
        px, py = base_points[i, 0], base_points[i, 1]

        # Translate to origin relative to center
        translated_x = px - cx
        translated_y = py - cy

        # Rotate by 120 degrees
        rotated_120_x = translated_x * _ROT_MATRIX_120[0,0] + translated_y * _ROT_MATRIX_120[0,1]
        rotated_120_y = translated_x * _ROT_MATRIX_120[1,0] + translated_y * _ROT_MATRIX_120[1,1]
        points[5+i, 0] = rotated_120_x + cx
        points[5+i, 1] = rotated_120_y + cy

        # Rotate by 240 degrees
        rotated_240_x = translated_x * _ROT_MATRIX_240[0,0] + translated_y * _ROT_MATRIX_240[0,1]
        rotated_240_y = translated_x * _ROT_MATRIX_240[1,0] + translated_y * _ROT_MATRIX_240[1,1]
        points[9+i, 0] = rotated_240_x + cx
        points[9+i, 1] = rotated_240_y + cy
    
    return points

def _calculate_convex_hull_area(points: np.ndarray) -> float:
    """
    Calculates the area of the convex hull of the given points.
    Handles potential Qhull errors for degenerate point sets using 'QJ' option.
    """
    if points.shape[0] < 3:
        return 0.0
    try:
        # 'QJ' option for ConvexHull helps to jog points slightly to avoid
        # numerical precision issues with collinear or coplanar points.
        return ConvexHull(points, qhull_options='QJ').volume
    except Exception:
        return 0.0

def _objective_function_symmetric(flat_symmetric_params: np.ndarray) -> float:
    """
    Objective function using the true (non-smooth) minimum triangle area.
    Includes robust quadratic penalties for out-of-bounds points and strong
    proportional penalties for degenerate hull/triangle areas.
    """
    points = _generate_symmetric_points_jit(flat_symmetric_params) # Use JIT-compiled function

    # Robust quadratic penalty for points generated outside the unit square [0,1]x[0,1]
    penalty_scale = 1e9 # Strong quadratic penalty for strict adherence to bounds
    out_of_bounds_x = np.maximum(0.0, points[:, 0] - 1.0) + np.maximum(0.0, -points[:, 0])
    out_of_bounds_y = np.maximum(0.0, points[:, 1] - 1.0) + np.maximum(0.0, -points[:, 1])
    out_of_bounds_penalty = penalty_scale * (np.sum(out_of_bounds_x**2) + np.sum(out_of_bounds_y**2))
    
    # Early exit if points are severely out of bounds to save computation
    if out_of_bounds_penalty > 0:
        return out_of_bounds_penalty

    min_tri_area = _calculate_min_triangle_area_numba(points)
    hull_area = _calculate_convex_hull_area(points)

    # Proportional penalties for degenerate configurations
    degeneracy_penalty = 0.0
    
    if hull_area < 1e-8: # Threshold for degenerate hull
        degeneracy_penalty += 1e10 + (1e-8 - hull_area) * 1e12 # Strong, proportional penalty
    if min_tri_area < 1e-12: # Threshold for degenerate triangle
        degeneracy_penalty += 1e10 + (1e-12 - min_tri_area) * 1e12 # Strong, proportional penalty

    if degeneracy_penalty > 0:
        return degeneracy_penalty

    # Add a small epsilon to both numerator and denominator for maximum numerical stability.
    normalized_area = (min_tri_area + 1e-18) / (hull_area + 1e-18)
    
    # The goal is to maximize normalized_area, so we return its negative for minimization.
    return -normalized_area

def heilbronn_convex13() -> np.ndarray:
    """
    Constructs an arrangement of 13 points to maximize the minimum triangle area
    by adopting the superior fixed-center 8D search strategy and reallocating
    computational budget to an extremely aggressive ensemble optimization protocol.
    """
    num_symmetric_params = 8  # 4 base points (8), center is FIXED at [0.5, 0.5]
    bounds_symmetric = [(0.0, 1.0)] * num_symmetric_params
    base_seed = 42

    # --- Step 1: Hyper-Aggressive Ensemble Global Search ---
    # --- Step 1: Hyper-Aggressive Ensemble Global Search ---
    # With the simplified 8D search space, we can afford a much deeper global search.
    K_runs = 8 # Number of independent differential_evolution runs
    maxiter_de_per_run = 18000 # Max iterations per DE run
    popsize_de_per_run = 270 # Slightly increased population size for more exploration (was 250)

    best_objective_value = np.inf # We are minimizing, so start with infinity
    best_optimal_symmetric_params = None

    for run_idx in range(K_runs):
        sampler = qmc.Sobol(d=num_symmetric_params, scramble=True, seed=base_seed + run_idx)
        # Generate initial population, ensuring it's within bounds.
        initial_population_sobol = sampler.random(n=popsize_de_per_run)

        result_de = differential_evolution(
            func=_objective_function_symmetric,
            bounds=bounds_symmetric,
            init=initial_population_sobol,
            strategy='best1bin',
            maxiter=maxiter_de_per_run,
            popsize=popsize_de_per_run, # Use the increased popsize
            recombination=0.9,
            tol=1e-10, # Extremely tight tolerance
            seed=base_seed + run_idx,
            disp=False,
            workers=-1, # Utilize all available CPU cores for parallelization
            polish=True # Perform a local optimization step at the end of DE
        )

        if result_de.success and result_de.fun < best_objective_value:
            best_objective_value = result_de.fun
            best_optimal_symmetric_params = result_de.x
        # Suppress warnings from DE runs that don't converge, as it's common in ensemble approaches
        # and we only care about the best overall result.

    if best_optimal_symmetric_params is None:
        raise RuntimeError("Ensemble Differential Evolution failed to find a valid solution.")

    # --- Step 2: Intensified Multi-Start Local Refinement ---
    # --- Step 2: Intensified Multi-Start Local Refinement ---
    # Perform a deep, greedy local search from the best global point found.
    # The first local search starts directly from DE's best, subsequent ones perturb the current best.
    num_local_restarts = 20 # Increased local restarts for even deeper search
    rng = np.random.default_rng(seed=base_seed + K_runs)

    for i in range(num_local_restarts):
        if i == 0:
            # Start the first local refinement directly from the best DE result (no perturbation)
            x0 = best_optimal_symmetric_params
        else:
            # Perturb the *current best overall solution* for subsequent restarts
            # Use a small perturbation range to fine-tune around the best known point
            perturbation = rng.uniform(-0.002, 0.002, size=best_optimal_symmetric_params.shape)
            x0 = np.clip(best_optimal_symmetric_params + perturbation, 0.0, 1.0)
        
        result_local = minimize(
            fun=_objective_function_symmetric,
            x0=x0,
            bounds=bounds_symmetric,
            method='L-BFGS-B',
            options={'maxiter': 5000, 'ftol': 1e-14, 'gtol': 1e-12, 'disp': False}, # Increased maxiter, even tighter tolerances
            tol=1e-13 # Overall tolerance for minimization
        )
        
        if result_local.success and result_local.fun < best_objective_value:
            best_objective_value = result_local.fun
            best_optimal_symmetric_params = result_local.x

    # --- Step 3: Final Point Generation and Normalization ---
    optimal_points = _generate_symmetric_points_jit(best_optimal_symmetric_params)

    # Final normalization to ensure the convex hull has an area of exactly 1.
    final_hull_area = _calculate_convex_hull_area(optimal_points)
    if final_hull_area > 1e-8: # Prevent division by near-zero area
        scale_factor = 1.0 / np.sqrt(final_hull_area)
        # Scale points around their geometric centroid to preserve the configuration's shape.
        centroid = np.mean(optimal_points, axis=0)
        optimal_points = (optimal_points - centroid) * scale_factor + centroid
    else:
        print("Warning: Final convex hull area is degenerate. Cannot perform scaling.")

    return optimal_points

# EVOLVE-BLOCK-END