# EVOLVE-BLOCK-START
import numpy as np
import itertools
from scipy.optimize import differential_evolution, dual_annealing
import sobol_seq # Used for low-discrepancy initial population
import random # For setting seed in sobol_seq
from functools import partial
from numba import jit # For JIT compilation of critical functions

# --- Constants and Configuration ---
_N_POINTS = 11
_SQRT3 = np.sqrt(3.0) # Ensure float literal for Numba
_UNIT_TRIANGLE_AREA = _SQRT3 / 4.0
_SEED = 42 # Fixed random seed for reproducibility
_EPSILON = 1e-12 # Threshold for detecting degenerate triangles, passed to JIT functions.

# Vertices of the equilateral triangle
_V1, _V2, _V3 = np.array([[0.0, 0.0], [1.0, 0.0], [0.5, _SQRT3 / 2.0]])

# Pre-compute combination indices for vectorized area calculation.
_COMBINATION_INDICES = np.array(list(itertools.combinations(range(_N_POINTS), 3)), dtype=np.intp)
_P1_INDICES = _COMBINATION_INDICES[:, 0]
_P2_INDICES = _COMBINATION_INDICES[:, 1]
_P3_INDICES = _COMBINATION_INDICES[:, 2]

# --- Helper Functions for Optimization ---

@jit(nopython=True, cache=True)
def _barycentric_to_cartesian_jit(r_coords: np.ndarray, V1: np.ndarray, V2: np.ndarray, V3: np.ndarray) -> np.ndarray:
    """
    JIT-compiled conversion from (r1, r2) barycentric-like coordinates to Cartesian (x, y) coordinates.
    This ensures points are within the triangle.
    Input r_coords is of shape (N, 2), outputs (N, 2).
    """
    n = r_coords.shape[0]
    points = np.empty((n, 2), dtype=r_coords.dtype)

    for i in range(n):
        r1 = r_coords[i, 0]
        r2 = r_coords[i, 1]
        
        # This specific transformation (from Inspiration 2/3) maps [0,1]^2 to the triangle.
        # P = (1 - sqrt(r1)) * V1 + (sqrt(r1) * (1 - r2)) * V2 + (sqrt(r1) * r2) * V3
        sqrt_r1 = np.sqrt(r1)
        
        points[i, 0] = (1.0 - sqrt_r1) * V1[0] + (sqrt_r1 * (1.0 - r2)) * V2[0] + (sqrt_r1 * r2) * V3[0]
        points[i, 1] = (1.0 - sqrt_r1) * V1[1] + (sqrt_r1 * (1.0 - r2)) * V2[1] + (sqrt_r1 * r2) * V3[1]
    
    return points


@jit(nopython=True, cache=True)
def _vectorized_min_area_jit(points: np.ndarray, p1_idx: np.ndarray, p2_idx: np.ndarray, p3_idx: np.ndarray, min_area_threshold: float) -> float:
    """
    JIT-compiled and vectorized calculation of the minimum area of all triangles.
    Returns 0.0 if any triangle is degenerate (area < min_area_threshold).
    """
    # Use pre-computed indices to select points for all C(n,3) triangles
    p1s = points[p1_idx]
    p2s = points[p2_idx]
    p3s = points[p3_idx]

    # Vectorized area calculation using the Shoelace formula
    areas = 0.5 * np.abs(
        p1s[:, 0] * (p2s[:, 1] - p3s[:, 1]) +
        p2s[:, 0] * (p3s[:, 1] - p1s[:, 1]) +
        p3s[:, 0] * (p1s[:, 1] - p2s[:, 1])
    )
    
    # Fast check for degenerate triangles. If any exist, return 0 for a max penalty.
    if np.any(areas < min_area_threshold):
        return 0.0

    return np.min(areas)

def _objective_function(r_coords_flat: np.ndarray, n: int, V1: np.ndarray, V2: np.ndarray, V3: np.ndarray, 
                        unit_triangle_area: float, p1_idx: np.ndarray, p2_idx: np.ndarray, p3_idx: np.ndarray,
                        min_area_threshold: float) -> float: # Added min_area_threshold parameter
    """
    Objective function for the Heilbronn problem using barycentric coordinates.
    It returns the negative of the normalized minimum triangle area for minimization.
    """
    # 1. Reshape parameters and convert to Cartesian coordinates
    r_coords = r_coords_flat.reshape((n, 2))
    points = _barycentric_to_cartesian_jit(r_coords, V1, V2, V3)

    # 2. Calculate the minimum area among all triplets using JIT-compiled function
    # Use a small threshold to detect degenerate triangles and penalize them.
    min_area = _vectorized_min_area_jit(points, p1_idx, p2_idx, p3_idx, min_area_threshold)

    # If min_area is 0, it means a degenerate triangle was found due to the threshold. Penalize.
    if min_area < min_area_threshold: # Re-check against the passed threshold
        return 1e6 # Return a very large positive value for a minimizer

    # 3. Normalize the area and return its negative (for minimization)
    min_area_normalized = min_area / unit_triangle_area
    return -min_area_normalized

def heilbronn_triangle11() -> np.ndarray:
    """
    Constructs an arrangement of 11 points within an equilateral triangle to maximize the
    minimum area of any triangle formed by three of these points.
    This implementation uses a hybrid optimization strategy: Differential Evolution for
    global exploration, followed by Dual Annealing for local refinement, leveraging
    barycentric coordinates to manage domain constraints and Numba for performance.
    
    Returns:
        np.ndarray: An array of shape (11, 2) containing the coordinates of the 11 points.
    """
    # Set seeds for reproducibility
    np.random.seed(_SEED)
    random.seed(_SEED)

    # Optimization variables are (r1, r2) pairs for barycentric coordinates, bounded in [0, 1].
    # This inherently enforces the triangular domain constraint, simplifying the objective.
    dimensions = 2 * _N_POINTS
    bounds = [(0.0, 1.0)] * dimensions

    # Create a partial function for the objective, passing all constants needed.
    obj_func_partial = partial(_objective_function,
                               n=_N_POINTS,
                               V1=_V1, V2=_V2, V3=_V3,
                               unit_triangle_area=_UNIT_TRIANGLE_AREA,
                               p1_idx=_P1_INDICES, p2_idx=_P2_INDICES, p3_idx=_P3_INDICES,
                               min_area_threshold=_EPSILON) # Pass _EPSILON to partial function

    try:
        # --- Hybrid Optimization Strategy (Differential Evolution + Dual Annealing) ---

        # Stage 1: Differential Evolution for broad global exploration.
        # Use Sobol sequence for initial population to ensure good coverage.
        popsize_de = 75 # Increased popsize for better exploration
        initial_pop_de = sobol_seq.i4_sobol_generate(dimensions, popsize_de, skip=1000)
        
        result_de = differential_evolution(
            obj_func_partial,
            bounds,
            strategy='best1bin',
            maxiter=7500, # Increased maxiter for DE
            popsize=popsize_de,
            tol=1e-6, # Tighter tolerance
            atol=1e-6, # Tighter tolerance
            mutation=(0.5, 1.0),
            recombination=0.9,
            seed=_SEED,
            polish=False, # No need to polish here; DA will do the fine-tuning.
            workers=-1, # Use all available CPU cores.
            init=initial_pop_de
        )
        
        # The best solution from DE becomes the initial guess for Dual Annealing.
        initial_guess_for_da = result_de.x

        # Stage 2: Dual Annealing for deep, focused search starting from the DE result.
        result_da = dual_annealing(
            obj_func_partial,
            bounds,
            seed=_SEED,
            maxiter=30000, # Increased maxiter for DA
            initial_temp=5230.0, # Adjusted initial_temp for better exploration
            visit=2.62, # Adjusted visit parameter
            accept=-5.0E-6, # Adjusted accept parameter
            x0=initial_guess_for_da # Use the high-quality seed from DE.
        )
        
        if not result_da.success:
            # print(f"Warning: Optimization may not have converged: {result_da.message}")
            pass

        best_r_coords_flat = result_da.x

    except Exception as e:
        # print(f"An error occurred during optimization: {e}. Falling back to random initialization.")
        # Fallback: if optimization fails, use a random configuration.
        best_r_coords_flat = np.random.rand(dimensions)

    # Convert the final optimal (r1, r2) coordinates back to Cartesian.
    best_r_coords = best_r_coords_flat.reshape((_N_POINTS, 2))
    final_points = _barycentric_to_cartesian_jit(best_r_coords, _V1, _V2, _V3)
    
    return final_points

# EVOLVE-BLOCK-END
