# EVOLVE-BLOCK-START
import numpy as np
# Removed itertools as it's no longer needed with the Numba direct min area calculation.
from scipy.spatial import ConvexHull
from scipy.optimize import differential_evolution, minimize
from scipy.stats import qmc # Re-added for explicit Sobol sequence generation and ensemble runs
import scipy.spatial.qhull # For QhullError handling
from numba import njit # Changed from jit to njit for clarity and performance
from numpy.random import default_rng # For local refinement restarts

# Fixed number of points for this problem
_N_POINTS = 13

# Constants for 3-fold symmetry rotation around (0.5, 0.5)
_COS_120 = -0.5
_SIN_120 = np.sqrt(3) / 2.0
_ROT_MATRIX_120 = np.array([
    [_COS_120, -_SIN_120],
    [_SIN_120, _COS_120]
], dtype=np.float64) # Explicit dtype for Numba compatibility (from Inspiration 1)
# Rotation matrix for 240 degrees (which is -120 degrees)
_ROT_MATRIX_240 = np.array([
    [_COS_120, _SIN_120],  # cos(240) = cos(-120) = cos(120) = -0.5
    [-_SIN_120, _COS_120] # sin(240) = sin(-120) = -sin(120) = -sqrt(3)/2
], dtype=np.float64) # Explicit dtype for Numba compatibility (from Inspiration 1)

# JIT-compiled function to directly calculate the minimum triangle area (Adopted from Inspiration 2)
@njit(cache=True, fastmath=True) # Changed from jit(nopython=True) to njit for brevity, added fastmath
def _calculate_min_triangle_area_numba_direct(points: np.ndarray) -> float:
    """
    Calculates the minimum triangle area for a given set of points using Numba.
    This function avoids creating an array of all areas, directly finding the minimum.
    Adds a small epsilon to areas to improve numerical stability for near-collinear points.
    
    Args:
        points (np.ndarray): An (N, 2) array of 2D points.
    Returns:
        float: The minimum area among all C(N,3) triangles.
    """
    n = points.shape[0]
    if n < 3:
        return 0.0 # Cannot form a triangle with less than 3 points

    min_area = np.inf
    for i in range(n):
        for j in range(i + 1, n):
            for k in range(j + 1, n):
                p1, p2, p3 = points[i], points[j], points[k]
                # Shoelace formula for triangle area
                area = 0.5 * np.abs(p1[0]*(p2[1] - p3[1]) + p2[0]*(p3[1] - p1[1]) + p3[0]*(p1[1] - p2[1]))
                # Add a small epsilon for numerical stability (from Inspiration 1), slightly smaller
                current_area = area + 1e-15 
                if current_area < min_area:
                    min_area = current_area
    return min_area

@njit(cache=True, fastmath=True) # Added njit and fastmath
def _generate_symmetric_points(params: np.ndarray) -> np.ndarray:
    """
    Generates 13 points with 3-fold rotational symmetry around a parametric center point.
    params: 1D array of (center_x, center_y, base1_x, base1_y, ..., base4_x, base4_y)
    Total 2 + 4*2 = 10 parameters.
    Ensures Numba compatibility with explicit dtypes.
    """
    num_rotational_base_points = 4
    # No need for explicit error check for params.shape[0] inside njit, assume valid input.

    points = np.zeros((_N_POINTS, 2), dtype=np.float64) # Explicit dtype
    
    # Point 0: Center point, taken from parameters
    center_point = params[0:2] # Numba handles slicing well
    points[0] = center_point

    # Rotational Base points (1 to 4) are from the rest of the params
    base_points_flat = params[2:]
    base_points = base_points_flat.reshape((num_rotational_base_points, 2)) # Numba handles reshape well
    points[1:5] = base_points

    # Rotate base points by 120 and 240 degrees around the (parametric) center
    translated_base = base_points - center_point
    
    # _ROT_MATRIX_120 and _ROT_MATRIX_240 are global constants, already defined as np.float64
    rotated_120 = (translated_base @ _ROT_MATRIX_120) + center_point
    points[5:9] = rotated_120

    rotated_240 = (translated_base @ _ROT_MATRIX_240) + center_point
    points[9:13] = rotated_240
    
    return points

def _calculate_convex_hull_area(points: np.ndarray) -> float:
    """
    Calculates the area of the convex hull of the given points.
    Handles degenerate cases robustly, using 'QJ' option (from Inspiration 1 and 2).
    """
    if points.shape[0] < 3:
        return 0.0
    try:
        # 'QJ' option for ConvexHull helps to jog points slightly to avoid
        # numerical precision issues with collinear or coplanar points.
        hull = ConvexHull(points, qhull_options='QJ') # Added qhull_options='QJ' for robustness
        return hull.volume
    except scipy.spatial.qhull.QhullError: # Handle cases where points might be collinear or degenerate
        return 0.0
    except Exception: # Catch any other unexpected errors, though QhullError should cover most
        return 0.0

def _objective_function_symmetric(flat_symmetric_params: np.ndarray) -> float:
    """
    Objective function for the Heilbronn problem, using a symmetric point generation scheme.
    It returns the negative of the normalized minimum triangle area: -(min_triangle_area / convex_hull_area).
    Includes robust penalties for points outside the unit square and for degenerate configurations.
    """
    points = _generate_symmetric_points(flat_symmetric_params)

    # Robust Penalty for points generated outside the unit square [0,1]x[0,1]
    # Adopt quadratic penalty from Inspiration 2 for smoother gradient
    penalty_scale_oob = 1e10 # Strong quadratic penalty for strict adherence to bounds
    out_of_bounds_x_low = np.maximum(0.0, -points[:, 0])
    out_of_bounds_x_high = np.maximum(0.0, points[:, 0] - 1.0)
    out_of_bounds_y_low = np.maximum(0.0, -points[:, 1])
    out_of_bounds_y_high = np.maximum(0.0, points[:, 1] - 1.0)
    
    # Calculate sum of squared violations
    total_out_of_bounds_squared = np.sum(out_of_bounds_x_low**2) + np.sum(out_of_bounds_x_high**2) + \
                                  np.sum(out_of_bounds_y_low**2) + np.sum(out_of_bounds_y_high**2)
    
    # If any point is significantly out of bounds, return a very large constant penalty
    # plus the proportional penalty for guiding it back.
    if total_out_of_bounds_squared > 1e-8: # Threshold for "significant" out-of-bounds
        return 1e15 + penalty_scale_oob * total_out_of_bounds_squared

    # Calculate minimum triangle area using the Numba-accelerated direct function
    min_tri_area = _calculate_min_triangle_area_numba_direct(points)
    hull_area = _calculate_convex_hull_area(points)

    # Degeneracy penalties (combined constant + proportional from Inspirations 1 & 2)
    degeneracy_penalty = 0.0
    
    # If hull area is extremely small or min triangle area is extremely small, apply strong penalties
    if hull_area < 1e-8: # Threshold for degenerate hull
        # Constant penalty + proportional penalty for how far below threshold (from Insp 2)
        degeneracy_penalty += 1e15 + (1e-8 - hull_area) * 1e12
    if min_tri_area < 1e-12: # Threshold for degenerate triangle (after epsilon addition)
        # Constant penalty + proportional penalty (from Insp 2)
        degeneracy_penalty += 1e15 + (1e-12 - min_tri_area) * 1e12

    if degeneracy_penalty > 0:
        return degeneracy_penalty # Return combined degeneracy penalty if any exists

    normalized_area = min_tri_area / hull_area
    # For minor out-of-bounds violations not caught by the hard penalty, add a small proportional penalty
    # This ensures points stay strictly within bounds even if the quadratic sum is very small.
    minor_oob_sum = np.sum(out_of_bounds_x_low) + np.sum(out_of_bounds_x_high) + \
                    np.sum(out_of_bounds_y_low) + np.sum(out_of_bounds_y_high)
    
    return -normalized_area + minor_oob_sum * 1e3 # Maximize normalized_area, so return negative

def heilbronn_convex13() -> np.ndarray:
    """
    Construct an arrangement of exactly 13 points within a unit-area convex region (unit square)
    to maximize the area of the smallest triangle formed by any three of these points.

    This implementation leverages 3-fold rotational symmetry to reduce the search space
    and combines ensemble global optimization (Differential Evolution) with multi-start local refinement (L-BFGS-B).

    Returns:
        points: np.ndarray of shape (13,2) with the x,y coordinates of the optimal points.
    """
    # We optimize for 1 center point (2 coords) + 4 base points (8 coords). Total 10 parameters.
    num_symmetric_params = 10
    
    # Define the search space bounds for each symmetric parameter: [0, 1] for x and y.
    bounds_symmetric = [(0.0, 1.0)] * num_symmetric_params

    # Use a fixed base seed for reproducibility.
    base_seed = 42
    
    # Ensemble Optimization Parameters (tuned for even better quality, acknowledging potential eval_time increase)
    # Ensemble Optimization Parameters (tuned for a better time-quality trade-off, balancing previous aggressive settings)
    K_runs = 5 # Reduced number of independent differential_evolution runs (from 8 to 5, similar to Insp 1)
    maxiter_per_run = 8000 # Reduced maxiter for global search (from 12000 to 8000, similar to Insp 2's DE maxiter)
    initial_population_size_per_run = 150 # Reduced popsize for better population diversity (from 180 to 150, similar to Insp 2)

    best_objective_value = np.inf # We are minimizing, so start with infinity
    best_optimal_symmetric_params = None

    # --- Step 1: Ensemble Global Search with Differential Evolution ---
    for run_idx in range(K_runs):
        # Generate a distinct Sobol sequence for each run to diversify initialization (from Insp 1 & 2)
        # Using scramble=True for qmc.Sobol can improve coverage.
        sampler = qmc.Sobol(d=num_symmetric_params, seed=base_seed + run_idx, scramble=True)
        initial_population_sobol = sampler.random(n=initial_population_size_per_run)

        result_de = differential_evolution(
            func=_objective_function_symmetric,
            bounds=bounds_symmetric,
            init=initial_population_sobol, # Initialize with a high-quality Sobol sequence
            strategy='best1bin',
            maxiter=maxiter_per_run,
            popsize=initial_population_size_per_run, # Set popsize to match init size for consistency
            tol=1e-10, # Tighter tolerance for more precise search
            recombination=0.9, # Increased recombination for more mixing (from Insp 1 & 2)
            mutation=(0.5, 1.0), # Retain good mutation range
            seed=base_seed + run_idx,
            disp=False, # Suppress intermediate output
            workers=-1, # Utilize all available CPU cores for efficiency
            polish=True # Apply local optimization to the best solution found by DE (from Insp 1 & 2)
        )

        if result_de.success and result_de.fun < best_objective_value:
            best_objective_value = result_de.fun
            best_optimal_symmetric_params = result_de.x
        # Suppress warnings for individual DE runs to keep output clean, as polish can sometimes fail but DE still finds a good point.
        # print(f"DE run {run_idx+1}/{K_runs} completed with fun: {result_de.fun}")


    if best_optimal_symmetric_params is None:
        raise RuntimeError("Ensemble Differential Evolution failed to find a valid solution after all runs.")

    # --- Step 2: Multiple Local Refinements with L-BFGS-B --- (from Inspirations 1 & 2)
    # Use the best result from Differential Evolution as an initial guess for local optimization.
    # Run multiple times from slightly perturbed versions to escape very close local minima.
    
    num_local_restarts = 10 # Reduced local restarts (from 15 to 10, similar to Insp 1)
    rng = default_rng(seed=base_seed + K_runs) # New seed for local restarts to ensure reproducibility

    for i in range(num_local_restarts):
        # Perturb the initial guess slightly and clip to bounds
        # Small perturbation range for a focused local search around the global optimum (from Insp 1 & 2)
        perturbation = rng.uniform(-0.002, 0.002, size=best_optimal_symmetric_params.shape)
        perturbed_x0 = np.clip(best_optimal_symmetric_params + perturbation, 0.0, 1.0)
        
        local_result = minimize(
            fun=_objective_function_symmetric,
            x0=perturbed_x0,
            bounds=bounds_symmetric,
            method='L-BFGS-B', # Robust local minimizer for bounded problems (from Insp 1 & 2)
            options={'maxiter': 2500, 'ftol': 1e-13, 'gtol': 1e-11, 'disp': False} # Reduced maxiter (from 3000 to 2500), kept tight ftol/gtol
        )
        
        if local_result.success and local_result.fun < best_objective_value:
            best_objective_value = local_result.fun
            best_optimal_symmetric_params = local_result.x
        # Suppress warnings for individual local runs, as minor failures are common in local search.
        # else:
        #     print(f"Local refinement run {i+1}/{num_local_restarts} failed or did not improve: {local_result.message}")


    # Generate the final point configuration
    optimal_points = _generate_symmetric_points(best_optimal_symmetric_params)
    
    # Final Step: Scale points to ensure their convex hull has unit area (from target and Insp 1 & 3)
    final_hull_area = _calculate_convex_hull_area(optimal_points)
    if final_hull_area > 1e-6: # Ensure hull area is not degenerate before scaling
        scale_factor = 1.0 / np.sqrt(final_hull_area)
        centroid = np.mean(optimal_points, axis=0)
        optimal_points = (optimal_points - centroid) * scale_factor + centroid
    # Else: If hull area is degenerate, return unscaled points. This should ideally not happen
    # if optimization was successful and penalties are good.
            
    return optimal_points

# EVOLVE-BLOCK-END