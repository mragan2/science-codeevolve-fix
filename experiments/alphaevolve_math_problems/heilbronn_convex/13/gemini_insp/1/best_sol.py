# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import differential_evolution, minimize # Added minimize for local refinement
from scipy.spatial import ConvexHull
from numba import njit # Using njit (alias for jit(nopython=True)) for better clarity
from scipy.stats import qmc # For Sobol sequence initialization
import itertools # Still needed for combinations if not using Numba for all_areas

# Helper function to calculate triangle area - JIT compiled for speed
@njit(cache=True, fastmath=True)
def _triangle_area(p1: np.ndarray, p2: np.ndarray, p3: np.ndarray) -> float:
    """Calculates the area of a triangle given three 2D points using Numba."""
    # Using the determinant formula: 0.5 * |x1(y2 - y3) + x2(y3 - y1) + x3(y1 - y2)|
    return 0.5 * np.abs(p1[0]*(p2[1] - p3[1]) + p2[0]*(p3[1] - p1[1]) + p3[0]*(p1[1] - p2[1]))

# JIT-compiled function to efficiently gather all C(n,3) triangle areas
@njit(cache=True, fastmath=True)
def _calculate_all_triangle_areas_numba(points: np.ndarray, n: int) -> np.ndarray:
    """
    Calculates all C(n,3) triangle areas for a given set of points.
    Returns a 1D array of areas. Optimized with Numba.
    """
    num_triangles = n * (n - 1) * (n - 2) // 6
    all_areas = np.empty(num_triangles, dtype=np.float64)
    
    idx = 0
    for i in range(n):
        for j in range(i + 1, n):
            for k in range(j + 1, n):
                all_areas[idx] = _triangle_area(points[i], points[j], points[k])
                idx += 1
    return all_areas

# JIT-compiled function to construct 13 points from 5 generators using 3-fold symmetry
@njit(cache=True)
def _unpack_symmetric_points(generators_flat: np.ndarray) -> np.ndarray:
    """
    Constructs 13 points from 5 generator points (1 center + 4 peripheral)
    assuming 3-fold rotational symmetry. All generated points are clipped to [0,1].
    The first two elements of generators_flat are (center_x, center_y).
    The next 8 elements are for 4 peripheral generator points (gx, gy).
    """
    points = np.empty((13, 2), dtype=np.float64)
    
    # First two elements are the center (xc, yc)
    # Clip center coordinates to be within [0,1]
    center_x = max(0.0, min(1.0, generators_flat[0]))
    center_y = max(0.0, min(1.0, generators_flat[1]))
    points[0, 0] = center_x
    points[0, 1] = center_y
    
    cos120 = -0.5
    sin120 = 0.8660254037844386 # np.sqrt(3.0) / 2.0
    
    current_idx = 1
    # The next 8 elements (generators_flat[2:] ) are for the 4 peripheral generator points
    for i in range(4): # Loop over the 4 peripheral generator points
        # Extract x, y for the i-th peripheral generator
        # These are already constrained by bounds in DE, but inner clipping adds robustness
        gx = max(0.0, min(1.0, generators_flat[2 + i*2]))
        gy = max(0.0, min(1.0, generators_flat[2 + i*2 + 1]))
        
        points[current_idx, 0], points[current_idx, 1] = gx, gy
        
        rel_x, rel_y = gx - center_x, gy - center_y
        
        # Rotate by 120 degrees and clip
        rot1_x = rel_x * cos120 - rel_y * sin120
        rot1_y = rel_x * sin120 + rel_y * cos120
        points[current_idx + 1, 0] = max(0.0, min(1.0, rot1_x + center_x))
        points[current_idx + 1, 1] = max(0.0, min(1.0, rot1_y + center_y))
        
        # Rotate by 240 degrees and clip
        rot2_x = rot1_x * cos120 - rot1_y * sin120
        rot2_y = rot1_x * sin120 + rot1_y * cos120
        points[current_idx + 2, 0] = max(0.0, min(1.0, rot2_x + center_x))
        points[current_idx + 2, 1] = max(0.0, min(1.0, rot2_y + center_y))

        current_idx += 3
        
    return points

# Objective function using LogSumExp (LSE) approximation and Convex Hull normalization
def _symmetric_objective_function_lse(generators_flat: np.ndarray, n: int, lse_alpha: float) -> float:
    """
    Wrapper objective function that uses the LogSumExp (LSE) approximation of the 
    minimum area, normalized by the convex hull area, for smooth optimization.
    """
    full_points = _unpack_symmetric_points(generators_flat)
    
    # Check for duplicate points (with rounding for numerical stability)
    if len(np.unique(np.round(full_points, 12), axis=0)) < n:
        return 1e10 # Heavy penalty for non-unique points

    all_areas = _calculate_all_triangle_areas_numba(full_points, n)

    min_true_area = np.min(all_areas)
    if min_true_area < 1e-12: # Penalize near-collinear or coincident points
        return 1e10 

    try:
        # ConvexHull requires at least 3 non-collinear points.
        # This check is implicitly handled by the duplicate point check above.
        hull = ConvexHull(full_points)
        hull_area = hull.volume # .volume is alias for .area for 2D
    except Exception: # Handle cases where ConvexHull might fail (e.g., all points collinear or too few points)
        return 1e10

    if hull_area < 1e-12: # If hull area is too small, likely degenerate
        return 1e10

    # LogSumExp approximation for min(all_areas).
    # min_area_approx = min_true_area - (1/alpha) * log(sum(exp(-alpha*(A_i - min_true_area))))
    # This form is more numerically stable than sum(exp(-alpha*A_i)) directly.
    scaled_areas = -lse_alpha * (all_areas - min_true_area)
    log_sum_exp = np.log(np.sum(np.exp(scaled_areas)))
    min_area_approx = min_true_area - log_sum_exp / lse_alpha

    # We want to maximize min_area_normalized, so we minimize its negative
    return -min_area_approx / hull_area

def heilbronn_convex13() -> np.ndarray:
    """
    Generates a high-quality arrangement of 13 points by optimizing a smaller set
    of "generator" points under a 3-fold symmetry constraint. It employs a multi-start
    Differential Evolution approach with a smooth LogSumExp objective, convex hull
    normalization, and Sobol sequence initialization, followed by local L-BFGS-B refinement.
    The center of symmetry is also optimized.
    """
    n = 13
    n_generators_total = 5 # 1 center point (x,y) + 4 peripheral generators (x,y) = 5*2 = 10 dimensions
    
    # Bounds for all generator points (center and 4 peripheral) within the unit square [0,1]x[0,1].
    bounds = [(0, 1)] * (n_generators_total * 2)

    best_min_neg_normalized_area = float('inf')
    best_generators_flat = None

    # LSE alpha values for DE and local refinement
    lse_alpha_de = 50000.0      # Higher alpha for DE, provides good balance
    lse_alpha_refine = 500000.0 # Even higher alpha for local refinement for sharper minimum

    # Differential Evolution parameters
    num_starts = 10             # Increased independent runs for broader global search
    maxiter_per_start = 2500    # Increased iterations per run (feasible with Numba + parallelization)
    popsize_val = 40            # Increased population size for better diversity and exploration
    tol_val = 1e-6              # Tighter tolerance for convergence

    # Use a single master seed for the entire multi-start process for reproducibility
    master_seed = 42

    # print("Phase 1: Starting multi-start Differential Evolution for global search...")
    for i in range(num_starts):
        current_seed = master_seed + i 
        
        # Generate initial population using Sobol sequence for better space coverage.
        sampler = qmc.Sobol(d=n_generators_total * 2, seed=current_seed)
        # The number of points for Sobol must be a power of 2 for random_base2.
        # Generate more than needed and truncate to popsize_val.
        m = int(np.ceil(np.log2(popsize_val)))
        initial_population = sampler.random_base2(m=m)[:popsize_val]
        
        result_de = differential_evolution(
            func=_symmetric_objective_function_lse,
            bounds=bounds,
            args=(n, lse_alpha_de), # Use lse_alpha_de for global search
            strategy='best1bin',
            maxiter=maxiter_per_start,
            popsize=popsize_val,
            tol=tol_val,
            mutation=(0.5, 1.0),
            recombination=0.7,
            seed=current_seed,
            disp=False,
            polish=True, # Polish with L-BFGS-B after DE
            init=initial_population, # Pass the Sobol-initialized population
            workers=-1, # Use all available CPU cores for parallelization
            updating='deferred' # Recommended for parallel execution
        )
        
        if result_de.fun < best_min_neg_normalized_area:
            best_min_neg_normalized_area = result_de.fun
            best_generators_flat = result_de.x
        
        # Optional: Print progress
        # print(f"  DE Run {i+1}/{num_starts} finished. Best obj: {result_de.fun:.6f}")

    if best_generators_flat is None:
        # Fallback if no runs found a valid solution (should not happen with these parameters)
        # print("Warning: No valid configuration found in global DE search. Returning a default symmetric configuration.")
        rng = np.random.default_rng(seed=master_seed)
        best_generators_flat = rng.random(n_generators_total * 2) # Random generators as a last resort

    # print("\nPhase 2: Starting local refinement with L-BFGS-B...")
    # Phase 2: Local Refinement using L-BFGS-B on the best DE result
    # Use a higher lse_alpha for a sharper approximation of the min function.
    local_result = minimize(
        fun=_symmetric_objective_function_lse,
        x0=best_generators_flat,
        args=(n, lse_alpha_refine), # Use lse_alpha_refine for local refinement
        method='L-BFGS-B',
        bounds=bounds,
        options={'maxiter': 2000, 'ftol': 1e-12, 'gtol': 1e-8, 'disp': False} # Tighter tolerances
    )

    # if not local_result.success:
    #     print(f"Warning: Local refinement (L-BFGS-B) did not converge: {local_result.message}")

    # Unpack the best found generators (from local refinement) into the full 13-point configuration.
    optimal_generators = local_result.x
    final_optimal_points = _unpack_symmetric_points(optimal_generators)
    
    # Final clipping to ensure points are strictly within the [0,1] range,
    # mitigating any minor boundary violations from optimization.
    final_optimal_points = np.clip(final_optimal_points, 0.0, 1.0)

    return final_optimal_points

# EVOLVE-BLOCK-END