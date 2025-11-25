# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import differential_evolution, minimize # Added minimize for local refinement
from scipy.spatial import ConvexHull
from numba import jit
import itertools
from scipy.stats import qmc # For Sobol sequence initialization

# Helper function to calculate triangle area - JIT compiled for speed
@jit(nopython=True, cache=True, fastmath=True) # Added fastmath for potential performance boost
def _triangle_area(p1: np.ndarray, p2: np.ndarray, p3: np.ndarray) -> float:
    """Calculates the area of a triangle given three 2D points using Numba."""
    return 0.5 * np.abs(p1[0]*(p2[1] - p3[1]) + p2[0]*(p3[1] - p1[1]) + p3[0]*(p1[1] - p2[1]))

# JIT-compiled function to efficiently gather all C(n,3) triangle areas
@jit(nopython=True, cache=True, fastmath=True) # Added fastmath
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

# JIT-compiled function to construct 13 points from 5 generators using 3-fold symmetry, with optimizable center
@jit(nopython=True, cache=True, fastmath=True) # Added fastmath
def _unpack_symmetric_points(generators_flat: np.ndarray) -> np.ndarray:
    """
    Constructs 13 points from 5 generator points (1 center + 4 peripheral)
    assuming 3-fold rotational symmetry. All generated points are clipped to [0,1].
    The first two elements of generators_flat are (center_x, center_y).
    The next 8 elements are for 4 peripheral generator points (gx, gy).
    """
    points = np.empty((13, 2), dtype=np.float64)
    
    # First two elements are the center (xc, yc)
    # Clip center coordinates to be within [0,1] for robustness
    center_x = max(0.0, min(1.0, generators_flat[0]))
    center_y = max(0.0, min(1.0, generators_flat[1]))
    points[0, 0] = center_x
    points[0, 1] = center_y
    
    # Pre-calculate rotation matrix elements for 120 degrees
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
        
        # Rotate by 240 degrees (rotate the 120-deg point again) and clip
        rot2_x = rot1_x * cos120 - rot1_y * sin120
        rot2_y = rot1_x * sin120 + rot1_y * cos120
        points[current_idx + 2, 0] = max(0.0, min(1.0, rot2_x + center_x))
        points[current_idx + 2, 1] = max(0.0, min(1.0, rot2_y + center_y))

        current_idx += 3
        
    return points

# Objective function using LogSumExp (LSE) approximation and Convex Hull normalization
def _symmetric_objective_function_lse(generators_flat: np.ndarray, n: int, lse_alpha: float) -> float:
    """
    Wrapper objective function that:
    1. Generates 13 symmetric points from 4 generator points.
    2. Calculates all triangle areas using Numba.
    3. Computes the LogSumExp (LSE) approximation of the minimum area.
    4. Normalizes by the convex hull area.
    Returns the negative of the normalized LSE-approximated minimum area (to be minimized).
    """
    full_points = _unpack_symmetric_points(generators_flat)
    
    # Check for duplicate points (with rounding for numerical stability)
    # Using np.round for robust comparison as in Inspiration 2
    unique_points = np.unique(np.round(full_points, 12), axis=0)
    if len(unique_points) < n:
        return 1e10 # Heavy penalty for non-unique points

    all_areas = _calculate_all_triangle_areas_numba(full_points, n)

    # Find the true minimum area to check for degeneracy and for LSE stabilization.
    min_true_area = np.min(all_areas)
    if min_true_area < 1e-12: # Threshold for considering an area as effectively zero
        return 1e10 # Penalize collinear or coincident points heavily

    # Calculate convex hull area for normalization.
    try:
        # ConvexHull requires at least 3 unique, non-collinear points.
        # Use the already checked unique_points to avoid issues.
        if len(unique_points) < 3: # If less than 3 unique points, ConvexHull will fail
            return 1e10
        hull = ConvexHull(unique_points) # Use unique points for hull computation
        hull_area = hull.volume # hull.volume is area in 2D
    except Exception: # Handle cases where ConvexHull might fail (e.g., all points collinear)
        return 1e10 # Penalize degenerate convex hulls

    if hull_area < 1e-12: # Also penalize extremely small hull areas
        return 1e10

    # LogSumExp approximation for min(all_areas).
    # Stabilization trick: subtract min_true_area before exp to prevent numerical overflow/underflow.
    # Ensure all_areas are positive before scaling for LSE.
    # Filter out any zero areas before LSE if min_true_area was 0 (already penalized above).
    positive_areas = all_areas[all_areas > 1e-15] # Ensure positive, non-degenerate areas for LSE
    if len(positive_areas) == 0: # All areas were degenerate after filtering
        return 1e10
    
    min_true_area_for_lse = np.min(positive_areas) # Use the min of positive areas for LSE stabilization
    scaled_areas = -lse_alpha * (positive_areas - min_true_area_for_lse)
    
    # To handle potential overflow/underflow in exp, we can further clip scaled_areas.
    # A common range is [-700, 700] for float64, but LSE is designed to handle this by subtracting min_true_area.
    # Adding a safety clamp just in case.
    scaled_areas = np.clip(scaled_areas, -100, 100) # Clamp to prevent extreme values if lse_alpha is very high
    
    log_sum_exp = np.log(np.sum(np.exp(scaled_areas)))
    min_area_approx = min_true_area_for_lse - log_sum_exp / lse_alpha

    # The goal is to maximize min_area_normalized, so we minimize its negative.
    return -min_area_approx / hull_area

def heilbronn_convex13()->np.ndarray:
    """
    Generates a high-quality arrangement of 13 points by optimizing a smaller set
    of "generator" points under a 3-fold symmetry constraint. It employs a multi-start
    Differential Evolution approach with a smooth LogSumExp objective, convex hull
    normalization, and Sobol sequence initialization for enhanced performance and solution quality.
    """
    n = 13
    # n_generators_total: 1 center point (x,y) + 4 peripheral generators (x,y) = 5*2 = 10 dimensions
    n_generators_total = 5 
    
    # Bounds for all generator points (center and 4 peripheral) within the unit square [0,1]x[0,1].
    bounds = [(0.0, 1.0)] * (n_generators_total * 2) # Explicitly use float bounds

    best_min_neg_normalized_area = float('inf')
    best_generators_flat = None

    # LSE smoothing parameters for a two-phase optimization strategy (from Inspirations 1 & 2).
    # A smoother landscape (lower alpha) for global search (DE).
    lse_alpha_global = 50000.0 # From Inspiration 2
    # A tighter approximation (higher alpha) for precise local refinement (L-BFGS-B).
    lse_alpha_local = 500000.0 # From Inspiration 2

    # Multi-start optimization strategy to improve robustness and solution quality.
    num_starts = 10             # Increased number of starts for broader global exploration
    maxiter_per_start = 2500    # Increased iterations for DE (from Inspiration 2)
    popsize_val = 40            # Increased population size for better diversity (from Inspiration 2)
    tol_val = 1e-6              # Tighter tolerance for higher precision (from Inspiration 2)

    print(f"Starting multi-start Differential Evolution with {num_starts} runs (lse_alpha_global={lse_alpha_global})...")

    # Use a single master seed for overall reproducibility
    master_seed = 42

    for i in range(num_starts):
        current_seed = master_seed + i # Vary seed for different initial populations
        
        # Generate initial population using Sobol sequence for better space coverage (from Inspiration 2).
        sampler = qmc.Sobol(d=n_generators_total * 2, seed=current_seed)
        # The number of points for Sobol must be a power of 2 for random_base2.
        # Generate more than needed and truncate to popsize_val.
        m = int(np.ceil(np.log2(popsize_val)))
        initial_population = sampler.random_base2(m=m)[:popsize_val]

        result_de = differential_evolution(
            func=_symmetric_objective_function_lse,
            bounds=bounds,
            args=(n, lse_alpha_global), # Use global alpha for smoother landscape
            strategy='best1bin',
            maxiter=maxiter_per_start,
            popsize=popsize_val,
            tol=tol_val,
            mutation=(0.5, 1.0),
            recombination=0.7,
            seed=current_seed,
            disp=False,
            polish=True, # LSE makes polishing effective even in global phase
            init=initial_population, # Initialize with Sobol sequence points
            workers=-1, # Use all available CPU cores for parallel evaluation
            updating='deferred' # Necessary for parallel workers
        )
        
        if result_de.fun < best_min_neg_normalized_area:
            best_min_neg_normalized_area = result_de.fun
            best_generators_flat = result_de.x

    if best_generators_flat is None:
        print("Error: No valid point configuration found after multiple DE runs. Falling back to random symmetric configuration.")
        rng = np.random.default_rng(seed=master_seed)
        best_generators_flat = rng.random(n_generators_total * 2)

    # --- Final Local Refinement with Tighter LSE Approximation (from Inspirations 1 & 2) ---
    print(f"\nApplying final local refinement with L-BFGS-B (lse_alpha_local={lse_alpha_local})...")
    
    def local_objective(x):
        # Use the tighter LSE approximation for high-precision refinement
        return _symmetric_objective_function_lse(x, n, lse_alpha_local)

    refinement_result = minimize(
        fun=local_objective,
        x0=best_generators_flat,
        method='L-BFGS-B',
        bounds=bounds,
        options={'ftol': 1e-12, 'gtol': 1e-12, 'maxiter': 10000, 'disp': False} # Tighter tolerances, disable disp
    )
    
    if refinement_result.success:
        print(f"Local refinement successful. Final LSE objective: {refinement_result.fun:.8f}")
        final_generators_flat = refinement_result.x
    else:
        print(f"Local refinement failed: {refinement_result.message}. Using best DE result.")
        final_generators_flat = best_generators_flat

    # Unpack the best found generators (potentially refined) into the full 13-point configuration
    optimal_points = _unpack_symmetric_points(final_generators_flat)
    
    # Final clip to ensure points are strictly within the [0,1] range.
    optimal_points = np.clip(optimal_points, 0.0, 1.0)

    return optimal_points

# EVOLVE-BLOCK-END