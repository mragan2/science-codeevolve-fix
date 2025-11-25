# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize, Bounds, NonlinearConstraint # Changed from differential_evolution
# from scipy.spatial.distance import pdist # No longer needed with Numba optimization
import numba # Added numba for JIT compilation
from joblib import Parallel, delayed # Added for parallel multi-start

# --- Helper functions for optimization ---

N_CIRCLES = 32
RANDOM_SEED = 42 # Define a global seed for reproducibility
np.random.seed(RANDOM_SEED)

@numba.njit(cache=True)
def _objective_func_numba(params: np.ndarray) -> float:
    """
    The objective function to be minimized.
    We want to maximize the sum of radii, so we minimize its negative.
    JIT compiled with numba for performance.
    """
    # Radii are every 3rd element starting from index 2
    radii = params[2::3]
    return -np.sum(radii)

@numba.njit(cache=True)
def _objective_jacobian_numba(params: np.ndarray) -> np.ndarray:
    """
    Analytical Jacobian for the objective function.
    The gradient of -sum(r_i) with respect to (x_1, y_1, r_1, ..., x_n, y_n, r_n)
    is [0, 0, -1, 0, 0, -1, ..., 0, 0, -1]
    JIT compiled with numba for performance.
    """
    grad = np.zeros_like(params)
    grad[2::3] = -1.0
    return grad

@numba.njit(cache=True)
def _constraints_func_numba(params: np.ndarray) -> np.ndarray:
    """
    Calculates the constraint violations.
    All constraints are formulated as g(x) >= 0.
    JIT compiled with numba for performance, replacing pdist and np.concatenate.
    """
    circles = params.reshape(N_CIRCLES, 3)
    x, y, r = circles[:, 0], circles[:, 1], circles[:, 2]

    num_containment = 4 * N_CIRCLES
    num_non_overlap = N_CIRCLES * (N_CIRCLES - 1) // 2
    total_constraints = num_containment + num_non_overlap

    # Pre-allocate a single array for all constraints to avoid concatenation overhead
    all_constraints = np.empty(total_constraints, dtype=params.dtype)

    # 1. Containment constraints (4 * N values)
    # r <= x <= 1 - r  => x - r >= 0 and 1 - x - r >= 0
    # r <= y <= 1 - r  => y - r >= 0 and 1 - y - r >= 0
    idx = 0
    for i in range(N_CIRCLES):
        all_constraints[idx] = x[i] - r[i]
        idx += 1
    for i in range(N_CIRCLES):
        all_constraints[idx] = 1 - x[i] - r[i]
        idx += 1
    for i in range(N_CIRCLES):
        all_constraints[idx] = y[i] - r[i]
        idx += 1
    for i in range(N_CIRCLES):
        all_constraints[idx] = 1 - y[i] - r[i]
        idx += 1

    # 2. Non-overlap constraints (N * (N-1) / 2 values)
    # (xi-xj)^2 + (yi-yj)^2 - (ri + rj)^2 >= 0
    if N_CIRCLES >= 2: # Only calculate if there are at least two circles
        for i in range(N_CIRCLES):
            for j in range(i + 1, N_CIRCLES): # Iterate through unique pairs
                dx = x[i] - x[j]
                dy = y[i] - y[j]
                dist_sq = dx*dx + dy*dy
                min_dist_sq = (r[i] + r[j])**2
                all_constraints[idx] = dist_sq - min_dist_sq
                idx += 1
    
    return all_constraints

def circle_packing32() -> np.ndarray:
    """
    Places 32 non-overlapping circles in the unit square in order to maximize the sum of radii.
    This is a highly non-convex global optimization problem. We use a multi-start local
    optimization approach with SLSQP, leveraging Numba for fast function evaluations
    and joblib for parallel execution.

    Returns:
        circles: np.array of shape (32,3), where the i-th row (x,y,r) stores the (x,y)
                 coordinates of the i-th circle of radius r.
    """
    n = N_CIRCLES
    
    # Define bounds for each variable [x1, y1, r1, x2, y2, r2, ...]
    bounds_list = []
    for _ in range(n):
        bounds_list.extend([(0.0, 1.0), (0.0, 1.0), (1e-6, 0.5)]) # (x_min, x_max), (y_min, y_max), (r_min, r_max)
    bounds = Bounds([b[0] for b in bounds_list], [b[1] for b in bounds_list])
        
    # Define the nonlinear constraints. All constraint values must be >= 0.
    num_containment = 4 * n
    num_non_overlap = n * (n - 1) // 2
    total_constraints = num_containment + num_non_overlap
    
    # Lower bound is 0 for all, upper is infinity
    # '2-point' uses finite differences to approximate the Jacobian, which is generally robust.
    # Analytical Jacobian for constraints is complex, so sticking with '2-point' for NonlinearConstraint.
    nonlinear_constraint = NonlinearConstraint(_constraints_func_numba, 0, np.inf, jac='2-point')

    # --- Multi-start Optimization ---
    best_sum_radii = -np.inf
    best_circles = np.zeros((n, 3))
    num_restarts = 800 # Increased number of restarts for better exploration
    max_iterations_per_run = 1500 # Max iterations for each SLSQP run

    def run_single_optimization(seed_offset):
        # Use a unique seed for each parallel run for reproducibility
        # This only affects initial guess generation, not the optimizer itself (SLSQP is deterministic)
        current_rng = np.random.default_rng(RANDOM_SEED + seed_offset)

        # Hybrid Initial Guess: choose between purely random and grid-based with jitter
        if current_rng.random() < 0.5: # 50% chance for pure random, 50% for grid-based
            # Pure random initial guess
            initial_x = current_rng.uniform(0.05, 0.95, n)
            initial_y = current_rng.uniform(0.05, 0.95, n)
            initial_r = current_rng.uniform(0.01, 0.05, n) # Small initial radii
        else:
            # Grid-based initial guess with jitter
            # Attempt a pseudo-hexagonal or square grid for better initial density
            num_rows = int(np.sqrt(n)) # e.g., 5 for 32 circles
            num_cols = int(np.ceil(n / num_rows)) # e.g., 7 for 32 circles (5*7=35 spots)

            # Generate grid coordinates, leaving some margin from the edges
            x_coords_grid = np.linspace(0.1, 0.9, num_cols)
            y_coords_grid = np.linspace(0.1, 0.9, num_rows)

            grid_x, grid_y = np.meshgrid(x_coords_grid, y_coords_grid)
            grid_x = grid_x.flatten()
            grid_y = grid_y.flatten()

            # Take only 'n' points from the grid, shuffling to avoid always taking the same corners
            indices = current_rng.permutation(len(grid_x))
            
            initial_x = grid_x[indices[:n]]
            initial_y = grid_y[indices[:n]]
            initial_r = np.full(n, 0.03) # Start with uniform small radii for grid

            # Add jitter to positions and radii
            jitter_amount_pos = 0.02 # small perturbation for positions
            jitter_amount_r = 0.01 # small perturbation for radii
            initial_x += current_rng.uniform(-jitter_amount_pos, jitter_amount_pos, n)
            initial_y += current_rng.uniform(-jitter_amount_pos, jitter_amount_pos, n)
            initial_r += current_rng.uniform(-jitter_amount_r, jitter_amount_r, n)
            
            # Ensure radii are positive after jitter
            initial_r = np.maximum(initial_r, 1e-6)

        initial_guess = np.stack([initial_x, initial_y, initial_r], axis=-1).flatten()

        # Ensure initial guess satisfies basic containment bounds based on its own radius
        # Order matters: radii first, then x/y centers based on clipped radii
        initial_guess[2::3] = np.clip(initial_guess[2::3], 1e-6, 0.5) # Radii
        initial_guess[0::3] = np.clip(initial_guess[0::3], initial_guess[2::3], 1 - initial_guess[2::3]) # X centers
        initial_guess[1::3] = np.clip(initial_guess[1::3], initial_guess[2::3], 1 - initial_guess[2::3]) # Y centers

        try:
            res = minimize(
                _objective_func_numba,
                initial_guess,
                method='SLSQP',
                jac=_objective_jacobian_numba, # Use analytical Jacobian for the objective
                bounds=bounds,
                constraints=[nonlinear_constraint],
                options={'ftol': 1e-9, 'maxiter': max_iterations_per_run, 'disp': False}
            )

            # Check if the optimization was successful and constraints are met
            if res.success:
                current_sum_radii = -res.fun # objective returns -sum_radii
                # Verify constraints numerically with a small tolerance for floating-point inaccuracies
                constraint_violations = _constraints_func_numba(res.x)
                if np.all(constraint_violations >= -1e-7): # check if all g(x) >= -tolerance
                    return (current_sum_radii, res.x.reshape((n, 3)))
            
        except Exception as e:
            # print(f"Run {seed_offset}: An error occurred during optimization: {e}")
            pass # Suppress error messages to keep output clean during many restarts
        return None # Indicate failure or invalid result

    # Parallelize the multi-start optimization runs
    parallel_results = Parallel(n_jobs=-1)(
        delayed(run_single_optimization)(i) for i in range(num_restarts)
    )

    # Aggregate results from parallel runs
    for result in parallel_results:
        if result is not None:
            current_sum_radii, current_circles = result
            if current_sum_radii > best_sum_radii:
                best_sum_radii = current_sum_radii
                best_circles = current_circles

    if best_sum_radii == -np.inf:
        # If no valid solution found after all restarts, return an array of zero radii
        # A warning might be printed if disp=True in `minimize` or outside this function.
        circles = np.zeros((n, 3))
    else:
        circles = best_circles
    
    return circles


# EVOLVE-BLOCK-END
