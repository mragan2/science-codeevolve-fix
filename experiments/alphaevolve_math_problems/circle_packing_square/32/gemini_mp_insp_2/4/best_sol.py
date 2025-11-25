# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize, Bounds, NonlinearConstraint
# from itertools import combinations # No longer needed for vectorized constraints
from joblib import Parallel, delayed # Added for parallel processing

# Set a fixed random seed for reproducibility
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)

def circle_packing32() -> np.ndarray:
    """
    Places 32 non-overlapping circles in the unit square in order to maximize the sum of radii.

    Returns:
        circles: np.array of shape (32,3), where the i-th row (x,y,r) stores the (x,y) coordinates of the i-th circle of radius r.
    """
    n = 32
    # num_params = n * 3 # x, y, r for each circle - not explicitly used, but good for clarity

    # --- Objective Function ---
    def objective(params):
        # We want to maximize sum_radii, so we minimize -sum_radii
        radii = params[2::3] # Every 3rd element starting from index 2 is a radius
        return -np.sum(radii)

    # --- Objective Function Jacobian ---
    # The gradient of -sum(r_i) with respect to (x_1, y_1, r_1, ..., x_n, y_n, r_n)
    # is [0, 0, -1, 0, 0, -1, ..., 0, 0, -1]
    def objective_jacobian(params):
        grad = np.zeros_like(params)
        grad[2::3] = -1.0
        return grad

    # --- Constraint Functions (Vectorized) ---
    # 1. Non-overlap constraints: (x_i - x_j)^2 + (y_i - y_j)^2 >= (r_i + r_j)^2
    # This translates to (x_i - x_j)^2 + (y_i - y_j)^2 - (r_i + r_j)^2 >= 0
    def non_overlap_constraints_vectorized(params):
        circles = params.reshape((n, 3))
        centers = circles[:, :2] # (N, 2)
        radii = circles[:, 2]    # (N,)

        # Compute squared distances between all pairs of centers using broadcasting
        # (N, 1, 2) - (1, N, 2) -> (N, N, 2)
        diff_centers = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
        dist_sq_matrix = np.sum(diff_centers**2, axis=2) # (N, N)

        # Compute squared sum of radii for all pairs using broadcasting
        # (N, 1) + (1, N) -> (N, N)
        radii_sum_sq_matrix = (radii[:, np.newaxis] + radii[np.newaxis, :])**2 # (N, N)

        # Extract upper triangle (excluding diagonal) for unique pairs
        # np.triu_indices(n, k=1) gives indices (rows, cols) for upper triangle
        upper_tri_indices = np.triu_indices(n, k=1)
        
        # Constraints: dist_sq - radii_sum_sq >= 0
        return (dist_sq_matrix - radii_sum_sq_matrix)[upper_tri_indices]

    # 2. Containment constraints: r_i <= x_i <= 1-r_i and r_i <= y_i <= 1-r_i
    # This translates to x_i - r_i >= 0, 1 - x_i - r_i >= 0, y_i - r_i >= 0, 1 - y_i - r_i >= 0
    def containment_constraints_vectorized(params):
        circles = params.reshape((n, 3))
        xi, yi, ri = circles[:, 0], circles[:, 1], circles[:, 2]
        
        # Concatenate all 4*N containment constraints
        return np.concatenate([
            xi - ri,          # x_i - r_i >= 0
            1 - xi - ri,       # 1 - x_i - r_i >= 0
            yi - ri,          # y_i - r_i >= 0
            1 - yi - ri        # 1 - y_i - r_i >= 0
        ])

    # Combine all non-linear inequality constraints
    # The lower bound for all these functions is 0 (i.e., g(x) >= 0)
    # The upper bound can be infinity (no upper limit on how much they can exceed 0)
    all_constraints_fun = lambda params: np.concatenate([
        non_overlap_constraints_vectorized(params),
        containment_constraints_vectorized(params)
    ])

    # Pre-calculate constants for Jacobian for efficiency and clarity
    num_params = n * 3
    num_non_overlap_constraints = n * (n - 1) // 2
    num_containment_constraints = 4 * n
    total_constraints = num_non_overlap_constraints + num_containment_constraints

    def all_constraints_jacobian(params):
        # Initialize Jacobian matrix with zeros
        jac = np.zeros((total_constraints, num_params))
        circles = params.reshape((n, 3))
        xs, ys, rs = circles[:, 0], circles[:, 1], circles[:, 2]

        # Non-overlap constraints Jacobian (first part of Jacobian matrix)
        rows_i, cols_j = np.triu_indices(n, k=1) # Indices for i and j circles in each pair (i < j)
        
        dx_ij = xs[rows_i] - xs[cols_j]
        dy_ij = ys[rows_i] - ys[cols_j]
        sum_r_ij = rs[rows_i] + rs[cols_j]
        
        # Global row indices for non-overlap constraints (0 to num_non_overlap_constraints - 1)
        k_non_overlap_global_rows = np.arange(num_non_overlap_constraints)
        
        # Derivatives for circle i (rows_i) parameters
        # d/dx_i = 2*(x_i - x_j)
        jac[k_non_overlap_global_rows, 3 * rows_i + 0] = 2 * dx_ij
        # d/dy_i = 2*(y_i - y_j)
        jac[k_non_overlap_global_rows, 3 * rows_i + 1] = 2 * dy_ij
        # d/dr_i = -2*(r_i + r_j)
        jac[k_non_overlap_global_rows, 3 * rows_i + 2] = -2 * sum_r_ij

        # Derivatives for circle j (cols_j) parameters
        # d/dx_j = -2*(x_i - x_j)
        jac[k_non_overlap_global_rows, 3 * cols_j + 0] = -2 * dx_ij
        # d/dy_j = -2*(y_i - y_j)
        jac[k_non_overlap_global_rows, 3 * cols_j + 1] = -2 * dy_ij
        # d/dr_j = -2*(r_i + r_j)
        jac[k_non_overlap_global_rows, 3 * cols_j + 2] = -2 * sum_r_ij

        # Containment constraints Jacobian (second part of Jacobian matrix)
        # Constraints are ordered: x-r, 1-x-r, y-r, 1-y-r
        
        idx = np.arange(n) # Global indices for circles (0 to n-1)
        
        # x_i - r_i >= 0 (starts at row num_non_overlap_constraints)
        current_row_offset = num_non_overlap_constraints
        jac[current_row_offset + idx, 3 * idx + 0] = 1.0   # d(xi-ri)/d(xi)
        jac[current_row_offset + idx, 3 * idx + 2] = -1.0  # d(xi-ri)/d(ri)

        # 1 - x_i - r_i >= 0 (starts at row num_non_overlap_constraints + n)
        current_row_offset += n
        jac[current_row_offset + idx, 3 * idx + 0] = -1.0 # d(1-xi-ri)/d(xi)
        jac[current_row_offset + idx, 3 * idx + 2] = -1.0 # d(1-xi-ri)/d(ri)

        # y_i - r_i >= 0 (starts at row num_non_overlap_constraints + 2n)
        current_row_offset += n
        jac[current_row_offset + idx, 3 * idx + 1] = 1.0  # d(yi-ri)/d(yi)
        jac[current_row_offset + idx, 3 * idx + 2] = -1.0 # d(yi-ri)/d(ri)

        # 1 - y_i - r_i >= 0 (starts at row num_non_overlap_constraints + 3n)
        current_row_offset += n
        jac[current_row_offset + idx, 3 * idx + 1] = -1.0 # d(1-yi-ri)/d(yi)
        jac[current_row_offset + idx, 3 * idx + 2] = -1.0 # d(1-yi-ri)/d(ri)

        return jac

    # Create a NonlinearConstraint object with analytical Jacobian
    nlc = NonlinearConstraint(all_constraints_fun, 0, np.inf, jac=all_constraints_jacobian)

    # --- Bounds for parameters ---
    # For each (x, y, r) triplet:
    # x: [0, 1], y: [0, 1], r: [1e-6, 0.5]
    bounds_list = []
    for _ in range(n):
        bounds_list.extend([(0.0, 1.0), (0.0, 1.0), (1e-6, 0.5)])
    bounds = Bounds([b[0] for b in bounds_list], [b[1] for b in bounds_list])

    # --- Multi-start Optimization ---
    best_sum_radii = -np.inf
    best_circles = np.zeros((n, 3))
    num_restarts = 720 # Increased number of restarts for more thorough exploration (double 360)
    max_iterations_per_run = 3000 # Adjusted maxiter per run to balance with increased restarts (from 4000)

    # print(f"Starting {num_restarts} multi-start optimizations for N={n} circles...")

    def run_single_optimization(seed_offset):
        # Use a unique RandomState for each parallel run for thread-safe reproducibility
        rng = np.random.RandomState(RANDOM_SEED + seed_offset)

        # Hybrid initial guess strategy (50% grid-based, 50% random)
        if rng.rand() < 0.5:
            # Grid-based placement with jitter
            grid_size = int(np.ceil(np.sqrt(n)))
            coords = np.linspace(0.1, 0.9, grid_size) # Spread centers more widely
            xx, yy = np.meshgrid(coords, coords)
            
            # Randomly select N points from the grid (if grid_size^2 > N)
            indices = np.arange(grid_size**2)
            rng.shuffle(indices)
            
            initial_x = xx.ravel()[indices[:n]]
            initial_y = yy.ravel()[indices[:n]]
            
            # Add jitter to break symmetry
            jitter_amount = 0.03 # Magnitude of jitter
            initial_x += rng.uniform(-jitter_amount, jitter_amount, n)
            initial_y += rng.uniform(-jitter_amount, jitter_amount, n)
            
            initial_r = rng.uniform(0.02, 0.06, n) # Slightly wider radius range for grid-based starts
        else:
            # Fully random initial guess
            initial_x = rng.uniform(0.05, 0.95, n)
            initial_y = rng.uniform(0.05, 0.95, n)
            initial_r = rng.uniform(0.005, 0.07, n) # Slightly wider radius range for random starts

        initial_guess = np.stack([initial_x, initial_y, initial_r], axis=-1).flatten()

        # Ensure initial guess satisfies basic containment bounds based on its own radius
        initial_guess[2::3] = np.clip(initial_guess[2::3], 1e-6, 0.5) # Clip radii
        initial_guess[0::3] = np.clip(initial_guess[0::3], initial_guess[2::3], 1 - initial_guess[2::3]) # Clip X centers
        initial_guess[1::3] = np.clip(initial_guess[1::3], initial_guess[2::3], 1 - initial_guess[2::3]) # Clip Y centers

        try:
            res = minimize(objective, initial_guess, method='SLSQP',
                           jac=objective_jacobian, # Use analytical Jacobian for the objective
                           bounds=bounds, constraints=[nlc],
                           options={'ftol': 1e-9, 'maxiter': max_iterations_per_run, 'disp': False, 'gtol': 1e-7, 'eps': 1e-10}) # Added eps for numerical precision

            # Check if the optimization was successful and constraints are met
            if res.success:
                current_sum_radii = -res.fun # objective returns -sum_radii
                # Verify constraints numerically with a small tolerance for floating-point inaccuracies
                constraint_violations = all_constraints_fun(res.x)
                if np.all(constraint_violations >= -1e-7): # check if all g(x) >= -tolerance
                    return (current_sum_radii, res.x.reshape((n, 3)))
        except Exception:
            # Suppress errors from individual runs to keep output clean during many restarts
            pass
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
        # Fallback: if no valid solution found after all parallel runs, try a deterministic grid-based start.
        # This increases robustness against all random starts failing.
        grid_dim = int(np.ceil(np.sqrt(n)))
        x_coords = np.linspace(1 / (2 * grid_dim), 1 - 1 / (2 * grid_dim), grid_dim)
        y_coords = np.linspace(1 / (2 * grid_dim), 1 - 1 / (2 * grid_dim), grid_dim)
        xx, yy = np.meshgrid(x_coords, y_coords)
        fallback_initial_pos = np.stack([xx.ravel(), yy.ravel()], axis=1)[:n]
        fallback_initial_radii = np.full(n, 1 / (2 * grid_dim) * 0.8) # Small enough to avoid immediate overlap
        fallback_p0 = np.hstack((fallback_initial_pos, fallback_initial_radii[:, np.newaxis])).ravel()
        
        try:
            fallback_result = minimize(
                objective, fallback_p0, method='SLSQP', jac=objective_jacobian,
                bounds=bounds, constraints=[nlc],
                options={'maxiter': 5000, 'ftol': 1e-9, 'gtol': 1e-7, 'disp': False} # High maxiter for thorough last attempt
            )
            # Check for success and constraint satisfaction for the fallback result
            if fallback_result.success:
                constraint_violations = all_constraints_fun(fallback_result.x)
                if np.all(constraint_violations >= -1e-7):
                    # If fallback found a valid solution, it becomes the best (since best_sum_radii was -inf)
                    best_sum_radii = -fallback_result.fun
                    best_circles = fallback_result.x.reshape((n, 3))
        except Exception:
            pass # Suppress any errors from the fallback run
        
        if best_sum_radii == -np.inf: # If fallback also fails to find a valid solution
            return np.zeros((n, 3)) # Return an array of zero radii as a safe default
    
    return best_circles


# EVOLVE-BLOCK-END
