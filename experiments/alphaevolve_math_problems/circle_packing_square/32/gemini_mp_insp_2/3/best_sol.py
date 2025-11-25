# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize, Bounds, NonlinearConstraint
from joblib import Parallel, delayed

# --- Global constants and functions for parallel execution ---
RANDOM_SEED = 42
N_CIRCLES = 32

def _objective(p):
    """Objective: Minimize the negative sum of radii."""
    return -np.sum(p[2::3])

def _objective_grad(p):
    """Gradient of the objective function."""
    grad = np.zeros_like(p)
    grad[2::3] = -1.0
    return grad

def _constraints_func(p):
    """All constraints (containment and non-overlap) vectorized."""
    n = N_CIRCLES
    params = p.reshape(n, 3)
    x, y, r = params[:, 0], params[:, 1], params[:, 2]
    containment = np.concatenate((x - r, 1 - x - r, y - r, 1 - y - r))
    pos = params[:, :2]
    dist_sq = np.sum((pos[:, np.newaxis, :] - pos[np.newaxis, :, :])**2, axis=-1)
    r_sum_sq = (r[:, np.newaxis] + r[np.newaxis, :])**2
    iu = np.triu_indices(n, k=1)
    overlap = dist_sq[iu] - r_sum_sq[iu]
    return np.concatenate((containment, overlap))

def _constraints_jac(p):
    """Jacobian of the constraint function (fully vectorized)."""
    n = N_CIRCLES
    params = p.reshape(n, 3)
    x, y, r = params[:, 0], params[:, 1], params[:, 2]
    num_vars = n * 3
    num_containment = 4 * n
    num_overlap = n * (n - 1) // 2
    jac = np.zeros((num_containment + num_overlap, num_vars))
    
    idx = np.arange(n)
    jac[idx, 3 * idx] = 1.0; jac[idx, 3 * idx + 2] = -1.0
    jac[n + idx, 3 * idx] = -1.0; jac[n + idx, 3 * idx + 2] = -1.0
    jac[2 * n + idx, 3 * idx + 1] = 1.0; jac[2 * n + idx, 3 * idx + 2] = -1.0
    jac[3 * n + idx, 3 * idx + 1] = -1.0; jac[3 * n + idx, 3 * idx + 2] = -1.0

    rows, cols = np.triu_indices(n, k=1)
    dx, dy, r_sum = x[rows] - x[cols], y[rows] - y[cols], r[rows] + r[cols]
    k = np.arange(num_overlap) + num_containment
    
    jac[k, 3 * rows] = 2 * dx; jac[k, 3 * rows + 1] = 2 * dy; jac[k, 3 * rows + 2] = -2 * r_sum
    jac[k, 3 * cols] = -2 * dx; jac[k, 3 * cols + 1] = -2 * dy; jac[k, 3 * cols + 2] = -2 * r_sum
    return jac

def _run_single_optimization(seed_offset, bounds, nlc):
    """A single optimization run for parallel execution."""
    n = N_CIRCLES
    local_rng = np.random.RandomState(RANDOM_SEED + seed_offset)
    
    if local_rng.rand() < 0.5:
        # Jittered grid-based initial guess
        grid_size = int(np.ceil(np.sqrt(n)))
        # Adjust grid coordinates to cover more of the square, then jitter
        coords = np.linspace(0.1, 0.9, grid_size)
        xx, yy = np.meshgrid(coords, coords)
        indices = np.arange(grid_size**2)
        local_rng.shuffle(indices)
        
        # Increased jitter magnitude for more exploration
        initial_x = xx.ravel()[indices[:n]] + local_rng.uniform(-0.03, 0.03, n)
        initial_y = yy.ravel()[indices[:n]] + local_rng.uniform(-0.03, 0.03, n)
        
        # Adjusted radius range for grid-based starts (slightly higher average, tighter range)
        initial_r = local_rng.uniform(0.03, 0.06, n)
    else:
        # Fully random initial guess
        initial_x = local_rng.uniform(0.05, 0.95, n)
        initial_y = local_rng.uniform(0.05, 0.95, n)
        # Adjusted radius range for random starts (wider range for more diversity)
        initial_r = local_rng.uniform(0.01, 0.07, n)

    initial_guess = np.stack([initial_x, initial_y, initial_r], axis=-1).flatten()
    # Ensure initial guess satisfies basic bounds and containment before optimization
    initial_guess[2::3] = np.clip(initial_guess[2::3], 1e-6, 0.5)
    initial_guess[0::3] = np.clip(initial_guess[0::3], initial_guess[2::3], 1 - initial_guess[2::3])
    initial_guess[1::3] = np.clip(initial_guess[1::3], initial_guess[2::3], 1 - initial_guess[2::3])
    
    try:
        # Tighter optimization tolerances and increased maxiter for higher precision
        result = minimize(
            _objective, initial_guess, method='SLSQP', jac=_objective_grad,
            bounds=bounds, constraints=[nlc],
            options={'maxiter': 5000, 'ftol': 1e-11, 'gtol': 1e-8, 'disp': False, 'eps': 1e-10}
        )
        # Post-optimization check for success and constraint satisfaction with a small tolerance
        if result.success and np.all(_constraints_func(result.x) >= -1e-7):
            return -result.fun, result.x
    except Exception:
        pass # Suppress exceptions from individual runs to not halt the entire parallel process
    return None

def circle_packing32() -> np.ndarray:
    """
    Generates an optimal arrangement of 32 circles using a parallelized multi-start
    SLSQP optimizer with fully vectorized analytical Jacobians.
    """
    n = N_CIRCLES
    np.random.seed(RANDOM_SEED) # Seed for overall reproducibility
    
    # Define bounds for x, y, r for all circles
    bounds_list = []
    for _ in range(n):
        bounds_list.extend([(0.0, 1.0), (0.0, 1.0), (1e-6, 0.5)]) # r must be positive
    bounds = Bounds([b[0] for b in bounds_list], [b[1] for b in bounds_list])
    
    # Define non-linear constraints using the vectorized functions and Jacobians
    nlc = NonlinearConstraint(_constraints_func, 0, np.inf, jac=_constraints_jac)
    
    # Increased number of restarts for broader exploration (from 360 to 500)
    num_restarts = 500 
    
    # Execute optimization runs in parallel
    parallel_results = Parallel(n_jobs=-1)(
        delayed(_run_single_optimization)(i, bounds, nlc) for i in range(num_restarts)
    )
    
    best_sum_radii = -np.inf
    best_params = None
    # Aggregate results from parallel runs
    for res in parallel_results:
        if res is not None:
            current_sum_radii, current_params = res
            if current_sum_radii > best_sum_radii:
                best_sum_radii = current_sum_radii
                best_params = current_params

    # Return an empty array if no feasible solution was found (should be rare)
    if best_params is None:
        # Fallback: try one more local run with a default initial guess
        # This can catch cases where all random starts fail, but a deterministic one might succeed.
        # This is a robust fallback, ensuring some circles are returned.
        grid_dim = int(np.ceil(np.sqrt(n)))
        x_coords = np.linspace(1 / (2 * grid_dim), 1 - 1 / (2 * grid_dim), grid_dim)
        y_coords = np.linspace(1 / (2 * grid_dim), 1 - 1 / (2 * grid_dim), grid_dim)
        xx, yy = np.meshgrid(x_coords, y_coords)
        fallback_initial_pos = np.stack([xx.ravel(), yy.ravel()], axis=1)[:n]
        fallback_initial_radii = np.full(n, 1 / (2 * grid_dim) * 0.8) # Slightly smaller to avoid immediate overlap
        fallback_p0 = np.hstack((fallback_initial_pos, fallback_initial_radii[:, np.newaxis])).ravel()
        
        try:
            # Fallback uses the same tightened options for a thorough last attempt
            fallback_result = minimize(
                _objective, fallback_p0, method='SLSQP', jac=_objective_grad,
                bounds=bounds, constraints=[nlc],
                options={'maxiter': 5000, 'ftol': 1e-11, 'gtol': 1e-8, 'disp': False, 'eps': 1e-10}
            )
            if fallback_result.success and np.all(_constraints_func(fallback_result.x) >= -1e-7):
                if -fallback_result.fun > best_sum_radii: # Compare with overall best, even if it was -inf
                    best_params = fallback_result.x
        except Exception:
            pass
        
        if best_params is None: # If fallback also fails
            return np.zeros((n, 3))
        
    return best_params.reshape(n, 3)


# EVOLVE-BLOCK-END
