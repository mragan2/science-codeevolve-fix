# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize, Bounds, NonlinearConstraint
from joblib import Parallel, delayed

# Set a fixed random seed for full reproducibility of the multi-start process.
RANDOM_SEED = 42

def circle_packing32() -> np.ndarray:
    """
    Generates an optimal arrangement of 32 non-overlapping circles within a unit square [0,1] × [0,1],
    by maximizing the sum of their radii using a parallelized multi-start optimization strategy.

    This approach combines the strengths of the inspiration programs:
    - Multi-start parallel optimization to explore the non-convex search space.
    - A hybrid initial guess strategy (random + perturbed grid) for diversity.
    - Fully analytical Jacobians for both objective and constraints for speed and precision.
    - Modern SciPy interface with Bounds and NonlinearConstraint objects.
    - Aggressive optimization parameters to aim for a high-precision solution.

    Returns:
        circles: np.array of shape (32,3), where each row (x, y, r) represents a circle.
    """
    n = 32
    num_restarts = 600  # High number of restarts for thorough exploration.
    best_sum_radii = -np.inf
    best_circles = None

    optimization_options = {
        'maxiter': 3000,  # Further increased iterations to allow deeper convergence for high precision.
        'ftol': 1e-12,    # Very tight tolerance for high precision.
        'disp': False,
        'eps': 1e-10      # Step size for finite difference approximation (used for internal checks/fallbacks).
    }

    # Objective Function: Minimize the negative sum of radii.
    def objective(p):
        return -np.sum(p[2::3])

    # Gradient of the Objective Function (Analytical Jacobian).
    def objective_grad(p):
        grad = np.zeros_like(p)
        grad[2::3] = -1.0
        return grad

    # Combined function for all inequality constraints (g(p) >= 0).
    def all_constraints_func(p):
        params = p.reshape(n, 3)
        x, y, r = params[:, 0], params[:, 1], params[:, 2]
        containment = np.concatenate((x - r, 1 - x - r, y - r, 1 - y - r))
        pos = params[:, :2]
        dist_sq = np.sum((pos[:, np.newaxis, :] - pos[np.newaxis, :, :])**2, axis=-1)
        r_sum_sq = (r[:, np.newaxis] + r[np.newaxis, :])**2
        iu = np.triu_indices(n, k=1)
        overlap = dist_sq[iu] - r_sum_sq[iu]
        return np.concatenate((containment, overlap))

    # Jacobian of the Constraint Function (Analytical & Vectorized).
    def all_constraints_jac(p):
        params = p.reshape(n, 3)
        x, y, r = params[:, 0], params[:, 1], params[:, 2]
        num_vars = n * 3
        num_containment_constraints = 4 * n
        num_overlap_constraints = n * (n - 1) // 2
        jac = np.zeros((num_containment_constraints + num_overlap_constraints, num_vars))

        # Vectorized population of Jacobian for containment constraints (inspired by Inspiration 1).
        x_indices, y_indices, r_indices = np.arange(0, num_vars, 3), np.arange(1, num_vars, 3), np.arange(2, num_vars, 3)
        rows = np.arange(n)
        jac[rows, x_indices] = 1.0; jac[rows, r_indices] = -1.0
        jac[n + rows, x_indices] = -1.0; jac[n + rows, r_indices] = -1.0
        jac[2*n + rows, y_indices] = 1.0; jac[2*n + rows, r_indices] = -1.0
        jac[3*n + rows, y_indices] = -1.0; jac[3*n + rows, r_indices] = -1.0

        # Loop-based population for non-overlap constraints.
        k = num_containment_constraints
        for i in range(n):
            for j in range(i + 1, n):
                dx, dy, r_sum = x[i] - x[j], y[i] - y[j], r[i] + r[j]
                jac[k, 3*i] = 2 * dx; jac[k, 3*j] = -2 * dx
                jac[k, 3*i + 1] = 2 * dy; jac[k, 3*j + 1] = -2 * dy
                jac[k, 3*i + 2] = -2 * r_sum; jac[k, 3*j + 2] = -2 * r_sum
                k += 1
        return jac

    nlc = NonlinearConstraint(all_constraints_func, 0, np.inf, jac=all_constraints_jac)

    # Bounds for each variable [x, y, r] for all circles.
    bounds_list = [(0.0, 1.0), (0.0, 1.0), (1e-6, 0.5)] * n
    bounds = Bounds([b[0] for b in bounds_list], [b[1] for b in bounds_list])

    # Function for a single optimization run, designed to be parallelized.
    def run_single_optimization(seed_offset):
        np.random.seed(RANDOM_SEED + seed_offset)
        
        # Hybrid Initial Guess Strategy: 50% random, 50% perturbed grid.
        if np.random.rand() < 0.5:
            # Pure random initial guess with slightly wider radius range for more diversity
            initial_x = np.random.uniform(0.05, 0.95, n)
            initial_y = np.random.uniform(0.05, 0.95, n)
            initial_r = np.random.uniform(0.005, 0.06, n) # Wider range for radii
        else:
            # Dynamic grid-based initial guess with increased jitter for positions and radii
            num_rows = int(np.floor(np.sqrt(n)))
            num_cols = int(np.ceil(n / num_rows))
            x_coords = np.linspace(0.1, 0.9, num_cols)
            y_coords = np.linspace(0.1, 0.9, num_rows)
            grid_x, grid_y = np.meshgrid(x_coords, y_coords)
            indices = np.arange(len(grid_x.flatten()))
            np.random.shuffle(indices)
            
            # Increased jitter for positions to explore more variations around grid points
            jitter_pos = np.random.uniform(-0.03, 0.03, n) # Increased from +/-0.02 to +/-0.03
            initial_x = grid_x.flatten()[indices[:n]] + jitter_pos
            initial_y = grid_y.flatten()[indices[:n]] + jitter_pos
            
            # Slightly larger base radius for grid and increased jitter for radii
            initial_r = np.full(n, 0.035) + np.random.uniform(-0.015, 0.015, n) # Base 0.035, jitter +/-0.015
        
        p0 = np.stack([initial_x, initial_y, initial_r], axis=-1).flatten()
        
        # Clip initial guess to be within a more feasible region.
        p0[2::3] = np.clip(p0[2::3], bounds.lb[2::3], bounds.ub[2::3])
        p0[0::3] = np.clip(p0[0::3], p0[2::3], 1 - p0[2::3])
        p0[1::3] = np.clip(p0[1::3], p0[2::3], 1 - p0[2::3])

        try:
            res = minimize(objective, p0, method='SLSQP', jac=objective_grad,
                           bounds=bounds, constraints=[nlc], options=optimization_options)
            if res.success and np.all(all_constraints_func(res.x) >= -1e-7):
                return (-res.fun, res.x.reshape((n, 3)))
        except Exception:
            pass
        return None

    # Execute the multi-start optimization in parallel.
    parallel_results = Parallel(n_jobs=-1)(
        delayed(run_single_optimization)(i) for i in range(num_restarts)
    )

    # Aggregate results and find the best valid solution.
    for result in parallel_results:
        if result is not None:
            current_sum_radii, current_circles = result
            if current_sum_radii > best_sum_radii:
                best_sum_radii = current_sum_radii
                best_circles = current_circles

    if best_circles is None:
        return np.zeros((n, 3))  # Fallback if no run succeeds.
    
    return best_circles


# EVOLVE-BLOCK-END
