# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
import warnings


def circle_packing32() -> np.ndarray:
    """
    Places 32 non-overlapping circles in the unit square in order to maximize the sum of radii.
    This version uses scipy.optimize.minimize with containment constraints only.

    Returns:
        circles: np.array of shape (32,3), where the i-th row (x,y,r) stores the (x,y) coordinates of the i-th circle of radius r.
    """
    # 1. Define N
    N = 32
    
    # Use a fixed random seed for determinism
    np.random.seed(42)

    # 2. Helper Function `unpack_params`
    def unpack_params(params: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Unpacks the 1D optimization vector into x, y, r arrays."""
        assert params.shape == (N * 3,)
        x = params[0::3]
        y = params[1::3]
        r = params[2::3]
        return x, y, r

    # 3. Objective Function `objective(params)`
    def objective(params: np.ndarray) -> float:
        """The objective function to minimize (negative sum of radii)."""
        _x, _y, r = unpack_params(params)
        return -np.sum(r)

    # 3a. Gradient of Objective Function
    def jac_objective(params: np.ndarray) -> np.ndarray:
        """Gradient of the objective function."""
        grad = np.zeros_like(params)
        # Derivative w.r.t. r_i is -1, others are 0
        grad[2::3] = -1.0
        return grad

    # 4. Bounds Definition
    r_min, r_max = 1e-6, 0.5
    bounds = []
    for i in range(N):
        bounds.append((0, 1))  # x_i
        bounds.append((0, 1))  # y_i
        bounds.append((r_min, r_max)) # r_i
    
    # Helper functions for constraint gradients
    def _jac_containment_x_min(p_val, i):
        grad = np.zeros_like(p_val)
        grad[3*i] = 1.0
        grad[3*i+2] = -1.0
        return grad

    def _jac_containment_x_max(p_val, i):
        grad = np.zeros_like(p_val)
        grad[3*i] = -1.0
        grad[3*i+2] = -1.0
        return grad

    def _jac_containment_y_min(p_val, i):
        grad = np.zeros_like(p_val)
        grad[3*i+1] = 1.0
        grad[3*i+2] = -1.0
        return grad

    def _jac_containment_y_max(p_val, i):
        grad = np.zeros_like(p_val)
        grad[3*i+1] = -1.0
        grad[3*i+2] = -1.0
        return grad

    def _jac_non_overlap(p_val, i, j):
        grad = np.zeros_like(p_val)
        
        # Unpack relevant parameters for clarity in gradient calculation
        xi, yi, ri = p_val[3*i], p_val[3*i+1], p_val[3*i+2]
        xj, yj, rj = p_val[3*j], p_val[3*j+1], p_val[3*j+2]

        # Derivatives w.r.t. x_i, x_j
        grad[3*i] = 2 * (xi - xj)
        grad[3*j] = -2 * (xi - xj) 
        
        # Derivatives w.r.t. y_i, y_j
        grad[3*i+1] = 2 * (yi - yj)
        grad[3*j+1] = -2 * (yi - yj)
        
        # Derivatives w.r.t. r_i, r_j
        grad[3*i+2] = -2 * (ri + rj)
        grad[3*j+2] = -2 * (ri + rj)
        
        return grad

    # 6. Containment Constraints
    constraints = []
    for i in range(N):
        # Constraint: xi - ri >= 0
        constraints.append({'type': 'ineq', 'fun': lambda p, i=i: p[3*i] - p[3*i+2], 'jac': lambda p, i=i: _jac_containment_x_min(p, i)})
        # Constraint: 1 - xi - ri >= 0
        constraints.append({'type': 'ineq', 'fun': lambda p, i=i: 1 - p[3*i] - p[3*i+2], 'jac': lambda p, i=i: _jac_containment_x_max(p, i)})
        # Constraint: yi - ri >= 0
        constraints.append({'type': 'ineq', 'fun': lambda p, i=i: p[3*i+1] - p[3*i+2], 'jac': lambda p, i=i: _jac_containment_y_min(p, i)})
        # Constraint: 1 - yi - ri >= 0
        constraints.append({'type': 'ineq', 'fun': lambda p, i=i: 1 - p[3*i+1] - p[3*i+2], 'jac': lambda p, i=i: _jac_containment_y_max(p, i)})

    # 6b. Non-overlap Constraints
    for i in range(N):
        for j in range(i + 1, N):
            # Constraint: (xi - xj)^2 + (yi - yj)^2 - (ri + rj)^2 >= 0
            constraints.append({
                'type': 'ineq',
                'fun': lambda p, i=i, j=j: \
                    (p[3*i] - p[3*j])**2 + \
                    (p[3*i+1] - p[3*j+1])**2 - \
                    (p[3*i+2] + p[3*j+2])**2,
                'jac': lambda p, i=i, j=j: _jac_non_overlap(p, i, j)
            })

    # Multi-start optimization strategy with improved initialization and tuning
    num_runs = 15 # A balance between exploration (runs) and exploitation (iterations)
    best_sum_radii = -np.inf
    best_circles = np.zeros((N, 3))

    # Create a 6x6 grid of potential center points for better initial distributions.
    # This provides well-spaced, non-overlapping starting configurations.
    grid_size = 6
    margin = 0.1
    x_coords = np.linspace(margin, 1 - margin, grid_size)
    y_coords = np.linspace(margin, 1 - margin, grid_size)
    grid_points = np.array(np.meshgrid(x_coords, y_coords)).T.reshape(-1, 2)
    
    for run_idx in range(num_runs):
        # For each run, generate a new initial guess from a shuffled grid
        np.random.shuffle(grid_points)
        initial_centers = grid_points[:N]
        
        x_initial = initial_centers[:, 0]
        y_initial = initial_centers[:, 1]
        
        # Initial radii are small to ensure no overlap on the grid
        r_initial = np.random.uniform(low=r_min, high=0.05, size=(N,))
        
        initial_guess = np.zeros(N * 3)
        initial_guess[0::3] = x_initial
        initial_guess[1::3] = y_initial
        initial_guess[2::3] = r_initial

        # 7. Scipy Optimization Call with more iterations and tighter tolerance for a more refined search
        res = minimize(
            fun=objective,
            jac=jac_objective, # Provide analytical Jacobian for objective
            x0=initial_guess,
            method='SLSQP',
            bounds=bounds,
            constraints=constraints,
            options={'maxiter': 4000, 'disp': False, 'ftol': 1e-9}
        )

        # 8. Evaluate and store the best result
        if res.success:
            current_sum_radii = -res.fun
            if current_sum_radii > best_sum_radii:
                best_sum_radii = current_sum_radii
                x_opt, y_opt, r_opt = unpack_params(res.x)
                best_circles = np.stack([x_opt, y_opt, r_opt], axis=1)
        # else:
            # Optionally log individual run failures, but not required for final return logic.
            # warnings.warn(f"Optimization run {run_idx+1}/{num_runs} failed to converge: {res.message}")

    # Final Return Value after all runs
    if best_sum_radii == -np.inf: # Indicates no run was successful
        warnings.warn("All optimization runs failed to converge. Returning an empty configuration.")
        return np.zeros((N, 3))
    else:
        return best_circles


# EVOLVE-BLOCK-END
