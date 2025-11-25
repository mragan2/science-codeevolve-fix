# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
from numba import jit

# Constants
N_CIRCLES = 32
MIN_RADIUS = 1e-7 # A very small positive radius to avoid degenerate solutions

@jit(nopython=True)
def _unpack_params(params):
    """Unpacks the 1D parameter array into x, y, r arrays."""
    n = len(params) // 3
    x = params[0::3]
    y = params[1::3]
    r = params[2::3]
    return x, y, r

@jit(nopython=True)
def _objective(params):
    """Objective function for Stage 2: maximize sum of radii -> minimize negative sum."""
    x, y, r = _unpack_params(params)
    return -np.sum(r)

@jit(nopython=True)
def _boundary_constraints_fun(params):
    """Constraint function for Stage 2: boundary containment."""
    x, y, r = _unpack_params(params)
    n = len(x)
    constraints = np.empty(4 * n)
    constraints[0*n : 1*n] = x - r
    constraints[1*n : 2*n] = 1 - x - r
    constraints[2*n : 3*n] = y - r
    constraints[3*n : 4*n] = 1 - y - r
    return constraints

@jit(nopython=True)
def _overlap_constraints_fun(params):
    """Constraint function for Stage 2: non-overlap."""
    x, y, r = _unpack_params(params)
    n = len(x)
    num_overlap_constraints = n * (n - 1) // 2
    overlap_values = np.empty(num_overlap_constraints)
    k = 0
    for i in range(n):
        for j in range(i + 1, n):
            dist_sq = (x[i] - x[j])**2 + (y[i] - y[j])**2
            min_dist_sq = (r[i] + r[j])**2
            overlap_values[k] = dist_sq - min_dist_sq
            k += 1
    return overlap_values

def _initial_sunflower_guess(n_circles):
    """
    Generates a superior initial configuration based on a sunflower seed pattern
    (Fermat's spiral), which distributes points more uniformly than a grid.
    """
    indices = np.arange(n_circles)
    phi = (1 + np.sqrt(5)) / 2  # Golden ratio
    theta = 2 * np.pi * indices / phi
    r_sqrt = np.sqrt((indices + 0.5) / n_circles)
    # Map to square [0.02, 0.98] to avoid starting on the boundary
    x_init = 0.5 + 0.48 * r_sqrt * np.cos(theta)
    y_init = 0.5 + 0.48 * r_sqrt * np.sin(theta)
    return np.stack([x_init, y_init], axis=-1)

def circle_packing32() -> np.ndarray:
    """
    Places 32 circles using a two-stage, multi-start SLSQP optimization strategy,
    inspired by high-performing solutions to avoid slow global optimizers.
    """
    n = N_CIRCLES
    np.random.seed(42)

    # --- Stage 1: Maximize uniform radius to find a dense packing ---
    initial_coords_s1 = _initial_sunflower_guess(n)
    initial_r_s1 = 0.05
    p0_stage1 = np.concatenate([initial_coords_s1.ravel(), [initial_r_s1]])

    def objective_stage1(p): return -p[-1]

    @jit(nopython=True)
    def _constraints_stage1_numba(p, n_circles):
        coords = p[:-1].reshape((n_circles, 2))
        r_uniform = p[-1]
        x, y = coords[:, 0], coords[:, 1]
        cons_boundary = np.empty(4 * n_circles)
        cons_boundary[0*n_circles:1*n_circles] = x - r_uniform
        cons_boundary[1*n_circles:2*n_circles] = 1.0 - x - r_uniform
        cons_boundary[2*n_circles:3*n_circles] = y - r_uniform
        cons_boundary[3*n_circles:4*n_circles] = 1.0 - y - r_uniform
        
        num_overlap = n_circles * (n_circles - 1) // 2
        cons_overlap = np.empty(num_overlap)
        min_dist_sq = (2 * r_uniform)**2
        k = 0
        for i in range(n_circles):
            for j in range(i + 1, n_circles):
                dist_sq = (x[i] - x[j])**2 + (y[i] - y[j])**2
                cons_overlap[k] = dist_sq - min_dist_sq
                k += 1
        return np.concatenate((cons_boundary, cons_overlap))

    def constraints_stage1_wrapper(p): return _constraints_stage1_numba(p, n)

    bounds_stage1 = [(0.0, 1.0)] * (2 * n) + [(MIN_RADIUS, 0.5)]
    
    res_stage1 = minimize(
        objective_stage1, p0_stage1, method='SLSQP', bounds=bounds_stage1,
        constraints={'type': 'ineq', 'fun': constraints_stage1_wrapper},
        options={'maxiter': 1500, 'ftol': 1e-9}
    )
    base_coords_s2 = res_stage1.x[:-1].reshape((n, 2))
    base_r_s2 = res_stage1.x[-1]

    # --- Stage 2: Maximize sum of individual radii (multi-start) ---
    best_sum_radii = -np.inf
    best_circles = None
    num_restarts_s2 = 25

    # Create a perturbed base solution to start Stage 2 exploration
    r_perturbed = base_r_s2 * (1.0 + 0.10 * (2 * np.random.rand(n) - 1))
    coords_perturbed = base_coords_s2 + 0.025 * (2 * np.random.rand(n, 2) - 1)
    base_x0_stage2 = np.hstack([coords_perturbed, np.clip(r_perturbed, MIN_RADIUS, 0.5).reshape(-1, 1)]).ravel()

    bounds_stage2 = [(0.0, 1.0), (0.0, 1.0), (MIN_RADIUS, 0.5)] * n
    constraints_s2 = [{'type': 'ineq', 'fun': _boundary_constraints_fun},
                      {'type': 'ineq', 'fun': _overlap_constraints_fun}]
    slsqp_options_s2 = {'maxiter': 5000, 'ftol': 1e-9, 'eps': 1e-9}

    for k in range(num_restarts_s2):
        current_x0_s2 = np.copy(base_x0_stage2)
        if k > 0:
            current_x0_s2[0::3] += np.random.uniform(-0.01, 0.01, n)
            current_x0_s2[1::3] += np.random.uniform(-0.01, 0.01, n)
            current_x0_s2[2::3] += np.random.uniform(-0.005, 0.005, n)
            for i in range(n):
                current_x0_s2[i*3+0] = np.clip(current_x0_s2[i*3+0], bounds_stage2[i*3+0][0], bounds_stage2[i*3+0][1])
                current_x0_s2[i*3+1] = np.clip(current_x0_s2[i*3+1], bounds_stage2[i*3+1][0], bounds_stage2[i*3+1][1])
                current_x0_s2[i*3+2] = np.clip(current_x0_s2[i*3+2], bounds_stage2[i*3+2][0], bounds_stage2[i*3+2][1])
        
        result_s2 = minimize(_objective, current_x0_s2, method='SLSQP',
                             bounds=bounds_stage2, constraints=constraints_s2,
                             options=slsqp_options_s2)

        if result_s2.success and -result_s2.fun > best_sum_radii:
            best_sum_radii = -result_s2.fun
            best_circles = result_s2.x.reshape((n, 3))

    if best_circles is None:
        best_circles = base_x0_stage2.reshape((n, 3))

    # Final clamping to ensure strict constraint adherence
    best_circles[:, 2] = np.clip(best_circles[:, 2], MIN_RADIUS, 0.5)
    best_circles[:, 0] = np.clip(best_circles[:, 0], best_circles[:, 2], 1 - best_circles[:, 2])
    best_circles[:, 1] = np.clip(best_circles[:, 1], best_circles[:, 2], 1 - best_circles[:, 2])
    
    return best_circles


# EVOLVE-BLOCK-END
