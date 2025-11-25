# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
from numba import jit

# Numba-accelerated function for Stage 1 constraints (uniform radius)
@jit(nopython=True, cache=True)
def _numba_constraints_stage1(p, n_circles):
    num_overlap_constraints = n_circles * (n_circles - 1) // 2
    total_constraints = 4 * n_circles + num_overlap_constraints
    cons_values = np.empty(total_constraints, dtype=p.dtype)
    
    coords = p[:-1].reshape((n_circles, 2))
    r_uniform = p[-1]
    
    idx = 0
    # Containment constraints
    for i in range(n_circles):
        cons_values[idx] = coords[i, 0] - r_uniform
        idx += 1
        cons_values[idx] = 1 - coords[i, 0] - r_uniform
        idx += 1
        cons_values[idx] = coords[i, 1] - r_uniform
        idx += 1
        cons_values[idx] = 1 - coords[i, 1] - r_uniform
        idx += 1

    # Non-overlap constraints
    for i in range(n_circles):
        for j in range(i + 1, n_circles):
            dx = coords[i, 0] - coords[j, 0]
            dy = coords[i, 1] - coords[j, 1]
            dist_sq = dx*dx + dy*dy
            min_dist_sq = (2 * r_uniform)**2
            cons_values[idx] = dist_sq - min_dist_sq
            idx += 1
    return cons_values

# Numba-accelerated function for Stage 2 constraints and their Jacobian
@jit(nopython=True, cache=True)
def _numba_constraints_and_jac_stage2(p, n_circles):
    num_params = n_circles * 3
    num_containment_cons = n_circles * 4
    num_overlap_cons = n_circles * (n_circles - 1) // 2
    total_cons = num_containment_cons + num_overlap_cons

    values = np.empty(total_cons, dtype=p.dtype)
    jacobian = np.zeros((total_cons, num_params), dtype=p.dtype)
    
    # --- Part 1: Containment Constraints ---
    for i in range(n_circles):
        c_idx_base = i * 4
        p_idx_base = i * 3
        x_i, y_i, r_i = p[p_idx_base], p[p_idx_base+1], p[p_idx_base+2]
        
        values[c_idx_base] = x_i - r_i
        jacobian[c_idx_base, p_idx_base] = 1.0; jacobian[c_idx_base, p_idx_base+2] = -1.0
        
        values[c_idx_base+1] = 1 - x_i - r_i
        jacobian[c_idx_base+1, p_idx_base] = -1.0; jacobian[c_idx_base+1, p_idx_base+2] = -1.0
        
        values[c_idx_base+2] = y_i - r_i
        jacobian[c_idx_base+2, p_idx_base+1] = 1.0; jacobian[c_idx_base+2, p_idx_base+2] = -1.0
        
        values[c_idx_base+3] = 1 - y_i - r_i
        jacobian[c_idx_base+3, p_idx_base+1] = -1.0; jacobian[c_idx_base+3, p_idx_base+2] = -1.0

    # --- Part 2: Overlap Constraints ---
    k = num_containment_cons
    for i in range(n_circles):
        p_idx_i = i * 3
        x_i, y_i, r_i = p[p_idx_i], p[p_idx_i+1], p[p_idx_i+2]
        for j in range(i + 1, n_circles):
            p_idx_j = j * 3
            x_j, y_j, r_j = p[p_idx_j], p[p_idx_j+1], p[p_idx_j+2]
            
            dx = x_i - x_j
            dy = y_i - y_j
            sum_r = r_i + r_j
            
            values[k] = dx*dx + dy*dy - sum_r*sum_r
            
            # Jacobian for this constraint
            dC_d_sum_r = -2 * sum_r
            jacobian[k, p_idx_i] = 2 * dx; jacobian[k, p_idx_i+1] = 2 * dy; jacobian[k, p_idx_i+2] = dC_d_sum_r
            jacobian[k, p_idx_j] = -2 * dx; jacobian[k, p_idx_j+1] = -2 * dy; jacobian[k, p_idx_j+2] = dC_d_sum_r
            k += 1
            
    return values, jacobian

def circle_packing32() -> np.ndarray:
    n_circles = 32
    np.random.seed(42)
    
    # --- Initial guess generation (Sunflower seed pattern) ---
    indices = np.arange(n_circles)
    phi = (1 + np.sqrt(5)) / 2
    theta = 2 * np.pi * indices / phi
    r_sqrt = np.sqrt((indices + 0.5) / n_circles)
    x_init = 0.5 + 0.49 * r_sqrt * np.cos(theta)
    y_init = 0.5 + 0.49 * r_sqrt * np.sin(theta)
    initial_coords = np.stack([x_init, y_init], axis=-1)
    p0_stage1 = np.concatenate([initial_coords.ravel(), [0.05]])

    # --- Stage 1: Maximize uniform radius (get a good starting layout) ---
    res_stage1 = minimize(
        lambda p: -p[-1], p0_stage1, method='SLSQP',
        bounds=[(0.0, 1.0)] * (2*n_circles) + [(1e-7, 0.5)],
        constraints={'type': 'ineq', 'fun': lambda p: _numba_constraints_stage1(p, n_circles)},
        options={'maxiter': 2000, 'ftol': 1e-11, 'disp': False} # Increased precision for Stage 1
    )

    coords_stage1_result = res_stage1.x[:-1].reshape((n_circles, 2))
    r_uniform_stage1_result = res_stage1.x[-1] if res_stage1.success and res_stage1.x[-1] > 1e-6 else 0.08
    
    # --- Stage 2: Multi-start SLSQP with Analytical Gradients ---
    bounds_stage2 = [(0.0, 1.0), (0.0, 1.0), (1e-7, 0.5)] * n_circles
    
    # Objective and its gradient
    def objective_stage2(p): return -np.sum(p[2::3])
    def jac_objective_stage2(p):
        grad = np.zeros_like(p)
        grad[2::3] = -1.0
        return grad

    # Constraints and their Jacobian (wrappers for the numba function)
    def constraints_stage2(p):
        values, _ = _numba_constraints_and_jac_stage2(p, n_circles)
        return values
    def jac_constraints_stage2(p):
        _, jac = _numba_constraints_and_jac_stage2(p, n_circles)
        return jac

    best_sum_radii = -np.inf
    best_final_circles = None

    N_ATTEMPTS = 20 # Increased number of multi-start attempts to explore more local optima
    perturbation_strength_r = 0.30 # Increased radius perturbation to encourage more varied circle sizes
    perturbation_strength_coords = 0.01

    for attempt in range(N_ATTEMPTS):
        random_factors_r = 1.0 + perturbation_strength_r * (2 * np.random.rand(n_circles) - 1)
        r_perturbed = np.clip(r_uniform_stage1_result * random_factors_r, 1e-7, 0.5-1e-9)
        
        random_factors_coords = perturbation_strength_coords * (2 * np.random.rand(n_circles, 2) - 1)
        coords_perturbed = np.clip(coords_stage1_result + random_factors_coords, 1e-7, 1.0-1e-7)
        
        p0_stage2 = np.hstack([coords_perturbed, r_perturbed.reshape(-1, 1)]).ravel()

        res_stage2 = minimize(
            objective_stage2, p0_stage2, method='SLSQP', jac=jac_objective_stage2,
            bounds=bounds_stage2,
            constraints={'type': 'ineq', 'fun': constraints_stage2, 'jac': jac_constraints_stage2},
            options={'maxiter': 15000, 'ftol': 1e-12, 'disp': False} # Increased maxiter, tighter ftol
        )

        if res_stage2.success:
            current_circles = res_stage2.x.reshape((n_circles, 3))
            current_sum_radii = np.sum(current_circles[:, 2])
            
            # Post-validation check with a small tolerance
            violations, _ = _numba_constraints_and_jac_stage2(res_stage2.x, n_circles)
            if np.all(violations >= -1e-8):
                if current_sum_radii > best_sum_radii:
                    best_sum_radii = current_sum_radii
                    best_final_circles = current_circles

    if best_final_circles is None:
        final_circles = np.hstack([coords_stage1_result, np.full((n_circles, 1), r_uniform_stage1_result)])
    else:
        final_circles = best_final_circles
    
    return final_circles


# EVOLVE-BLOCK-END
