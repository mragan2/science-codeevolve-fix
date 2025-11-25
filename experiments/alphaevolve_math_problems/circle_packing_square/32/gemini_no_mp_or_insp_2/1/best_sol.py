# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
# from scipy.spatial.distance import pdist, squareform # No longer directly used inside numba function
from numba import njit # New import for Numba JIT compilation


# Numba-optimized helper for constraint calculations.
# This replaces the scipy.spatial.distance functions for overlap checks
# to ensure full Numba acceleration and avoid potential overheads.
@njit(cache=True)
def _constraints_numba(params_flat, n):
    """Numba-optimized helper for constraint calculations."""
    p = params_flat.reshape(n, 3)
    xy = p[:, :2]
    r = p[:, 2]

    # a) Containment constraints (4*n constraints)
    #    ri <= xi, ri <= 1-xi  =>  xi - ri >= 0, (1-xi) - ri >= 0
    #    ri <= yi, ri <= 1-yi  =>  yi - ri >= 0, (1-yi) - ri >= 0
    # Use a tuple for np.concatenate to be Numba-compatible
    con_contain = np.concatenate((
        xy[:, 0] - r,
        1.0 - xy[:, 0] - r,
        xy[:, 1] - r,
        1.0 - xy[:, 1] - r
    ))

    # b) Non-overlap constraints (n*(n-1)/2 constraints)
    #    sqrt((xi-xj)^2 + (yi-yj)^2) >= ri + rj
    #    Equivalent to: (xi-xj)^2 + (yi-yj)^2 - (ri+rj)^2 >= 0
    #    Manual distance calculation for Numba compatibility and better performance
    
    # Pre-allocate array for overlap constraints for Numba efficiency
    num_overlap_constraints = n * (n - 1) // 2
    con_overlap = np.empty(num_overlap_constraints, dtype=xy.dtype)
    k = 0
    for i in range(n):
        for j in range(i + 1, n): # Only check unique pairs
            dx = xy[i, 0] - xy[j, 0]
            dy = xy[i, 1] - xy[j, 1]
            dist_sq = dx*dx + dy*dy
            radii_sum_sq = (r[i] + r[j])**2
            con_overlap[k] = dist_sq - radii_sum_sq
            k += 1
            
    # Use a tuple for np.concatenate to be Numba-compatible
    return np.concatenate((con_contain, con_overlap))


def circle_packing32() -> np.ndarray:
    """
    Places 32 non-overlapping circles in the unit square in order to maximize the sum of radii,
    using a Sequential Least Squares Programming (SLSQP) optimizer.
    """
    n = 32
    # Base seed for reproducibility of the overall process
    base_seed = 42
    # Ensure global numpy random state is reproducible for any non-rng calls
    np.random.seed(base_seed)

    # Define optimization parameters
    max_slsqp_iter = 2500  # Base maxiter for multi-start runs
    ftol_slsqp = 1e-10     # Tightened ftol for higher precision in multi-start
    num_starts = 20        # Reduced to budget time for the new "Shake and Refine" phase

    best_sum_radii = -np.inf
    best_circles_solution = np.zeros((n, 3)) # Initialize with an empty solution

    # 2. Objective Function:
    def objective(params):
        """Minimize the negative sum of radii."""
        return -np.sum(params[2::3])

    # 3. Constraints Function:
    def constraints(params):
        """Wrapper for the Numba-optimized constraint function."""
        return _constraints_numba(params, n)

    # 4. Bounds for each variable (x, y, r)
    # 0 <= x, y <= 1
    # 1e-6 < r <= 0.5 (max possible radius)
    bounds = []
    for i in range(n):
        bounds.append((0.0, 1.0))      # x_i
        bounds.append((0.0, 1.0))      # y_i
        bounds.append((1e-6, 0.5))     # r_i (must be positive)

    cons_dict = {'type': 'ineq', 'fun': constraints}

    # Multi-start loop to explore different local optima
    for start_idx in range(num_starts):
        # Generate initial guess for this start with a unique random state
        current_rng = np.random.RandomState(base_seed + start_idx)
        x0 = np.zeros(n * 3)
        
        # Alternate between hexagonal-inspired and purely random initial guesses
        # Half starts from hexagonal, half from random
        if start_idx < num_starts // 2:
            # Hexagonal-inspired initial guess
            row_counts = [6, 5, 6, 5, 6, 4]
            num_rows = len(row_counts)
            max_circles_in_row = max(row_counts)

            r_h = 1.0 / (2 * max_circles_in_row) 
            r_v = 1.0 / ( (num_rows - 1) * np.sqrt(3) + 2 )
            # More slack (0.95) for hexagonal packing
            radius_init = min(r_h, r_v) * 0.95
            
            x_spacing = 2 * radius_init
            y_spacing = np.sqrt(3) * radius_init

            packing_width = (max_circles_in_row - 1) * x_spacing + 2 * radius_init
            packing_height = (num_rows - 1) * y_spacing + 2 * radius_init
            
            x_overall_offset = (1.0 - packing_width) / 2.0
            y_overall_offset = (1.0 - packing_height) / 2.0
            
            current_circle_idx = 0
            for row_idx, count in enumerate(row_counts):
                y_center = y_overall_offset + radius_init + row_idx * y_spacing
                
                row_width_actual = (count - 1) * x_spacing + 2 * radius_init
                x_row_internal_offset = (packing_width - row_width_actual) / 2.0
                
                if row_idx % 2 != 0:
                    x_row_internal_offset += radius_init
                    
                for col_idx in range(count):
                    x_center = x_overall_offset + x_row_internal_offset + radius_init + col_idx * x_spacing
                    
                    x0[3 * current_circle_idx] = x_center
                    x0[3 * current_circle_idx + 1] = y_center
                    x0[3 * current_circle_idx + 2] = radius_init
                    current_circle_idx += 1
                    
            # Apply larger random perturbations to break initial symmetry more aggressively
            perturbation_scale = radius_init / 2 # Increased from /3 for more exploration
            for i in range(n):
                x0[3 * i] += (current_rng.rand() - 0.5) * perturbation_scale
                x0[3 * i + 1] += (current_rng.rand() - 0.5) * perturbation_scale
                x0[3 * i + 2] += (current_rng.rand() - 0.5) * (perturbation_scale / 2)
        else:
            # Purely random initial guess with varied small radii
            for i in range(n):
                # Random radii from a small range for diversity
                r_val = current_rng.uniform(0.005, 0.015) 
                x0[3 * i + 2] = r_val
                # Random positions ensuring containment for the current small radius
                x0[3 * i] = current_rng.uniform(r_val, 1.0 - r_val)
                x0[3 * i + 1] = current_rng.uniform(r_val, 1.0 - r_val)
            # No further perturbation for random starts, they are already diverse.

        # Ensure initial guess remains within optimizer's bounds and respects containment
        for i in range(n):
            r_i = np.clip(x0[3 * i + 2], 1e-6, 0.5)
            x0[3 * i + 2] = r_i
            x0[3 * i] = np.clip(x0[3 * i], r_i, 1.0 - r_i)
            x0[3 * i + 1] = np.clip(x0[3 * i + 1], r_i, 1.0 - r_i)

        # Run the SLSQP Optimizer
        res = minimize(
            objective,
            x0,
            method='SLSQP',
            bounds=bounds,
            constraints=[cons_dict],
            options={'maxiter': max_slsqp_iter, 'ftol': ftol_slsqp, 'disp': False}
        )

        # Update best result if successful and improved
        if res.success:
            current_sum_radii = -res.fun
            if current_sum_radii > best_sum_radii:
                best_sum_radii = current_sum_radii
                best_circles_solution = res.x.reshape(n, 3)
        else:
            # If optimization failed, check if the *initial* guess for this run was feasible
            # and if its sum of radii is better than the current best.
            initial_cons_values = _constraints_numba(x0, n)
            # Use a slightly more lenient tolerance for initial feasibility check
            if np.all(initial_cons_values >= -1e-4):
                current_sum_radii_from_x0 = np.sum(x0[2::3])
                if current_sum_radii_from_x0 > best_sum_radii:
                    best_sum_radii = current_sum_radii_from_x0
                    best_circles_solution = x0.reshape(n, 3)

    # "Shake and Refine" Phase: Apply multiple focused perturbations to the best solution
    if best_sum_radii != -np.inf:
        num_refine_starts = 10 # Number of shakes to perform
        # Use a dedicated RNG for this phase for reproducibility
        refine_rng = np.random.RandomState(base_seed + num_starts) 

        for _ in range(num_refine_starts):
            x_shake_start = best_circles_solution.flatten().copy()
            avg_radius_best = np.mean(x_shake_start[2::3])
            # A more substantial perturbation than the previous "jiggle"
            shake_scale = avg_radius_best * 0.005 # 0.5% of average radius

            # Perturb all parameters
            perturbations = (refine_rng.rand(n * 3) - 0.5) * shake_scale
            x_shake_start += perturbations

            # Re-clip the shaken configuration to ensure it's a valid starting point
            for i in range(n):
                r_i = np.clip(x_shake_start[3 * i + 2], 1e-6, 0.5)
                x_shake_start[3 * i + 2] = r_i
                x_shake_start[3 * i] = np.clip(x_shake_start[3 * i], r_i, 1.0 - r_i)
                x_shake_start[3 * i + 1] = np.clip(x_shake_start[3 * i + 1], r_i, 1.0 - r_i)

            refine_res = minimize(
                objective,
                x_shake_start,
                method='SLSQP',
                bounds=bounds,
                constraints=[cons_dict],
                # Strong refinement settings
                options={'maxiter': 5000, 'ftol': 1e-11, 'disp': False}
            )
            if refine_res.success and -refine_res.fun > best_sum_radii:
                best_sum_radii = -refine_res.fun
                best_circles_solution = refine_res.x.reshape(n, 3)

    # Final Fallback: If no successful optimization or feasible initial guess was ever found
    # (i.e., best_sum_radii is still its initial -np.inf value), return an empty array.
    if best_sum_radii == -np.inf:
        print("Warning: No feasible solution found after multiple starts. Returning empty array.")
        return np.zeros((n, 3))

    return best_circles_solution


# EVOLVE-BLOCK-END
