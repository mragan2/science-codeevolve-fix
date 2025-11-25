# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import differential_evolution, minimize
from scipy.spatial import ConvexHull
from numba import njit # Use njit alias for jit(nopython=True)
from scipy.stats import qmc

# Constants for the Heilbronn problem for n=13
N_POINTS = 13
N_GENERATORS = 4 # 1 central point + 4 generators, each generating 3 points (itself + 2 rotations) = 1 + 4*3 = 13
OPTIMIZATION_SEED = 42
# The convex region is assumed to be the unit square [0,1]x[0,1]
# as it's a standard choice for this problem and simplifies bounds.
# Bounds for the *generator* points (8 dimensions total)
GENERATOR_BOUNDS = [(0.0, 1.0) for _ in range(N_GENERATORS * 2)]

# Rotation constants for 120 and 240 degrees (for Numba compatibility and performance)
# Derived from R_120 * R_120 for explicit values.
COS120 = -0.5
SIN120 = np.sqrt(3.0) / 2.0
COS240 = COS120 * COS120 - SIN120 * SIN120
SIN240 = 2 * SIN120 * COS120

# Precompute rotation matrices for gradient calculation in Python (Numba will handle constants)
R_120_GLOBAL = np.array([[COS120, -SIN120], [SIN120, COS120]])
R_240_GLOBAL = np.array([[COS240, -SIN240], [SIN240, COS240]])


@njit(cache=True)
def _calculate_triangle_area(p1_x, p1_y, p2_x, p2_y, p3_x, p3_y) -> float:
    """
    Calculates the area of a triangle given its three vertices.
    Optimized for Numba with explicit coordinate arguments for type stability.
    """
    # Shoelace formula variant 0.5 * |(x2-x1)(y3-y1) - (x3-x1)(y2-y1)|
    return 0.5 * np.abs((p2_x - p1_x) * (p3_y - p1_y) - (p3_x - p1_x) * (p2_y - p1_y))

@njit(cache=True)
def _calculate_all_triangle_areas_numba(points_flat: np.ndarray, n: int) -> np.ndarray:
    """
    Calculates all C(n,3) triangle areas for a set of n points.
    Optimized for Numba.
    """
    num_triangles = n * (n - 1) * (n - 2) // 6
    all_areas = np.empty(num_triangles, dtype=np.float64)
    points = points_flat.reshape(n, 2)
    
    count = 0
    for i in range(n):
        for j in range(i + 1, n):
            for k in range(j + 1, n):
                p1, p2, p3 = points[i], points[j], points[k]
                area = _calculate_triangle_area(p1[0], p1[1], p2[0], p2[1], p3[0], p3[1])
                all_areas[count] = area
                count += 1
    return all_areas

@njit(cache=True)
def _calculate_min_triangle_area_numba_exact(points_flat: np.ndarray, n: int) -> float:
    """
    Finds the exact minimum area among all possible triangles.
    Used for the "hard-min" objective function in the final polish.
    Includes an early exit for near-degenerate triangles (from Inspirations 1 & 3).
    """
    points = points_flat.reshape((n, 2))
    min_area = np.finfo(np.float64).max
    for i in range(n):
        for j in range(i + 1, n):
            for k in range(j + 1, n):
                area = _calculate_triangle_area(points[i,0], points[i,1], points[j,0], points[j,1], points[k,0], points[k,1])
                if area < 1e-12: # Early exit: if any triangle is near-degenerate, return 0.0
                    return 0.0    # This will trigger a heavy penalty in the objective function
                if area < min_area:
                    min_area = area
    return min_area

# JIT-compiled function to construct 13 points from 4 generators using 3-fold symmetry
@njit(cache=True)
def _unpack_symmetric_points(generators_flat: np.ndarray) -> np.ndarray:
    """
    Constructs 13 points from 4 generator points assuming 3-fold rotational symmetry
    around a central point (0.5, 0.5) in the unit square.
    """
    points = np.empty((N_POINTS, 2), dtype=np.float64)
    generators = generators_flat.reshape((N_GENERATORS, 2))
    
    center_x, center_y = 0.5, 0.5
    points[0, 0] = center_x
    points[0, 1] = center_y # The central fixed point
    
    current_idx = 1
    for i in range(N_GENERATORS):
        gx, gy = generators[i, 0], generators[i, 1]
        
        # Original generator point
        points[current_idx, 0], points[current_idx, 1] = gx, gy
        
        # Relative coordinates for rotation
        rel_x, rel_y = gx - center_x, gy - center_y
        
        # Rotated by 120 degrees
        rot1_x = rel_x * COS120 - rel_y * SIN120
        rot1_y = rel_x * SIN120 + rel_y * COS120
        points[current_idx + 1, 0] = rot1_x + center_x
        points[current_idx + 1, 1] = rot1_y + center_y
        
        # Rotated by 240 degrees (rotate the 120-deg point again)
        rot2_x = rot1_x * COS120 - rot1_y * SIN120
        rot2_y = rot1_x * SIN120 + rot1_y * COS120
        points[current_idx + 2, 0] = rot2_x + center_x
        points[current_idx + 2, 1] = rot2_y + center_y

        current_idx += 3
        
    return points

@njit(cache=True)
def _penalty_out_of_bounds(points_flat: np.ndarray) -> float:
    """
    Calculates a penalty for points outside the [0,1]x[0,1] unit square.
    Increased penalty factor for stronger boundary enforcement.
    """
    penalty = 0.0
    penalty_factor = 50.0 # Increased from 10.0 for stronger penalty
    for coord in points_flat:
        if coord < 0.0:
            penalty += (coord * penalty_factor)**2
        elif coord > 1.0:
            penalty += ((coord - 1.0) * penalty_factor)**2
    return penalty

@njit(cache=True)
def _grad_penalty_out_of_bounds(points_flat: np.ndarray) -> np.ndarray:
    """
    Calculates the gradient of the penalty function.
    Adjusted for the increased penalty factor.
    """
    grad = np.zeros_like(points_flat)
    penalty_factor = 50.0 # Consistent with _penalty_out_of_bounds
    for i, coord in enumerate(points_flat):
        if coord < 0.0:
            grad[i] = 2.0 * coord * (penalty_factor**2) # d/dx (x*factor)^2 = 2*x*factor^2
        elif coord > 1.0:
            grad[i] = 2.0 * (coord - 1.0) * (penalty_factor**2) # d/dx ((x-1)*factor)^2 = 2*(x-1)*factor^2
    return grad

@njit(cache=True)
def _triangle_area_gradient_numba(p1_x, p1_y, p2_x, p2_y, p3_x, p3_y):
    """
    Calculates the gradient of the *absolute* triangle area with respect to its 6 coordinates.
    Based on V = (p2_x - p1_x)*(p3_y - p1_y) - (p3_x - p1_x)*(p2_y - p1_y). Area = 0.5 * |V|.
    The derivative d(abs(V))/dx = sign(V) * dV/dx.
    """
    grad = np.empty(6, dtype=np.float64)

    V = (p2_x - p1_x) * (p3_y - p1_y) - (p3_x - p1_x) * (p2_y - p1_y)
    
    # Handle the case where V is exactly zero to avoid sign(0) issue
    sign_V = np.sign(V) if V != 0.0 else 0.0

    # Gradient of 0.5 * abs(V)
    # dV/dx1 = y2 - y3
    grad[0] = 0.5 * sign_V * (p2_y - p3_y) 
    # dV/dy1 = x3 - x2
    grad[1] = 0.5 * sign_V * (p3_x - p2_x) 
    # dV/dx2 = y3 - y1
    grad[2] = 0.5 * sign_V * (p3_y - p1_y) 
    # dV/dy2 = x1 - x3
    grad[3] = 0.5 * sign_V * (p1_x - p3_x) 
    # dV/dx3 = y1 - y2
    grad[4] = 0.5 * sign_V * (p1_y - p2_y) 
    # dV/dy3 = x2 - x1
    grad[5] = 0.5 * sign_V * (p2_x - p1_x) 

    return grad

# --- Objective functions for optimization ---

def _objective_function_normalized_with_penalty(generators_flat: np.ndarray) -> float:
    """
    Calculates the negative of the true (hard) smallest triangle area,
    CORRECTLY NORMALIZED by the convex hull area, plus a boundary penalty.
    Used for Differential Evolution and Powell's method (minimization).
    """
    full_points_flat = _unpack_symmetric_points(generators_flat).flatten()
    n_points = N_POINTS
    points = full_points_flat.reshape((n_points, 2))

    # Check for duplicate points (within tolerance)
    if len(np.unique(points, axis=0)) < n_points:
        return np.finfo(np.float64).max # Penalize heavily for non-unique points

    min_tri_area = _calculate_min_triangle_area_numba_exact(full_points_flat, n_points)

    if min_tri_area < 1e-12: # Penalize near-collinear points
        return np.finfo(np.float64).max

    try:
        hull = ConvexHull(points)
        hull_area = hull.volume # volume for 2D is area
    except Exception: # Catch QhullError for degenerate configurations
        return np.finfo(np.float64).max

    if hull_area < 1e-12: # Penalize collapsed configurations
        return np.finfo(np.float64).max
    
    normalized_area = min_tri_area / hull_area
    penalty = _penalty_out_of_bounds(full_points_flat)
    
    # We want to maximize normalized_area, so we minimize -normalized_area.
    # Add penalty to this for minimization.
    return -normalized_area + penalty

@njit(cache=True)
def _smooth_objective_and_grad_numba(points_flat: np.ndarray, n: int, alpha: float):
    """
    Efficiently calculates both the smooth objective value (LogSumExp) and its gradient
    with respect to all point coordinates in a single Numba-jitted pass.
    This objective aims to MAXIMIZE the minimum *raw* area, so it returns the negative
    of the smooth approximation to `min_area`.
    """
    points = points_flat.reshape(n, 2)
    grad_total = np.zeros_like(points_flat, dtype=np.float64)

    all_areas = _calculate_all_triangle_areas_numba(points_flat, n)
    
    # --- Calculate Objective Value (corrected for maximization) ---
    # LogSumExp stabilization for maximizing min_area: -(1/alpha) * log(sum(exp(alpha * area_i)))
    # Let x_i = alpha * area_i. Stabilize with c = max(x_i) = alpha * max(area_i)
    
    max_area_val = np.max(all_areas)
    x_stabilized = alpha * (all_areas - max_area_val)
    
    # log(sum(exp(x_i))) = c + log(sum(exp(x_i - c)))
    log_sum_exp_val = alpha * max_area_val + np.log(np.sum(np.exp(x_stabilized)))
    
    # The smooth approximation to min_area is S_approx = (1/alpha) * log_sum_exp_val
    # We want to minimize -S_approx for scipy.optimize.minimize
    value = -(1.0 / alpha) * log_sum_exp_val

    # --- Calculate Gradient (corrected for maximization) ---
    # dF/dP_k = - sum_j (w_j * dA_j/dP_k) where w_j = exp(alpha * A_j) / sum_m exp(alpha * A_m)
    # Using stabilized weights: w_j = exp(x_stabilized_j) / sum_m exp(x_stabilized_m)
    
    sum_exp_x_stabilized = np.sum(np.exp(x_stabilized))
    # Handle numerical issues, fall back to non-smooth value if underflow
    if sum_exp_x_stabilized <= 1e-30:
        # This fallback for gradient is a simplification, but prevents division by zero
        # and large numerical errors. It implies extremely small/degenerate triangles,
        # which are heavily penalized by the objective function's value.
        return value, grad_total 

    weights = np.exp(x_stabilized) / sum_exp_x_stabilized

    count = 0
    for i in range(n):
        for j in range(i + 1, n):
            for k in range(j + 1, n):
                p1, p2, p3 = points[i], points[j], points[k]
                
                grad_area_ijk = _triangle_area_gradient_numba(p1[0], p1[1], p2[0], p2[1], p3[0], p3[1])
                weight_ijk = weights[count]

                # Accumulate weighted gradient: grad_F = - sum(weight_k * grad_area_k)
                grad_total[i*2]     -= weight_ijk * grad_area_ijk[0]
                grad_total[i*2 + 1] -= weight_ijk * grad_area_ijk[1]
                grad_total[j*2]     -= weight_ijk * grad_area_ijk[2]
                grad_total[j*2 + 1] -= weight_ijk * grad_area_ijk[3]
                grad_total[k*2]     -= weight_ijk * grad_area_ijk[4]
                grad_total[k*2 + 1] -= weight_ijk * grad_area_ijk[5]
                
                count += 1
    
    return value, grad_total

def _smooth_symmetric_objective_and_grad(generators_flat: np.ndarray, alpha: float) -> tuple[float, np.ndarray]:
    """
    Python wrapper that combines all parts of the objective and gradient calculation
    for the L-BFGS-B optimizer, using the efficient combined Numba function.
    Returns (total_value, total_gradient_wrt_generators).
    """
    full_points_flat = _unpack_symmetric_points(generators_flat).flatten()
    
    # Get smooth value and gradient w.r.t points from the efficient Numba function
    smooth_val, grad_smooth_points_flat = _smooth_objective_and_grad_numba(full_points_flat, N_POINTS, alpha)
    
    # Get penalty value and gradient w.r.t points
    penalty_val = _penalty_out_of_bounds(full_points_flat)
    grad_penalty_points_flat = _grad_penalty_out_of_bounds(full_points_flat)

    # Combine values and gradients
    total_value = smooth_val + penalty_val
    grad_full_points_total = grad_smooth_points_flat + grad_penalty_points_flat
    
    grad_generators_flat = np.zeros_like(generators_flat, dtype=np.float64)
    
    # Propagate gradients back through the symmetry transformation (Chain Rule)
    # points[0] is the center (0.5, 0.5), it does not contribute to generator gradients
    
    for j in range(N_GENERATORS):
        # Extract gradient components for the original generator point and its rotations
        # The indices for points are 1 + j*3 (original), 1 + j*3 + 1 (120 deg), 1 + j*3 + 2 (240 deg)
        
        # Contribution to generator from the original generator point P_j = G_j
        # dP_j/dG_j is identity matrix
        grad_generators_flat[j*2 : j*2 + 2] += grad_full_points_total[(1 + j*3)*2 : (1 + j*3)*2 + 2]
        
        # Contribution from 120-degree rotated point
        grad_p_j_rot1 = grad_full_points_total[(1 + j*3 + 1)*2 : (1 + j*3 + 1)*2 + 2]
        # Chain rule: dF/dG = dF/dP_rot * dP_rot/dG. Since P_rot = R * (G - C) + C, dP_rot/dG = R.
        # So dF/dG = (dP_rot/dG)^T * dF/dP_rot = R^T * dF/dP_rot
        grad_generators_flat[j*2 : j*2 + 2] += np.dot(R_120_GLOBAL.T, grad_p_j_rot1) 

        # Contribution from 240-degree rotated point
        grad_p_j_rot2 = grad_full_points_total[(1 + j*3 + 2)*2 : (1 + j*3 + 2)*2 + 2]
        grad_generators_flat[j*2 : j*2 + 2] += np.dot(R_240_GLOBAL.T, grad_p_j_rot2) 
    
    return total_value, grad_generators_flat


def heilbronn_convex13() -> np.ndarray:
    """
    Constructs an arrangement of exactly 13 points within a unit square
    to maximize the area of the smallest triangle formed by these points,
    normalized by the convex hull area.
    
    This uses a sophisticated multi-stage hybrid optimization strategy:
    1. Dimensionality Reduction: Exploits 3-fold rotational symmetry for N=13
       to optimize 4 generator points (8 dimensions) instead of 13 points (26 dimensions).
    2. Global Search (Differential Evolution): Multi-start DE with Sobol initialization
       optimizes the hard-min normalized objective + boundary penalty.
    3. Local Refinement (L-BFGS-B with Continuation Method): Uses an efficient,
       combined smooth objective/gradient function (maximizing raw min area + penalty),
       gradually increasing 'alpha' to refine the solution with exact analytical gradients.
    4. Final Polish (Powell): A derivative-free method on the true, non-smooth,
       normalized objective + penalty to fine-tune the result.
    """
    
    num_generator_dims = N_GENERATORS * 2
    
    best_overall_obj_value = np.finfo(np.float64).max # Minimize this
    best_overall_generators = None

    # --- Phase 1: Multi-start Differential Evolution (Global Search on Normalized Metric) ---
    print("Starting multi-start Differential Evolution (global search on normalized metric)...")
    # Reverting to parameters that previously yielded a strong min_area_normalized
    # with reasonable evaluation time. Aggressively increasing them further didn't help.
    de_runs = 7 # Reverted from 10 runs
    de_maxiter = 2500 # Reverted from 3000 iterations
    de_popsize = 70   # Reverted from 80 population size
    de_tol = 1e-7     # Stricter tolerance for DE (unchanged)
    
    # Generate a large pool of Sobol points once for multi-start initialization
    sobol_sampler = qmc.Sobol(d=num_generator_dims, seed=OPTIMIZATION_SEED)
    sobol_pool = sobol_sampler.random(de_popsize * de_runs + 1) # +1 for injecting initial guess

    for i in range(de_runs):
        current_seed = OPTIMIZATION_SEED + i
        rng = np.random.default_rng(seed=current_seed)
        
        # Prepare initial population for current DE run
        start_idx = i * de_popsize
        init_pop_slice = sobol_pool[start_idx : start_idx + de_popsize]
        
        # Inject a geometrically-inspired symmetric initial guess for the first run
        if i == 0:
            print("  Injecting a symmetric initial guess for the first DE run.")
            # A simple symmetric configuration for generators
            # These are relative to the center 0.5,0.5. Values are adjusted to be within [0,1]
            initial_guess_generators = np.array([
                [0.2, 0.5], # Generator 1
                [0.5, 0.2], # Generator 2
                [0.7, 0.7], # Generator 3
                [0.3, 0.3]  # Generator 4
            ]).flatten()
            # Ensure it's within bounds by clipping and add some noise
            initial_guess_generators = np.clip(initial_guess_generators, 0.05, 0.95)
            # Increased perturbation for initial geometric guess to enhance diversity (from -0.01, 0.01)
            initial_guess_generators += rng.uniform(-0.02, 0.02, size=num_generator_dims)
            initial_guess_generators = np.clip(initial_guess_generators, 0, 1)
            
            # Replace one of the Sobol points with the geometric guess
            init_pop_slice[0] = initial_guess_generators
        
        result_de = differential_evolution(
            func=_objective_function_normalized_with_penalty,
            bounds=GENERATOR_BOUNDS,
            args=(), # N_POINTS is a global constant
            seed=current_seed,
            maxiter=de_maxiter,
            popsize=de_popsize,
            init=init_pop_slice, # Use the prepared initial population
            tol=de_tol,
            disp=False,
            workers=-1,
            polish=True, # Apply local optimization to the best DE solution
            strategy='best1bin'
        )

        print(f"  DE Run {i+1}/{de_runs} finished. Best objective for this run: {result_de.fun:.8f}")
        if result_de.fun < best_overall_obj_value:
            best_overall_obj_value = result_de.fun
            best_overall_generators = result_de.x
            print(f"  New best overall DE objective found: {best_overall_obj_value:.8f}")
        
        if not result_de.success:
            print(f"  Warning: DE Run {i+1} did not converge successfully: {result_de.message}")

    if best_overall_generators is None:
        raise RuntimeError("Differential Evolution failed to find any valid solution.")

    current_generators = best_overall_generators
    print(f"\nDE phase complete. Best objective: {best_overall_obj_value:.8f}")

    # --- Phase 2: Local Refinement with L-BFGS-B (Continuation Method on Raw Area) ---
    print("Starting local refinement with L-BFGS-B (continuation method, raw area)...")
    # Adopted denser and expanded alpha schedule from Inspiration Program 2
    # This provides a more gradual continuation path for L-BFGS-B.
    alpha_schedule = [100.0, 300.0, 1000.0, 3000.0, 10000.0, 30000.0, 100000.0, 300000.0, 1000000.0, 3000000.0] # 10 steps
    
    for i, alpha in enumerate(alpha_schedule):
        print(f"  L-BFGS-B step {i+1}/{len(alpha_schedule)} with alpha={alpha:.1f}...")
        local_result = minimize(
            fun=_smooth_symmetric_objective_and_grad,
            x0=current_generators,
            args=(alpha,),
            method='L-BFGS-B',
            jac=True, # Inform the optimizer that the function returns the gradient
            bounds=GENERATOR_BOUNDS,
            options={'maxiter': 2000, 'ftol': 1e-15, 'gtol': 1e-12} # Maxiter reverted to 2000 for balance
        )
        
        if not local_result.success:
            print(f"  Warning: L-BFGS-B at alpha={alpha} did not converge: {local_result.message}")
        
        current_generators = local_result.x
        current_obj_val = _objective_function_normalized_with_penalty(current_generators)
        print(f"  Current normalized objective after L-BFGS-B step: {current_obj_val:.8f}")
        
        if current_obj_val < best_overall_obj_value:
            best_overall_obj_value = current_obj_val
            best_overall_generators = current_generators
            print(f"  New best overall objective found after L-BFGS-B: {best_overall_obj_value:.8f}")

    print(f"\nL-BFGS-B phase complete. Best objective: {best_overall_obj_value:.8f}")

    # --- Phase 3: Final Polish with Powell's method (Normalized Metric + Penalty) ---
    print("Starting final polish with Powell's method (normalized metric)...")
    final_polish_result = minimize(
        fun=_objective_function_normalized_with_penalty, # Using the general normalized objective + penalty
        x0=best_overall_generators, # Start from the best found so far
        method='Powell',
        bounds=GENERATOR_BOUNDS,
        options={'maxiter': 2500, 'ftol': 1e-13, 'disp': False} # Maxiter reverted to 2500 for balance
    )

    if not final_polish_result.success:
        print(f"Warning: Final refinement (Powell) did not converge: {final_polish_result.message}")
    
    final_obj_val = final_polish_result.fun
    print(f"Powell phase complete. Final objective: {final_obj_val:.8f}")

    if final_obj_val < best_overall_obj_value:
        best_overall_obj_value = final_obj_val
        best_overall_generators = final_polish_result.x
        print(f"New best overall objective found after Powell: {best_overall_obj_value:.8f}")
    else:
        print("Powell's method did not improve upon previous stages. Retaining best from prior stages.")

    # Unpack the best found generators into the full 13-point configuration
    optimal_points = _unpack_symmetric_points(best_overall_generators)

    # --- Final Normalization to Unit-Area Convex Hull for the OUTPUT points ---
    # The optimization functions already work with normalized area in their objectives,
    # but the final returned points may not have a unit convex hull area themselves.
    # This step ensures the output points are scaled correctly (from Inspirations 2 & 3).

    # 1. Center the points around their geometric mean (centroid)
    centered_points = optimal_points - np.mean(optimal_points, axis=0)

    # 2. Compute the convex hull and its area
    try:
        # Ensure there are at least 3 distinct points for a valid convex hull
        if len(np.unique(centered_points, axis=0)) < 3:
            # Fallback for highly degenerate cases, though unlikely with strong penalties
            print("Warning: Fewer than 3 unique points after centering, cannot compute ConvexHull for final scaling.")
            return optimal_points # Return unscaled points as a fallback

        # qhull_options='QJ' can help with precision issues by joggling the input
        hull = ConvexHull(centered_points, qhull_options='QJ')
        hull_area = hull.area
    except Exception as e:
        print(f"Error computing convex hull for final normalization: {e}. Returning unscaled points.")
        return optimal_points # Return unscaled points as a fallback

    # 3. Scale the points to normalize the convex hull area to 1
    if hull_area > 1e-9: # Avoid division by near-zero area
        scale_factor = 1.0 / np.sqrt(hull_area)
        normalized_points = centered_points * scale_factor
    else:
        print("Warning: Convex hull area too small for final normalization. Returning unscaled points.")
        normalized_points = centered_points

    return normalized_points

# EVOLVE-BLOCK-END