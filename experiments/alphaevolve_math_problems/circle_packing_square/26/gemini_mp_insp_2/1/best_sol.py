# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import dual_annealing, minimize, NonlinearConstraint
import time
import math
from numba import njit # NEW IMPORT: Numba for performance optimization

# Constants
N_CIRCLES = 26
EPSILON = 1e-7 # Small value to ensure strict inequalities for constraints and numerical stability

# --- Helper Functions for Objective and Constraints (Numba-optimized) ---

@njit(cache=True)
def _objective_numba(params: np.ndarray) -> float:
    """Objective function: negative sum of radii (to be minimized), Numba optimized."""
    r = params[2::3]
    return -np.sum(r)

@njit(cache=True)
def _all_constraints_func_numba(params: np.ndarray) -> np.ndarray:
    """
    Combines all non-linear constraints into a single array (Numba optimized).
    All returned values must be >= 0 for scipy.optimize.NonlinearConstraint.
    This function replaces the original _all_constraints_func by inlining _unpack_params
    and optimizing the overlap constraint loop with Numba.
    """
    # Unpack parameters directly in Numba-jitted function
    x = params[0::3]
    y = params[1::3]
    r = params[2::3]
    
    # Containment constraints: ri <= xi <= 1-ri and ri <= yi <= 1-ri
    containment_constraints = np.concatenate((
        x - r,
        1 - x - r,
        y - r,
        1 - y - r
    ))

    # Non-overlap constraints: (xi-xj)² + (yi-yj)² - (ri + rj)² >= 0
    num_circles = len(x)
    # Pre-allocate array for overlap constraints for Numba efficiency
    overlap_constraints_values = np.empty(num_circles * (num_circles - 1) // 2, dtype=params.dtype)
    k = 0
    for i in range(num_circles):
        for j in range(i + 1, num_circles):
            dist_sq = (x[i] - x[j])**2 + (y[i] - y[j])**2
            radii_sum_sq = (r[i] + r[j])**2
            overlap_constraints_values[k] = dist_sq - radii_sum_sq
            k += 1
            
    return np.concatenate((containment_constraints, overlap_constraints_values))

def _get_bounds(n: int) -> list[tuple[float, float]]:
    """Generates bounds for x, y, r for all N circles."""
    bounds = []
    for _ in range(n):
        bounds.append((0.0, 1.0)) # x_i coordinates
        bounds.append((0.0, 1.0)) # y_i coordinates
        bounds.append((EPSILON, 0.5)) # r_i radius (must be > 0 and <= 0.5)
    return bounds

def _generate_initial_guess(n: int, method: str = 'grid_and_jiggle') -> np.ndarray:
    """Generates an initial guess for the circle parameters."""
    if method == 'random':
        # Random positions, small random radii
        initial_params = np.random.rand(n * 3)
        initial_params[2::3] = np.random.uniform(0.01, 0.05, n) # Small initial radii
    elif method == 'grid_and_jiggle':
        # Attempt a grid-like initial placement for better starting point
        # This helps in avoiding all circles collapsing to 0 radius or overlapping heavily
        side_len = math.ceil(math.sqrt(n))
        
        # Calculate spacing such that circles can potentially grow
        # Using 1.0 / (side_len + 1) creates a margin from the edges
        spacing = 1.0 / (side_len + 1)
        
        centers = []
        for row in range(side_len):
            for col in range(side_len):
                if len(centers) < n: # Ensure we don't create more than N centers
                    # Center in grid cell, slightly offset from 0,0 and 1,1
                    x_center = (col + 0.5) * spacing
                    y_center = (row + 0.5) * spacing
                    centers.append((x_center, y_center))
        
        initial_params = np.zeros(n * 3)
        for i in range(n):
            # Apply a small random jiggle to avoid perfect symmetry traps
            initial_params[i*3] = centers[i][0] + np.random.uniform(-0.02, 0.02)
            initial_params[i*3+1] = centers[i][1] + np.random.uniform(-0.02, 0.02)
            initial_params[i*3+2] = 0.01 # Start with a small, uniform radius
        
        # Clip positions to ensure they are within the square, considering radius bounds
        min_pos_val = EPSILON + 0.01 # Smallest possible x/y given r=0.01
        max_pos_val = 1.0 - (EPSILON + 0.01) # Largest possible x/y given r=0.01
        initial_params[0::3] = np.clip(initial_params[0::3], min_pos_val, max_pos_val)
        initial_params[1::3] = np.clip(initial_params[1::3], min_pos_val, max_pos_val)
        
        # Ensure radii are within their defined bounds
        initial_params[2::3] = np.clip(initial_params[2::3], EPSILON, 0.5)

    return initial_params

def _validate_packing(circles: np.ndarray, epsilon: float = 1e-6) -> tuple[bool, str]:
    """
    Validates a packing of circles for containment and non-overlap.
    circles: np.array of shape (N,3) with (x,y,r) for each circle.
    epsilon: Tolerance for constraint violations.
    """
    is_valid = True
    error_msg = []
    
    n = circles.shape[0]
    x = circles[:, 0]
    y = circles[:, 1]
    r = circles[:, 2]

    # 1. Positive Radii
    if np.any(r < 0 - epsilon):
        is_valid = False
        error_msg.append(f"Negative radius found: {r[r < 0 - epsilon]}")
        # Clip negative radii to 0 for further checks, but mark as invalid
        r = np.clip(r, 0, None)

    # 2. Containment: r <= x <= 1-r and r <= y <= 1-r
    for i in range(n):
        if r[i] - x[i] > epsilon:
            is_valid = False
            error_msg.append(f"Circle {i} (x={x[i]:.4f}, r={r[i]:.4f}) violates x_min containment (r > x by {r[i]-x[i]:.4f})")
        if x[i] + r[i] - 1 > epsilon:
            is_valid = False
            error_msg.append(f"Circle {i} (x={x[i]:.4f}, r={r[i]:.4f}) violates x_max containment (x+r > 1 by {x[i]+r[i]-1:.4f})")
        if r[i] - y[i] > epsilon:
            is_valid = False
            error_msg.append(f"Circle {i} (y={y[i]:.4f}, r={r[i]:.4f}) violates y_min containment (r > y by {r[i]-y[i]:.4f})")
        if y[i] + r[i] - 1 > epsilon:
            is_valid = False
            error_msg.append(f"Circle {i} (y={y[i]:.4f}, r={r[i]:.4f}) violates y_max containment (y+r > 1 by {y[i]+r[i]-1:.4f})")

    # 3. Non-overlap: (xi-xj)² + (yi-yj)² >= (ri + rj)²
    for i in range(n):
        for j in range(i + 1, n):
            dist_sq = (x[i] - x[j])**2 + (y[i] - y[j])**2
            radii_sum_sq = (r[i] + r[j])**2
            if dist_sq < radii_sum_sq - epsilon:
                is_valid = False
                error_msg.append(f"Circles {i} (r={r[i]:.4f}) and {j} (r={r[j]:.4f}) overlap: dist_sq={dist_sq:.6f} < radii_sum_sq={radii_sum_sq:.6f} (violation by {radii_sum_sq - dist_sq:.6f})")

    return is_valid, "\n".join(error_msg)


def circle_packing26() -> np.ndarray:
    """
    Places 26 non-overlapping circles in the unit square in order to maximize the sum of radii.

    Returns:
        circles: np.array of shape (26,3), where the i-th row (x,y,r) stores the (x,y) coordinates of the i-th circle of radius r.
    """
    np.random.seed(42) # For reproducibility of stochastic elements
    
    n = N_CIRCLES
    start_time = time.time()
    max_time = 170  # Reserve 10 seconds for validation and final processing
    
    # 1. Define bounds for x, y, and r for all circles
    bounds = _get_bounds(n)
    
    # 2. Define non-linear constraints
    num_containment_constraints = 4 * n
    num_overlap_constraints = n * (n - 1) // 2
    
    nlc = NonlinearConstraint(
        _all_constraints_func_numba, # Use Numba-optimized constraint function
        lb=np.zeros(num_containment_constraints + num_overlap_constraints),
        ub=np.full(num_containment_constraints + num_overlap_constraints, np.inf)
    )

    # 3. Multi-stage hybrid optimization approach
    best_result = None
    best_sum_radii = -np.inf
    
    # Stage 1: Fast global exploration with dual_annealing (optimized parameters)
    initial_params = _generate_initial_guess(n, method='grid_and_jiggle')
    
    # Use faster local optimizer for dual_annealing phase
    fast_local_options = {
        'method': 'SLSQP',  # Faster than trust-constr for initial exploration
        'constraints': nlc,
        'bounds': bounds,
        'options': {
            'maxiter': 700,    # Increased local iterations given Numba speedup
            'ftol': 1e-6,      # Tighter tolerance for speed
        }
    }

    # Execute dual_annealing with adjusted parameters for better global search within time limits
    # Given Numba speedup, we can afford more global iterations.
    if time.time() - start_time < max_time * 0.7:  # Use 70% of time for global search
        try:
            result1 = dual_annealing(
                func=_objective_numba, # Use Numba-optimized objective function
                bounds=bounds,
                minimizer_kwargs=fast_local_options,
                x0=initial_params,
                maxiter=1500,        # Increased maxiter for more thorough global search
                initial_temp=5000.0, # Moderate initial temperature for broad exploration
                seed=42,
                maxfun=100000,       # Increased function evaluations limit
            )
            
            if result1.success:
                sum_radii1 = -result1.fun
                if sum_radii1 > best_sum_radii:
                    best_result = result1
                    best_sum_radii = sum_radii1
        except Exception as e:
            print(f"Stage 1 optimization failed: {e}")
    
    # Stage 2: Local refinement with high precision (if time permits)
    if best_result is not None and time.time() - start_time < max_time * 0.9:
        try:
            # Use the best result from stage 1 as starting point for precise local optimization
            precise_local_options = {
                'method': 'trust-constr',
                'constraints': nlc,
                'bounds': bounds,
                'options': {
                    'verbose': 0,
                    'maxiter': 1500,     # Increased local iterations for precision
                    'gtol': 1e-8,        # Tighter tolerance for final precision
                    'xtol': 1e-8,
                    'barrier_tol': 1e-8
                }
            }
            
            result2 = minimize(
                fun=_objective_numba, # Use Numba-optimized objective function
                x0=best_result.x,
                **precise_local_options
            )
            
            if result2.success:
                sum_radii2 = -result2.fun
                if sum_radii2 > best_sum_radii:
                    best_result = result2
                    best_sum_radii = sum_radii2
        except Exception as e:
            print(f"Stage 2 refinement failed: {e}")
    
    # Stage 3: Fallback - try alternative starting points if time permits
    # Reduced maxiter for fallback to be faster, considering it's a last resort
    if time.time() - start_time < max_time * 0.95:
        for seed_offset in [1, 2, 3, 4]: # More fallback attempts if time permits
            try:
                np.random.seed(42 + seed_offset)
                alt_initial = _generate_initial_guess(n, method='random')
                
                alt_result = minimize(
                    fun=_objective_numba, # Use Numba-optimized objective function
                    x0=alt_initial,
                    method='SLSQP',
                    bounds=bounds,
                    constraints=nlc,
                    options={'maxiter': 200, 'ftol': 1e-6} # Further reduced maxiter, tighter ftol
                )
                
                if alt_result.success:
                    alt_sum_radii = -alt_result.fun
                    if alt_sum_radii > best_sum_radii:
                        best_result = alt_result
                        best_sum_radii = alt_sum_radii
                        
                if time.time() - start_time > max_time * 0.95:
                    break
            except Exception:
                continue
    
    # Use best result found, or fallback to initial guess
    if best_result is not None:
        optimized_params = best_result.x
    else:
        print("WARNING: All optimization stages failed, using initial guess")
        optimized_params = initial_params
    
    final_circles = optimized_params.reshape((n, 3))

    # 6. Validate the final packing
    is_valid, validation_msg = _validate_packing(final_circles, epsilon=EPSILON)
    if not is_valid:
        print(f"WARNING: Final circle packing failed validation with tolerance {EPSILON}:\n{validation_msg}")
    
    # Ensure radii are not negative due to numerical issues
    final_circles[:, 2] = np.maximum(final_circles[:, 2], 0)

    return final_circles


# EVOLVE-BLOCK-END
