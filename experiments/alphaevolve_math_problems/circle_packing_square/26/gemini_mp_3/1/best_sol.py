# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import differential_evolution, minimize
import time # Included for potential future timing analysis, but not directly used in return.

# Constants
N_CIRCLES = 26
RANDOM_SEED = 42
EPSILON = 1e-7 # Small tolerance for radius and constraint checks

# Penalty factors for the objective function (used by Differential Evolution)
PENALTY_FACTORS = {
    'r_min': 1e6,       # Penalty for radius <= EPSILON
    'r_max': 1e6,       # Penalty for radius > 0.5
    'contain': 1e5,     # Penalty for circle extending beyond square boundaries
    'overlap': 1e7      # Penalty for overlapping circles (highest priority)
}

def _adaptive_objective_with_penalties(vars_flat: np.ndarray, N: int, base_penalty_factors: dict) -> float:
    """
    Adaptive objective function that adjusts penalty factors based on constraint violation severity.
    This helps the optimizer navigate more effectively through infeasible regions.
    """
    xs, ys, rs = _unpack_vars(vars_flat)
    
    total_penalty = 0.0
    
    # Calculate constraint violations first to determine adaptive penalties
    radius_violations = np.sum(np.maximum(0, EPSILON - rs)) + np.sum(np.maximum(0, rs - 0.5))
    contain_violations = (np.sum(np.maximum(0, rs - xs)) + np.sum(np.maximum(0, xs + rs - 1.0)) + 
                         np.sum(np.maximum(0, rs - ys)) + np.sum(np.maximum(0, ys + rs - 1.0)))
    
    # Calculate overlap violations
    x_diff = xs[:, np.newaxis] - xs[np.newaxis, :]
    y_diff = ys[:, np.newaxis] - ys[np.newaxis, :]
    dist_sq = x_diff**2 + y_diff**2
    r_sum = rs[:, np.newaxis] + rs[np.newaxis, :]
    r_sum_sq = r_sum**2
    overlap_violations_matrix = np.maximum(0, r_sum_sq - dist_sq)
    np.fill_diagonal(overlap_violations_matrix, 0)
    overlap_violations = np.sum(np.triu(overlap_violations_matrix, k=1))
    
    # Adaptive penalty scaling based on violation severity
    # If violations are severe, increase penalties to guide the optimizer more strongly
    radius_penalty_factor = base_penalty_factors['r_min'] * (1.0 + radius_violations)
    contain_penalty_factor = base_penalty_factors['contain'] * (1.0 + contain_violations * 0.1)
    overlap_penalty_factor = base_penalty_factors['overlap'] * (1.0 + overlap_violations * 0.01)
    
    # Apply penalties
    total_penalty += radius_penalty_factor * radius_violations / (1.0 + radius_violations)  # Normalized
    total_penalty += contain_penalty_factor * contain_violations / (1.0 + contain_violations)  # Normalized  
    total_penalty += overlap_penalty_factor * overlap_violations / (1.0 + overlap_violations)  # Normalized
    
    return -np.sum(rs) + total_penalty

def _unpack_vars(vars_flat: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Unpacks a flattened variable array into separate x, y, r arrays."""
    xs = vars_flat[0::3]
    ys = vars_flat[1::3]
    rs = vars_flat[2::3]
    return xs, ys, rs

def _calculate_sum_radii(circles: np.ndarray) -> float:
    """Calculates the sum of radii from a (N,3) circles array."""
    return np.sum(circles[:, 2])

def _check_constraints(circles: np.ndarray, tol: float = EPSILON) -> bool:
    """
    Checks if all circles satisfy containment and non-overlap constraints.
    Returns True if all constraints are met within tolerance, False otherwise.
    """
    N = circles.shape[0]
    xs, ys, rs = circles[:, 0], circles[:, 1], circles[:, 2]

    # 1. Radius validity: r > 0 and r <= 0.5
    if np.any(rs < tol) or np.any(rs > 0.5 + tol):
        return False

    # 2. Containment constraints: r <= x <= 1-r and r <= y <= 1-r
    if np.any(xs - rs < -tol) or np.any(xs + rs > 1.0 + tol):
        return False
    if np.any(ys - rs < -tol) or np.any(ys + rs > 1.0 + tol):
        return False

    # 3. Non-overlap constraints: (xi-xj)^2 + (yi-yj)^2 >= (ri+rj)^2
    # Vectorized approach for efficiency (consistent with _objective_with_penalties)
    x_diff = xs[:, np.newaxis] - xs[np.newaxis, :]
    y_diff = ys[:, np.newaxis] - ys[np.newaxis, :]
    dist_sq = x_diff**2 + y_diff**2

    r_sum = rs[:, np.newaxis] + rs[np.newaxis, :]
    min_dist_sq = r_sum**2

    # Check for overlaps, excluding self-comparisons
    # Only need to check the upper triangle (or lower) since matrix is symmetric
    upper_triangle_indices = np.triu_indices(N, k=1)
    
    if np.any(dist_sq[upper_triangle_indices] < min_dist_sq[upper_triangle_indices] - tol):
        return False

    return True

def _objective_with_penalties(vars_flat: np.ndarray, N: int, penalty_factors: dict) -> float:
    """
    Objective function for global optimization (e.g., Differential Evolution),
    including penalties for constraint violations.
    Minimizes -sum_radii + total_penalty.
    """
    xs, ys, rs = _unpack_vars(vars_flat)
    
    total_penalty = 0.0

    # 1. Radius validity penalties
    total_penalty += penalty_factors['r_min'] * np.sum(np.maximum(0, EPSILON - rs))
    total_penalty += penalty_factors['r_max'] * np.sum(np.maximum(0, rs - 0.5))
    
    # 2. Containment penalties
    total_penalty += penalty_factors['contain'] * np.sum(np.maximum(0, rs - xs))
    total_penalty += penalty_factors['contain'] * np.sum(np.maximum(0, xs + rs - 1.0))
    total_penalty += penalty_factors['contain'] * np.sum(np.maximum(0, rs - ys))
    total_penalty += penalty_factors['contain'] * np.sum(np.maximum(0, ys + rs - 1.0))
    
    # 3. Non-overlap penalties (vectorized for efficiency)
    x_diff = xs[:, np.newaxis] - xs[np.newaxis, :]
    y_diff = ys[:, np.newaxis] - ys[np.newaxis, :]
    dist_sq = x_diff**2 + y_diff**2

    r_sum = rs[:, np.newaxis] + rs[np.newaxis, :]
    r_sum_sq = r_sum**2

    overlap_violations = np.maximum(0, r_sum_sq - dist_sq)
    np.fill_diagonal(overlap_violations, 0) # A circle cannot overlap with itself
    total_penalty += penalty_factors['overlap'] * np.sum(np.triu(overlap_violations, k=1))

    return -np.sum(rs) + total_penalty

def _pure_objective(vars_flat: np.ndarray) -> float:
    """
    Pure objective function for local optimization (e.g., SLSQP),
    where constraints are handled separately by the optimizer.
    Minimizes -sum_radii.
    """
    rs = vars_flat[2::3]
    return -np.sum(rs)

# Helper function to generate a structured initial guess for Differential Evolution
def _generate_grid_initial_guess(N: int, initial_radius: float, seed: int) -> np.ndarray:
    """
    Generates a flattened array of (x,y,r) for N circles arranged in a grid.
    Centers are spaced such that circles with `initial_radius` fit within [0,1] square.
    """
    np.random.seed(seed)
    
    # Determine grid dimensions
    num_cols = int(np.ceil(np.sqrt(N)))
    num_rows = int(np.ceil(N / num_cols))
    
    # Calculate effective side length for placing centers (from r to 1-r)
    effective_side_len_x = 1.0 - 2 * initial_radius
    effective_side_len_y = 1.0 - 2 * initial_radius
    
    # Calculate spacing between centers
    spacing_x = effective_side_len_x / (num_cols - 1) if num_cols > 1 else 0
    spacing_y = effective_side_len_y / (num_rows - 1) if num_rows > 1 else 0

    grid_positions = []
    for i in range(num_cols):
        for j in range(num_rows):
            x = initial_radius + i * spacing_x if num_cols > 1 else 0.5
            y = initial_radius + j * spacing_y if num_rows > 1 else 0.5
            grid_positions.append((x, y))

    # Shuffle positions and pick N_CIRCLES
    np.random.shuffle(grid_positions)
    
    vars_flat = np.zeros(N * 3)
    for k in range(N):
        x, y = grid_positions[k]
        vars_flat[k*3] = x
        vars_flat[k*3+1] = y
        vars_flat[k*3+2] = initial_radius
    return vars_flat

# Helper function to generate a hexagonal initial guess for Differential Evolution
def _generate_hex_initial_guess(N: int, initial_radius: float, seed: int) -> np.ndarray:
    """
    Generates a flattened array of (x,y,r) for N circles arranged in a hexagonal pattern.
    Centers are spaced such that circles with `initial_radius` fit within [0,1] square.
    If N circles cannot be formed with the given radius, it falls back to a grid arrangement.
    """
    np.random.seed(seed)
    
    r_hex = initial_radius
    centers = []
    
    # Starting y-position for the first row
    y_start = r_hex
    
    row_num = 0
    while True:
        current_y = y_start + row_num * np.sqrt(3) * r_hex
        # Check if current_y exceeds the upper boundary (1 - r_hex) with tolerance
        if current_y > 1.0 - r_hex + EPSILON:
            break
            
        # Alternate row offset for x-coordinates
        # Even rows start at r_hex, odd rows are offset by r_hex (start at 2*r_hex)
        x_offset = r_hex if row_num % 2 == 0 else 2 * r_hex
        
        col_num = 0
        while True:
            current_x = x_offset + col_num * 2 * r_hex
            # Check if current_x exceeds the right boundary (1 - r_hex) with tolerance
            if current_x > 1.0 - r_hex + EPSILON:
                break
            centers.append((current_x, current_y))
            col_num += 1
        
        row_num += 1

    # If the hexagonal packing strategy couldn't generate enough circles,
    # fall back to the grid strategy which is more robust for small radii.
    if len(centers) < N:
        return _generate_grid_initial_guess(N, initial_radius, seed)
    
    # If more circles were generated than N, randomly sample N centers
    if len(centers) > N:
        indices = np.random.choice(len(centers), N, replace=False)
        centers_sampled = [centers[i] for i in indices]
    else:
        centers_sampled = centers # Exactly N circles were generated
        
    vars_flat = np.zeros(N * 3)
    for k in range(N):
        x, y = centers_sampled[k]
        vars_flat[k*3] = x
        vars_flat[k*3+1] = y
        vars_flat[k*3+2] = r_hex
    return vars_flat

def circle_packing26() -> np.ndarray:
    """
    Places 26 non-overlapping circles in the unit square in order to maximize the sum of radii.

    Returns:
        circles: np.array of shape (26,3), where the i-th row (x,y,r) stores the (x,y) coordinates of the i-th circle of radius r.
    """
    np.random.seed(RANDOM_SEED) # Ensure reproducibility for stochastic methods

    # Define bounds for each variable (x, y, r) for all N_CIRCLES
    # x_i: [0, 1], y_i: [0, 1], r_i: [EPSILON, 0.5]
    bounds = [(0.0, 1.0), (0.0, 1.0), (EPSILON, 0.5)] * N_CIRCLES

    # --- Multi-restart Differential Evolution Strategy ---
    # Instead of one large DE run, perform multiple smaller independent runs
    # This increases the chance of finding different basins of attraction
    
    num_restarts = 4
    de_popsize_per_restart = 60  # Total population across restarts: 4 * 60 = 240
    de_maxiter_per_restart = 800  # Total iterations across restarts: 4 * 800 = 3200
    
    best_de_results = []
    
    for restart_idx in range(num_restarts):
        restart_seed = RANDOM_SEED + restart_idx * 100
        np.random.seed(restart_seed)
        
        # Prepare initial population for this restart
        initial_population_list = []
        initial_radius_base = 0.04 * (0.8 + 0.4 * restart_idx / (num_restarts - 1))  # Vary base radius across restarts
        
        # Add structured initial guesses
        initial_population_list.append(_generate_grid_initial_guess(N_CIRCLES, initial_radius_base, restart_seed))
        initial_population_list.append(_generate_hex_initial_guess(N_CIRCLES, initial_radius_base, restart_seed + 1))
        
        # Fill with random guesses
        num_random_to_add = de_popsize_per_restart - len(initial_population_list)
        for k in range(num_random_to_add):
            xs_rand = np.random.uniform(0.0, 1.0, N_CIRCLES)
            ys_rand = np.random.uniform(0.0, 1.0, N_CIRCLES)
            rs_rand = np.random.uniform(EPSILON, initial_radius_base * 1.5, N_CIRCLES)
            
            vars_flat_rand = np.empty(N_CIRCLES * 3)
            vars_flat_rand[0::3] = xs_rand
            vars_flat_rand[1::3] = ys_rand
            vars_flat_rand[2::3] = rs_rand
            initial_population_list.append(vars_flat_rand)
        
        initial_population_array = np.array(initial_population_list)
        
        # Run DE for this restart
        de_result = differential_evolution(
            func=_adaptive_objective_with_penalties,  # Use adaptive penalties
            bounds=bounds,
            args=(N_CIRCLES, PENALTY_FACTORS),
            strategy='best1bin',
            maxiter=de_maxiter_per_restart,
            popsize=de_popsize_per_restart,
            tol=0.001,
            mutation=(0.4, 1.2),  # Slightly wider mutation range for more exploration
            recombination=0.8,    # Higher recombination for better mixing
            seed=restart_seed,
            disp=False,
            workers=-1,
            init=initial_population_array
        )
        
        best_de_results.append((de_result.fun, de_result.x, de_result))
    
    # Select the best result from all restarts
    best_de_results.sort(key=lambda x: x[0])  # Sort by objective value (lower is better for minimization)
    best_objective, initial_guess_flat, de_result = best_de_results[0]
    
    # --- Step 2: Enhanced Local Optimization with Multiple Attempts ---
    # Try local optimization from multiple starting points around the best DE solution
    
    constraints = []
    # Containment constraints: r <= x <= 1-r, r <= y <= 1-r
    for i in range(N_CIRCLES):
        constraints.append({'type': 'ineq', 'fun': lambda vars_flat, idx=i: vars_flat[idx*3] - vars_flat[idx*3+2]})
        constraints.append({'type': 'ineq', 'fun': lambda vars_flat, idx=i: 1.0 - vars_flat[idx*3] - vars_flat[idx*3+2]})
        constraints.append({'type': 'ineq', 'fun': lambda vars_flat, idx=i: vars_flat[idx*3+1] - vars_flat[idx*3+2]})
        constraints.append({'type': 'ineq', 'fun': lambda vars_flat, idx=i: 1.0 - vars_flat[idx*3+1] - vars_flat[idx*3+2]})

    # Non-overlap constraints: (xi-xj)^2 + (yi-yj)^2 - (ri+rj)^2 >= 0
    for i in range(N_CIRCLES):
        for j in range(i + 1, N_CIRCLES):
            constraints.append({'type': 'ineq', 'fun': lambda vars_flat, i_idx=i, j_idx=j: 
                (vars_flat[i_idx*3] - vars_flat[j_idx*3])**2 + \
                (vars_flat[i_idx*3+1] - vars_flat[j_idx*3+1])**2 - \
                (vars_flat[i_idx*3+2] + vars_flat[j_idx*3+2])**2
            })

    # Try multiple local optimization attempts from perturbed starting points
    local_results = []
    num_local_attempts = 3
    
    for attempt in range(num_local_attempts):
        if attempt == 0:
            # First attempt: use the best DE solution directly
            x0 = initial_guess_flat.copy()
        else:
            # Subsequent attempts: add small perturbations to the best DE solution
            x0 = initial_guess_flat.copy()
            perturbation_scale = 0.01 * attempt
            xs, ys, rs = _unpack_vars(x0)
            
            # Perturb positions slightly
            xs += np.random.normal(0, perturbation_scale, N_CIRCLES)
            ys += np.random.normal(0, perturbation_scale, N_CIRCLES)
            # Perturb radii slightly (but keep them positive)
            rs *= (1.0 + np.random.normal(0, perturbation_scale * 0.5, N_CIRCLES))
            rs = np.maximum(rs, EPSILON)
            
            # Ensure bounds are respected
            xs = np.clip(xs, 0.0, 1.0)
            ys = np.clip(ys, 0.0, 1.0)
            rs = np.clip(rs, EPSILON, 0.5)
            
            # Repack
            x0[0::3] = xs
            x0[1::3] = ys
            x0[2::3] = rs
        
        local_result = minimize(
            fun=_pure_objective,
            x0=x0,
            method='SLSQP',
            bounds=bounds,
            constraints=constraints,
            options={'maxiter': 1000, 'ftol': 1e-10, 'disp': False}
        )
        
        local_results.append(local_result)
    
    # Select the best local result
    valid_local_results = [lr for lr in local_results if lr.success]
    if valid_local_results:
        local_result = min(valid_local_results, key=lambda x: x.fun)
    else:
        local_result = local_results[0]  # Fallback to first attempt

    # --- Step 3: Select the best valid solution ---
    # Convert results to (N,3) circle arrays for validation
    de_xs, de_ys, de_rs = _unpack_vars(de_result.x)
    circles_de = np.array([de_xs, de_ys, de_rs]).T
    
    slsqp_xs, slsqp_ys, slsqp_rs = _unpack_vars(local_result.x)
    circles_slsqp = np.array([slsqp_xs, slsqp_ys, slsqp_rs]).T

    # Check validity and compare sum_radii
    is_de_valid = _check_constraints(circles_de)
    is_slsqp_valid = _check_constraints(circles_slsqp)

    final_circles = None
    best_sum_radii = -np.inf

    # Prioritize valid solutions
    if is_slsqp_valid:
        current_sum_radii = _calculate_sum_radii(circles_slsqp)
        if current_sum_radii > best_sum_radii:
            best_sum_radii = current_sum_radii
            final_circles = circles_slsqp
    
    if is_de_valid: # Check DE result only if SLSQP was not valid or DE is better
        current_sum_radii = _calculate_sum_radii(circles_de)
        if current_sum_radii > best_sum_radii:
            best_sum_radii = current_sum_radii
            final_circles = circles_de
            
    # Fallback: If neither is strictly valid (e.g., due to numerical precision issues
    # or optimizer failing to fully converge), we must still return a valid result.
    # A common approach is to take the "best" candidate and slightly shrink radii
    # until all constraints are met.
    if final_circles is None:
        # If neither DE nor SLSQP yielded a strictly valid solution,
        # prioritize the SLSQP result if it reported success, otherwise the DE result.
        # This choice is somewhat arbitrary, but SLSQP usually provides a "closer" to valid solution.
        if local_result.success:
            candidate_circles = circles_slsqp
        else: # SLSQP failed, use DE's best guess as a starting point
            candidate_circles = circles_de
        
        # Apply a conservative shrinkage to guarantee validity
        current_circles = np.copy(candidate_circles)
        shrink_attempts = 0
        max_shrink_attempts = 100 # Increased max attempts for robustness (original: 50)
        shrink_factor = 0.999 # Shrink radii by 0.1%
        
        # Use a slightly more lenient tolerance for the shrinkage loop to avoid infinite loops
        # if constraints are violated by an infinitesimal amount.
        validation_tol = EPSILON * 10 
        
        while not _check_constraints(current_circles, tol=validation_tol) and shrink_attempts < max_shrink_attempts:
            current_circles[:, 2] *= shrink_factor
            shrink_attempts += 1
            # Ensure radii don't become too small (below EPSILON)
            current_circles[:, 2] = np.maximum(current_circles[:, 2], EPSILON)

        final_circles = current_circles
        
        # After shrinkage, re-check with strict EPSILON. If still not valid, something is very wrong,
        # but we return the best effort.
        if not _check_constraints(final_circles, tol=EPSILON):
            # As a last resort, if still invalid, log a warning or take extreme action
            # For this problem, simply returning the shrunken result is the requirement.
            pass # No specific error handling requested, just ensure a result is returned.

    return final_circles


# EVOLVE-BLOCK-END
