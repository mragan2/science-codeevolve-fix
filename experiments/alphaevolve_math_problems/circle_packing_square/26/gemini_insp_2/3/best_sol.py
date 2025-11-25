# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize, differential_evolution, NonlinearConstraint
import time
from numba import njit # Added Numba for JIT compilation

# Constants for the problem (tuned based on inspiration programs)
N_CIRCLES = 26
UNIT_SQUARE_SIDE = 1.0
GLOBAL_RANDOM_SEED = 42 # For reproducibility

# Optimization Parameters
# Using separate penalty factors inspired by IP1/IP2
PENALTY_FACTOR_DE_OVERLAP = 3500.0 # From IP2/IP3, increased for stronger enforcement
PENALTY_FACTOR_DE_BOUNDARY = 3500.0 # From IP2/IP3, increased for stronger enforcement
EPSILON = 1e-9           # Smaller epsilon for higher numerical precision, used for min radius

DE_MAXITER = 6000        # Increased max iterations for Differential Evolution (from IP3)
DE_POPSIZE = 80          # Increased population size for Differential Evolution (from IP3)
DE_WORKERS = -1          # Use all available CPU cores for DE (from IP1/IP2/IP3)

# SLSQP parameters for initial refinement
SLSQP_INITIAL_MAXITER = 5000 # Increased maxiter (from IP1/IP3)
SLSQP_INITIAL_FTOL = 1e-12   # Tightened ftol (from IP2/IP3)
SLSQP_INITIAL_GTOL = 1e-9    # Tightened gtol (from IP2/IP3)

# ILS parameters
NUM_ILS_ITERATIONS = 80  # Increased number of perturbation-refinement cycles for ILS (from IP3)
NOISE_SCALE_XY = 0.015   # Perturbation for coordinates in ILS (from IP2/IP3)
NOISE_SCALE_R = 0.0075   # Perturbation for radii in ILS (from IP2/IP3)
SLSQP_ILS_MAXITER = 2500 # Maxiter for ILS internal SLSQP (from IP3)
SLSQP_ILS_FTOL = 1e-11   # ftol for ILS internal SLSQP (from IP3)
SLSQP_ILS_GTOL = 1e-8    # gtol for ILS internal SLSQP (from IP3)

# --- Numba-accelerated Helper Functions --- (from IP2/IP3)
@njit(cache=True)
def _reshape_params_numba(params_flat: np.ndarray, num_c: int) -> np.ndarray:
    """Reshapes 1D parameter array into (num_c, 3) circle array."""
    return params_flat.reshape((num_c, 3))

@njit(cache=True, fastmath=True)
def _calculate_penalties_de_numba(params_flat, n_circles, epsilon, unit_square_side):
    """
    Numba-optimized function to calculate penalty for boundary violations and overlaps.
    Returns sum of squared violations for boundary and overlap separately.
    (Adapted from IP2/IP3 to return separate penalty sums).
    """
    circles = _reshape_params_numba(params_flat, n_circles)
    centers = circles[:, :2]
    radii = circles[:, 2]

    # Ensure radii are positive for calculations, as negative radii are not physically meaningful
    radii_positive = np.maximum(radii, epsilon)
    
    boundary_penalty_sum_sq = 0.0
    for i in range(n_circles):
        # Boundary Containment Penalties (squared magnitude of violation)
        boundary_penalty_sum_sq += np.maximum(0.0, radii_positive[i] - centers[i, 0])**2
        boundary_penalty_sum_sq += np.maximum(0.0, radii_positive[i] - (unit_square_side - centers[i, 0]))**2
        boundary_penalty_sum_sq += np.maximum(0.0, radii_positive[i] - centers[i, 1])**2
        boundary_penalty_sum_sq += np.maximum(0.0, radii_positive[i] - (unit_square_side - centers[i, 1]))**2

    overlap_penalty_sum_sq = 0.0
    for i in range(n_circles):
        for j in range(i + 1, n_circles):
            # Overlap Penalties (squared magnitude of violation)
            dx = centers[i, 0] - centers[j, 0]
            dy = centers[i, 1] - centers[j, 1]
            dist_sq = dx*dx + dy*dy
            sum_radii_sq = (radii_positive[i] + radii_positive[j])**2
            violation_magnitude = sum_radii_sq - dist_sq
            overlap_penalty_sum_sq += np.maximum(0.0, violation_magnitude)**2
            
    return boundary_penalty_sum_sq, overlap_penalty_sum_sq

@njit(cache=True, fastmath=True)
def _slsqp_constraints_func_numba(params_flat, n_circles, unit_square_side):
    """
    Numba-optimized function to compute SLSQP constraints.
    Returns a vector where each element must be >= 0 for the constraint to be met.
    (From IP3)
    """
    circles = _reshape_params_numba(params_flat, n_circles)
    centers = circles[:, :2]
    radii = circles[:, 2]

    # Total constraints: N*4 (containment) + N*(N-1)/2 (overlap)
    # Radii >= 0 handled by bounds in SLSQP
    num_boundary_constraints = 4 * n_circles
    num_overlap_constraints = n_circles * (n_circles - 1) // 2
    total_constraints = num_boundary_constraints + num_overlap_constraints
    
    constraints_array = np.empty(total_constraints, dtype=np.float64)
    idx = 0

    # Boundary constraints: x - r >= 0, 1 - x - r >= 0, y - r >= 0, 1 - y - r >= 0
    for i in range(n_circles):
        constraints_array[idx] = centers[i, 0] - radii[i]
        idx += 1
        constraints_array[idx] = unit_square_side - centers[i, 0] - radii[i]
        idx += 1
        constraints_array[idx] = centers[i, 1] - radii[i]
        idx += 1
        constraints_array[idx] = unit_square_side - centers[i, 1] - radii[i]
        idx += 1
    
    # Overlap constraints: (x_i-x_j)^2 + (y_i-y_j)^2 - (r_i+r_j)^2 >= 0
    for i in range(n_circles):
        for j in range(i + 1, n_circles):
            dx = centers[i, 0] - centers[j, 0]
            dy = centers[i, 1] - centers[j, 1]
            dist_sq = dx*dx + dy*dy
            sum_radii_sq = (radii[i] + radii[j])**2
            constraints_array[idx] = dist_sq - sum_radii_sq
            idx += 1
            
    return constraints_array

# --- Helper for generating structured initial configurations --- (from IP3)
def _generate_uniform_grid_seed(num_circles, target_rows, target_cols, base_r_factor=0.95):
    """
    Generates a flattened array of (x,y,r) for circles arranged in a uniform grid.
    base_r_factor: A factor to slightly reduce the calculated max radius to allow wiggle room.
    """
    r_x_max = 0.5 / target_cols if target_cols > 0 else 0.5
    r_y_max = 0.5 / target_rows if target_rows > 0 else 0.5
    
    base_r = min(r_x_max, r_y_max) * base_r_factor
    base_r = np.clip(base_r, EPSILON, 0.5)

    circles_list = []
    current_count = 0
    
    x_step = (UNIT_SQUARE_SIDE - 2 * base_r) / (target_cols - 1) if target_cols > 1 else 0
    y_step = (UNIT_SQUARE_SIDE - 2 * base_r) / (target_rows - 1) if target_rows > 1 else 0

    for i in range(target_rows):
        y = base_r + i * y_step if target_rows > 1 else UNIT_SQUARE_SIDE / 2
        for j in range(target_cols):
            if current_count >= num_circles:
                break
            x = base_r + j * x_step if target_cols > 1 else UNIT_SQUARE_SIDE / 2
            
            circles_list.append([x, y, base_r])
            current_count += 1
        if current_count >= num_circles:
            break
            
    while len(circles_list) < num_circles:
        circles_list.append([0.5, 0.5, EPSILON]) # Fill remaining with minimal circles
        
    return np.array(circles_list).flatten()

def _generate_initial_population(N, bounds, popsize, rng):
    """
    Generates an initial population for Differential Evolution, combining
    structured seeds and Latin Hypercube Sampling. (From IP3)
    """
    dim = len(bounds)
    initial_population = np.zeros((popsize, dim))
    num_seeds_generated = 0

    # Structured seeds for various grid configurations
    grid_configs = [
        (5, 6), (6, 5), (4, 7), (7, 4), (5, 5),
        (3, 9), (9, 3), (8, 4), (4, 8) # Added more from IP3
    ]
    
    for rows, cols in grid_configs:
        if num_seeds_generated < popsize:
            initial_population[num_seeds_generated] = _generate_uniform_grid_seed(N, rows, cols, base_r_factor=0.95)
            num_seeds_generated += 1

    # Perturbed versions of the best initial seeds
    if num_seeds_generated < popsize:
        seed1_perturbed = initial_population[0].copy()
        noise_scale_pos = 0.02
        noise_scale_rad = 0.01
        seed1_perturbed[::3] += rng.uniform(-noise_scale_pos, noise_scale_pos, N) # x
        seed1_perturbed[1::3] += rng.uniform(-noise_scale_pos, noise_scale_pos, N) # y
        seed1_perturbed[2::3] += rng.uniform(-noise_scale_rad, noise_scale_rad, N) # r
        for k in range(dim): # Clip to bounds
            seed1_perturbed[k] = np.clip(seed1_perturbed[k], bounds[k][0], bounds[k][1])
        initial_population[num_seeds_generated] = seed1_perturbed
        num_seeds_generated += 1
    
    if num_seeds_generated < popsize:
        seed2_perturbed = initial_population[1].copy()
        noise_scale_pos = 0.02
        noise_scale_rad = 0.01
        seed2_perturbed[::3] += rng.uniform(-noise_scale_pos, noise_scale_pos, N) # x
        seed2_perturbed[1::3] += rng.uniform(-noise_scale_pos, noise_scale_pos, N) # y
        seed2_perturbed[2::3] += rng.uniform(-noise_scale_rad, noise_scale_rad, N) # r
        for k in range(dim): # Clip to bounds
            seed2_perturbed[k] = np.clip(seed2_perturbed[k], bounds[k][0], bounds[k][1])
        initial_population[num_seeds_generated] = seed2_perturbed
        num_seeds_generated += 1

    # Fill the rest of the population with Latin Hypercube Sampling
    num_lhs_samples = popsize - num_seeds_generated
    if num_lhs_samples > 0:
        l_bounds = np.array([b[0] for b in bounds])
        u_bounds = np.array([b[1] for b in bounds])
        diff = u_bounds - l_bounds

        lhs_samples = np.zeros((num_lhs_samples, dim))
        for j in range(dim):
            points = (rng.permutation(num_lhs_samples) + rng.uniform(0, 1, num_lhs_samples)) / num_lhs_samples
            lhs_samples[:, j] = points
        
        initial_population[num_seeds_generated:] = l_bounds + lhs_samples * diff
        
    # Final check to ensure all initial population members are within bounds
    for i in range(popsize):
        for k in range(dim):
            initial_population[i, k] = np.clip(initial_population[i, k], bounds[k][0], bounds[k][1])

    return initial_population

# --- Objective Functions for DE and SLSQP ---
def objective_de(params, penalty_factor_overlap, penalty_factor_boundary):
    """
    Objective function to minimize for Differential Evolution.
    Returns -sum_radii + scaled_penalties.
    (Adapted from IP1/IP2 to accept separate penalty factors).
    """
    circles = _reshape_params_numba(params, N_CIRCLES)
    radii = circles[:, 2]
    radii_positive = np.maximum(radii, EPSILON) # Ensure radii are positive for sum calculation

    # Calculate penalties using the Numba-optimized function
    boundary_penalty_sq, overlap_penalty_sq = _calculate_penalties_de_numba(params, N_CIRCLES, EPSILON, UNIT_SQUARE_SIDE)

    total_penalty = penalty_factor_boundary * boundary_penalty_sq + penalty_factor_overlap * overlap_penalty_sq
    
    # Minimize -sum_radii to maximize sum_radii
    return -np.sum(radii_positive) + total_penalty

def _get_slsqp_constraints_nonlinear():
    """Helper to create NonlinearConstraint object for SLSQP optimization. (From IP1/IP3)"""
    num_overlap_constraints = N_CIRCLES * (N_CIRCLES - 1) // 2
    total_constraints = 4 * N_CIRCLES + num_overlap_constraints
    return NonlinearConstraint(
        lambda params: _slsqp_constraints_func_numba(params, N_CIRCLES, UNIT_SQUARE_SIDE),
        lb=np.zeros(total_constraints, dtype=np.float64),
        ub=np.full(total_constraints, np.inf, dtype=np.float64)
    )

def objective_slsqp(params):
    """
    Objective function to minimize for SLSQP (simply -sum_radii). (From IP3)
    """
    circles = params.reshape(N_CIRCLES, 3)
    radii = circles[:, 2]
    return -np.sum(np.maximum(radii, EPSILON)) # Ensure positive radii for objective calculation

def circle_packing26() -> np.ndarray:
    """
    Places 26 non-overlapping circles in the unit square in order to maximize the sum of radii.
    Uses a three-stage optimization:
    1. Differential Evolution for global search.
    2. Initial Local Refinement with SLSQP.
    3. Iterated Local Search (ILS) with perturbations and further SLSQP refinements.

    Returns:
        circles: np.array of shape (26,3), where the i-th row (x,y,r) stores the (x,y) coordinates of the i-th circle of radius r.
    """
    start_time = time.time()
    
    # Set random seed for reproducibility
    np.random.seed(GLOBAL_RANDOM_SEED)
    rng = np.random.default_rng(GLOBAL_RANDOM_SEED) # For use with _generate_initial_population and perturbations

    # Define bounds for each (x, y, r) parameter
    bounds = []
    for _ in range(N_CIRCLES):
        bounds.append((0.0, UNIT_SQUARE_SIDE)) # x-coordinate
        bounds.append((0.0, UNIT_SQUARE_SIDE)) # y-coordinate
        bounds.append((EPSILON, 0.5)) # radius (max radius is 0.5 for a single contained circle)

    # Generate custom initial population using structured seeds and LHS
    initial_pop_de = _generate_initial_population(N_CIRCLES, bounds, DE_POPSIZE, rng)

    # Stage 1: Global Optimization with Differential Evolution
    print(f"Starting Differential Evolution for {N_CIRCLES} circles (max_iter={DE_MAXITER}, popsize={DE_POPSIZE})...")
    de_result = differential_evolution(
        func=objective_de,
        bounds=bounds,
        args=(PENALTY_FACTOR_DE_OVERLAP, PENALTY_FACTOR_DE_BOUNDARY), # Pass separate penalty factors
        maxiter=DE_MAXITER,
        popsize=DE_POPSIZE,
        workers=DE_WORKERS,
        seed=GLOBAL_RANDOM_SEED,
        disp=False, # Set to True for verbose output
        polish=True, # Enable polish for a better starting point for SLSQP (from IP1/IP3)
        init=initial_pop_de # Provide the custom initial population
    )
    
    de_runtime = time.time() - start_time
    print(f"Differential Evolution finished in {de_runtime:.2f} seconds.")
    print(f"DE Best objective (including penalties): {de_result.fun}")
    
    # Extract DE result and evaluate its unpenalized sum of radii and penalties
    initial_guess_for_slsqp = de_result.x
    de_circles_raw = _reshape_params_numba(initial_guess_for_slsqp, N_CIRCLES)
    de_radii_sum = np.sum(np.maximum(de_circles_raw[:, 2], EPSILON))
    de_boundary_pen_sq, de_overlap_pen_sq = _calculate_penalties_de_numba(initial_guess_for_slsqp, N_CIRCLES, EPSILON, UNIT_SQUARE_SIDE)
    print(f"DE Solution (pre-SLSQP): Sum Radii = {de_radii_sum:.8f}, Boundary Pen = {de_boundary_pen_sq:.6e}, Overlap Pen = {de_overlap_pen_sq:.6e}")
    
    # Stage 2: Initial Local Refinement with SLSQP
    print(f"\nStarting initial SLSQP refinement...")
    slsqp_constraint_obj = _get_slsqp_constraints_nonlinear()

    # Ensure initial_guess for SLSQP strictly respects the bounds
    initial_guess_clipped = np.copy(initial_guess_for_slsqp)
    for i, (lower, upper) in enumerate(bounds):
        initial_guess_clipped[i] = np.clip(initial_guess_clipped[i], lower, upper)

    slsqp_result = minimize(
        fun=objective_slsqp,
        x0=initial_guess_clipped,
        method='SLSQP',
        bounds=bounds,
        constraints=slsqp_constraint_obj,
        options={'maxiter': SLSQP_INITIAL_MAXITER, 'ftol': SLSQP_INITIAL_FTOL, 'gtol': SLSQP_INITIAL_GTOL, 'disp': False} # Set to True for verbose output
    )
    
    best_params = slsqp_result.x
    best_sum_radii = -slsqp_result.fun if slsqp_result.success else -np.inf
    best_slsqp_result = slsqp_result # Keep track of the result object

    print(f"Initial SLSQP finished. Best Sum of Radii: {best_sum_radii:.12f}")

    # Stage 3: Adaptive Iterated Local Search (ILS) for further improvement
    print(f"\nStarting Adaptive Iterated Local Search (ILS) with {N_CIRCLES} circles (iterations={NUM_ILS_ITERATIONS})...")
    rng_perturb = np.random.default_rng(GLOBAL_RANDOM_SEED + 1) # Use a different seed for perturbations

    # Adaptive parameters that adjust based on progress
    base_noise_xy = NOISE_SCALE_XY
    base_noise_r = NOISE_SCALE_R
    no_improvement_count = 0
    max_no_improvement = 15 # Early termination if no improvement for this many iterations (from IP3)

    for i in range(NUM_ILS_ITERATIONS):
        # Adaptive noise scaling - increase perturbation if no recent improvements
        noise_multiplier = 1.0 + (no_improvement_count * 0.1) # Gradually increase perturbation
        current_noise_scale_xy = base_noise_xy * noise_multiplier
        current_noise_scale_r = base_noise_r * noise_multiplier
        
        # print(f"\nILS Iteration {i+1}/{NUM_ILS_ITERATIONS} (Current best sum_radii: {best_sum_radii:.8f}, noise_mult: {noise_multiplier:.2f})...")
        
        # Perturb the current best solution with adaptive noise
        perturbed_guess = np.copy(best_params)
        
        perturbed_guess[0::3] += rng_perturb.normal(0, current_noise_scale_xy, N_CIRCLES)
        perturbed_guess[1::3] += rng_perturb.normal(0, current_noise_scale_xy, N_CIRCLES)
        perturbed_guess[2::3] += rng_perturb.normal(0, current_noise_scale_r, N_CIRCLES)

        # Ensure the perturbed guess remains within the defined bounds
        for j in range(len(bounds)):
            low, high = bounds[j]
            perturbed_guess[j] = np.clip(perturbed_guess[j], low, high)

        # Run SLSQP again from the perturbed starting point
        current_slsqp_result = minimize(
            fun=objective_slsqp,
            x0=perturbed_guess,
            method='SLSQP',
            bounds=bounds,
            constraints=slsqp_constraint_obj,
            options={'maxiter': SLSQP_ILS_MAXITER, 'ftol': SLSQP_ILS_FTOL, 'gtol': SLSQP_ILS_GTOL, 'disp': False}
        )
        
        current_sum_radii = -current_slsqp_result.fun if current_slsqp_result.success else -np.inf

        if current_sum_radii > best_sum_radii:
            # print(f"  Improved solution found! Sum of radii: {current_sum_radii:.12f}")
            best_sum_radii = current_sum_radii
            best_params = current_slsqp_result.x
            best_slsqp_result = current_slsqp_result
            no_improvement_count = 0
        else:
            # print(f"  No improvement in this ILS iteration.")
            no_improvement_count += 1
            
        if no_improvement_count >= max_no_improvement:
            print(f"  Early termination: No improvement for {max_no_improvement} consecutive iterations.")
            break

    final_result = best_slsqp_result

    total_runtime = time.time() - start_time
    print(f"\nOptimization finished. Total optimization time: {total_runtime:.2f} seconds.")

    # Final analysis and output formatting
    final_params = final_result.x
    final_circles = _reshape_params_numba(final_params, N_CIRCLES)
    final_circles[:, 2] = np.maximum(final_circles[:, 2], EPSILON) # Ensure positive radii for output

    final_sum_radii = np.sum(final_circles[:, 2])
    final_boundary_pen_sq, final_overlap_pen_sq = _calculate_penalties_de_numba(final_params, N_CIRCLES, EPSILON, UNIT_SQUARE_SIDE)
    
    print(f"\nFinal Solution Summary:")
    print(f"Total Circles: {N_CIRCLES}")
    print(f"Sum of Radii: {final_sum_radii:.12f}")
    print(f"Final Boundary Violations (squared sum): {final_boundary_pen_sq:.6e}")
    print(f"Final Overlap Violations (squared sum): {final_overlap_pen_sq:.6e}")
    print(f"SLSQP Final Success: {final_result.success}")
    print(f"SLSQP Final Message: {final_result.message}")

    if final_boundary_pen_sq > 1e-8 or final_overlap_pen_sq > 1e-8:
        print("WARNING: Final solution has detectable constraint violations. Quality might be compromised.")

    # Fallback if SLSQP failed to converge or has significant violations
    if not final_result.success or final_boundary_pen_sq > 1e-6 or final_overlap_pen_sq > 1e-6: # Use slightly looser tolerance for fallback trigger
        print("WARNING: Final SLSQP result is not fully feasible or converged. Falling back to DE result if feasible.")
        de_constraint_values = _slsqp_constraints_func_numba(de_result.x, N_CIRCLES, UNIT_SQUARE_SIDE)
        if np.all(de_constraint_values >= -1e-6): # Check feasibility of DE result with SLSQP constraints
            print(f"DE result is feasible and will be used. Sum_radii: {np.sum(np.maximum(de_result.x[2::3], EPSILON)):.6f}")
            circles = de_result.x.reshape((N_CIRCLES, 3))
            circles[:, 2] = np.maximum(circles[:, 2], EPSILON)
        else:
            print("Both SLSQP and DE results are infeasible. Returning a fallback grid.")
            circles = np.zeros((N_CIRCLES, 3))
            grid_side = int(np.ceil(np.sqrt(N_CIRCLES)))
            r_grid = 1.0 / (2.0 * grid_side)
            idx = 0
            for row in range(grid_side):
                for col in range(grid_side):
                    if idx < N_CIRCLES:
                        circles[idx] = [(2*col + 1) * r_grid, (2*row + 1) * r_grid, r_grid]
                        idx += 1
            circles[:, 2] = np.maximum(circles[:, 2], EPSILON) # Ensure positive radii
    else:
        circles = final_circles

    return circles


# EVOLVE-BLOCK-END
