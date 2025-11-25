# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize, differential_evolution
from numba import njit
import time # For eval_time tracking

# Constants for the problem
N_CIRCLES = 26
UNIT_SQUARE_SIDE = 1.0
MAX_RADIUS = UNIT_SQUARE_SIDE / 2.0  # Max radius for a single circle in the square
RNG_SEED = 42 # Fixed random seed for reproducibility

# Optimization parameters - tuned based on inspiration programs for performance and accuracy
RADIUS_MIN = 1e-7 # Minimum radius to ensure positive radii, aligned with inspiration 1
PENALTY_FACTOR = 1e5 # Significantly increased penalty factor for DE to strongly prioritize feasibility
DE_MAXITER = 2000 # Max iterations for Differential Evolution (from Inspiration 1)
DE_POPSIZE = 30   # Population size for Differential Evolution (from Inspiration 1)

# SLSQP parameters for local refinement
SLSQP_INITIAL_MAXITER = 2000 # Max iterations for the first SLSQP run (from Inspiration 1)
SLSQP_ILS_MAXITER = 4000     # Max iterations for SLSQP during Iterated Local Search (ILS) (from Inspiration 1)
SLSQP_FTOL = 1e-9            # Tighter tolerance for SLSQP convergence (from Inspiration 1)

# Iterated Local Search (ILS) parameters
NUM_ILS_ITERATIONS = 50 # Number of perturbation-refinement cycles (from Inspiration 1)
ILS_NOISE_SCALE_XY = 0.015 # Perturbation scale for x, y coordinates (from Inspiration 1)
ILS_NOISE_SCALE_R = 0.005  # Perturbation scale for radii (from Inspiration 1)

# --- Helper Functions ---
def unpack_params(params: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Unpacks a 1D parameter array into x, y, r arrays."""
    x = params[0::3]
    y = params[1::3]
    r = params[2::3]
    return x, y, r

@njit(cache=True, fastmath=True) # Numba decorator for performance with caching and fast math
def _calculate_penalty_violations(x: np.ndarray, y: np.ndarray, r: np.ndarray, n: int) -> float:
    """
    Calculates the total *squared* violation of containment and non-overlap constraints.
    Returns a scalar sum of all positive violations squared.
    Squared violations create a smoother landscape for gradient-free optimizers like DE.
    Adapted from Inspiration 1's robust penalty formulation.
    """
    total_violation_squared = 0.0

    # 1. Containment constraints: r <= pos <= 1-r  =>  pos - r >= 0  AND  1 - pos - r >= 0
    # Also, r >= RADIUS_MIN
    for i in range(n):
        total_violation_squared += max(0.0, r[i] - x[i])**2
        total_violation_squared += max(0.0, x[i] + r[i] - UNIT_SQUARE_SIDE)**2
        total_violation_squared += max(0.0, r[i] - y[i])**2
        total_violation_squared += max(0.0, y[i] + r[i] - UNIT_SQUARE_SIDE)**2
        total_violation_squared += max(0.0, RADIUS_MIN - r[i])**2 # Ensure radius is not too small

    # 2. Non-overlap constraints: (xi-xj)^2 + (yi-yj)^2 >= (ri + rj)^2
    # Violation is max(0, (ri+rj) - dist) (sum of radii minus actual distance), then squared.
    for i in range(n):
        for j in range(i + 1, n): # Only check unique pairs
            dx = x[i] - x[j]
            dy = y[i] - y[j]
            dist_sq = dx*dx + dy*dy
            min_dist = r[i] + r[j]
            
            if min_dist*min_dist > dist_sq: # If (r_i+r_j)^2 > dist_sq, then there is an overlap
                total_violation_squared += (min_dist - np.sqrt(dist_sq))**2

    return total_violation_squared

@njit(cache=True, fastmath=True) # Numba decorator for performance with caching and fast math
def _calculate_slsqp_constraints(params_flat, n_circles, square_side, min_radius_val):
    """
    Numba-optimized function to evaluate all SLSQP constraints at once.
    Returns a vector where each element must be >= 0 for the constraint to be met.
    Adapted from the original _check_overlap_and_boundary and Inspiration 1.
    """
    circles = params_flat.reshape(n_circles, 3)
    
    num_constraints = n_circles * 6 + n_circles * (n_circles - 1) // 2
    constraints_array = np.empty(num_constraints, dtype=params_flat.dtype)
    constraint_idx = 0

    # Boundary and radius constraints
    for i in range(n_circles):
        x_i, y_i, r_i = circles[i]
        
        # r <= x <= 1-r  and  r <= y <= 1-r
        constraints_array[constraint_idx] = x_i - r_i
        constraint_idx += 1
        constraints_array[constraint_idx] = square_side - x_i - r_i
        constraint_idx += 1
        constraints_array[constraint_idx] = y_i - r_i
        constraint_idx += 1
        constraints_array[constraint_idx] = square_side - y_i - r_i
        constraint_idx += 1
        
        # Ensure radius is positive and within max_radius
        constraints_array[constraint_idx] = r_i - min_radius_val # r_i >= min_radius_val
        constraint_idx += 1
        constraints_array[constraint_idx] = MAX_RADIUS - r_i # r_i <= MAX_RADIUS
        constraint_idx += 1

    # Non-overlap constraints
    for i in range(n_circles):
        for j in range(i + 1, n_circles):
            x_i, y_i, r_i = circles[i]
            x_j, y_j, r_j = circles[j]

            dist_sq = (x_i - x_j)**2 + (y_i - y_j)**2
            min_dist_sq = (r_i + r_j)**2
            constraints_array[constraint_idx] = dist_sq - min_dist_sq # dist_sq - min_dist_sq >= 0
            constraint_idx += 1
            
    return constraints_array


def objective_function_de(params_flat, n_circles, penalty_factor):
    """
    Objective function for Differential Evolution: minimize -sum_radii + penalty.
    Adapted from Inspiration 1.
    """
    x, y, r = unpack_params(params_flat)
    
    sum_radii = np.sum(r)
    total_violation_squared = _calculate_penalty_violations(x, y, r, n_circles)
    
    # We want to maximize sum_radii, so we minimize -sum_radii.
    # Add a penalty for constraint violations.
    return -sum_radii + penalty_factor * total_violation_squared

def objective_function_slsqp(params_flat):
    """
    Objective function for SLSQP: minimize -sum_radii (assuming constraints are handled separately).
    Adapted from Inspiration 1.
    """
    radii = params_flat[2::3] # Every third element starting from index 2 is a radius
    return -np.sum(radii)

def generate_initial_guess_grid(n_circles: int, grid_cols: int = 5) -> np.ndarray:
    """
    Generates a structured initial guess by placing circles on a grid.
    Adapted from Inspiration 1.
    """
    params = np.zeros(n_circles * 3)
    grid_rows = int(np.ceil(n_circles / grid_cols))
    
    # Create grid points slightly inside the unit square to start
    x_steps = np.linspace(0.1, UNIT_SQUARE_SIDE - 0.1, grid_cols)
    y_steps = np.linspace(0.1, UNIT_SQUARE_SIDE - 0.1, grid_rows)
    
    # Use an initial radius that's a reasonable guess for a dense packing
    initial_radius_base = (UNIT_SQUARE_SIDE / (2 * max(grid_rows, grid_cols))) * 0.9 # Slightly smaller than max possible for grid
    
    rng = np.random.default_rng(RNG_SEED)
    idx = 0
    for r_idx in range(grid_rows):
        for c_idx in range(grid_cols):
            if idx < n_circles:
                params[3*idx] = x_steps[c_idx] # x coordinate
                params[3*idx+1] = y_steps[r_idx] # y coordinate
                # Add a small jitter to radii to break symmetry and encourage exploration
                params[3*idx+2] = np.clip(initial_radius_base + rng.uniform(-0.005, 0.005), RADIUS_MIN, MAX_RADIUS)
                idx += 1
    return params

def get_slsqp_constraints_dict(n_circles, square_side, min_radius_val):
    """
    Generates a single, vectorized constraint function for SLSQP.
    This is vastly more efficient than providing a list of individual lambda functions.
    Adapted from Inspiration 1 and the original target program.
    """
    return [{'type': 'ineq', 'fun': lambda params_flat: _calculate_slsqp_constraints(params_flat, n_circles, square_side, min_radius_val)}]


def circle_packing26() -> np.ndarray:
    """
    Places 26 non-overlapping circles in the unit square in order to maximize the sum of radii.
    Employs a multi-phase optimization: Differential Evolution for global search,
    followed by SLSQP for local refinement and Iterated Local Search (ILS).
    This implementation synthesizes the best practices from the provided inspiration programs,
    especially Inspiration 1 which successfully beat the benchmark.

    Returns:
        circles: np.array of shape (26,3), where the i-th row (x,y,r) stores the (x,y) coordinates of the i-th circle of radius r.
    """
    n = N_CIRCLES
    start_time = time.time()
    np.random.seed(RNG_SEED) # Set numpy seed for reproducibility

    # Define bounds for each parameter (x, y, r).
    bounds = [(0.0, UNIT_SQUARE_SIDE), (0.0, UNIT_SQUARE_SIDE), (RADIUS_MIN, MAX_RADIUS)] * n

    # --- Prepare initial population for Differential Evolution ---
    # Generate a structured initial guess (grid-based)
    grid_guess = generate_initial_guess_grid(n)

    # Create an initial population for DE: first member is the smart grid guess, rest are random
    initial_population = np.zeros((DE_POPSIZE, len(bounds)))
    initial_population[0, :] = grid_guess # First member is the smart guess

    rng_de_init = np.random.default_rng(RNG_SEED)
    for i in range(1, DE_POPSIZE):
        for j in range(len(bounds)):
            low, high = bounds[j]
            initial_population[i, j] = rng_de_init.uniform(low, high)

    # --- Phase 1: Global Optimization with Differential Evolution ---
    print("Starting Differential Evolution (Phase 1)...")
    result_de = differential_evolution(
        func=objective_function_de,
        bounds=bounds,
        args=(n, PENALTY_FACTOR),
        strategy='best1bin', # A robust strategy for DE
        maxiter=DE_MAXITER,
        popsize=DE_POPSIZE,
        tol=0.001, # Tighter tolerance for DE to find a better starting point
        mutation=(0.5, 1.0), # Differential weight F
        recombination=0.7, # Crossover probability CR
        seed=RNG_SEED, # Fixed seed for determinism
        disp=True, # Display progress
        workers=-1, # Use all available cores
        init=initial_population # Provide the smart initial population
    )

    best_params_de = result_de.x
    x_de, y_de, r_de = unpack_params(best_params_de)
    unpenalized_sum_radii_de = np.sum(r_de)
    violations_de = _calculate_penalty_violations(x_de, y_de, r_de, n)
    print(f"Differential Evolution finished. Best objective (penalized): {result_de.fun:.6f}")
    print(f"Sum of radii after DE (unpenalized): {unpenalized_sum_radii_de:.6f}")
    print(f"Total squared constraint violation after DE: {violations_de:.6e}")

    # --- Phase 2: Initial Local Refinement with SLSQP ---
    print("\nStarting SLSQP Initial Local Refinement (Phase 2)...")
    slsqp_constraints = get_slsqp_constraints_dict(n, UNIT_SQUARE_SIDE, RADIUS_MIN)

    initial_slsqp_result = minimize(
        fun=objective_function_slsqp,
        x0=best_params_de, # Start from DE's best result
        method='SLSQP',
        bounds=bounds,
        constraints=slsqp_constraints,
        options={'maxiter': SLSQP_INITIAL_MAXITER, 'ftol': SLSQP_FTOL, 'disp': True}
    )

    best_params = initial_slsqp_result.x
    best_sum_radii = -initial_slsqp_result.fun if initial_slsqp_result.success else -np.inf

    print(f"Initial SLSQP finished. Success: {initial_slsqp_result.success}, Message: {initial_slsqp_result.message}")
    print(f"Sum of radii after initial SLSQP: {best_sum_radii:.12f}")

    # --- Phase 3: Iterated Local Search (ILS) ---
    print("\nStarting Iterated Local Search (Phase 3)...")
    rng_perturb = np.random.default_rng(RNG_SEED + 1) # Use a different seed for perturbations

    for i in range(NUM_ILS_ITERATIONS):
        # print(f"\nILS Iteration {i+1}/{NUM_ILS_ITERATIONS}...") # Reduced verbosity for quicker runs
        
        # Perturb the current best solution
        perturbed_guess = np.copy(best_params)
        
        # Apply noise to coordinates and radii
        perturbed_guess[0::3] += rng_perturb.normal(0, ILS_NOISE_SCALE_XY, n)
        perturbed_guess[1::3] += rng_perturb.normal(0, ILS_NOISE_SCALE_XY, n)
        perturbed_guess[2::3] += rng_perturb.normal(0, ILS_NOISE_SCALE_R, n)

        # Ensure the perturbed guess remains within the defined bounds
        for j in range(len(bounds)):
            low, high = bounds[j]
            perturbed_guess[j] = np.clip(perturbed_guess[j], low, high)

        # Run SLSQP again from the perturbed starting point
        current_slsqp_result = minimize(
            fun=objective_function_slsqp,
            x0=perturbed_guess,
            method='SLSQP',
            bounds=bounds,
            constraints=slsqp_constraints,
            options={'maxiter': SLSQP_ILS_MAXITER, 'ftol': SLSQP_FTOL * 0.1, 'disp': False} # Tighter ftol for ILS, no display
        )
        
        current_sum_radii = -current_slsqp_result.fun if current_slsqp_result.success else -np.inf

        if current_sum_radii > best_sum_radii:
            # print(f"  Improved solution found! Sum of radii: {current_sum_radii:.12f}") # Reduced verbosity
            best_sum_radii = current_sum_radii
            best_params = current_slsqp_result.x
        # else:
            # print(f"  No improvement. Current best: {best_sum_radii:.12f}") # Reduced verbosity

    # Final result is the best found across all phases
    final_params = best_params

    # Final validation and cleanup
    final_x, final_y, final_r = unpack_params(final_params)
    final_violations = _calculate_penalty_violations(final_x, final_y, final_r, n)
    
    print(f"\nOptimization complete. Final sum of radii: {np.sum(final_r):.12f}")
    print(f"Total squared constraint violation: {final_violations:.6e}")

    # Ensure radii are strictly positive
    final_r = np.maximum(final_r, RADIUS_MIN)
    
    # Construct the output array
    circles = np.column_stack((final_x, final_y, final_r))

    return circles


# EVOLVE-BLOCK-END
