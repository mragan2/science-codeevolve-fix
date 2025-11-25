# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import differential_evolution, minimize
from numba import njit
import time # Added for accurate runtime tracking

# --- Configuration Parameters ---
N_CIRCLES = 26
UNIT_SQUARE_SIDE = 1.0
GLOBAL_RANDOM_SEED = 42 # For reproducibility

# Common constants
PENALTY_FACTOR = 1e5   # Penalty for constraint violations in DE objective
EPSILON_R_BOUND = 1e-6 # Lower bound for radii in optimization
EPSILON_FINAL_CLIP = 1e-9 # For final clipping of radii to ensure positivity

# Initial guess parameters
INITIAL_RADIUS_GRID = 0.095
INITIAL_RADIUS_JITTER_RANGE = (-7e-4, 7e-4)

# Differential Evolution (DE) parameters
DE_MAXITER = 4000 # Further reduced max iterations for DE
DE_POPSIZE = 35
DE_TOL = 0.001
DE_MUTATION = (0.5, 1.0)
DE_RECOMBINATION = 0.7
DE_WORKERS = -1

# SLSQP parameters for initial refinement
SLSQP_MAXITER_INITIAL = 3000 # Further reduced max iterations for initial SLSQP
SLSQP_FTOL_INITIAL = 1e-9

# Iterated Local Search (ILS) parameters
NUM_ILS_ITERATIONS = 50 # Reduced number of perturbation-refinement cycles
ILS_PERTURB_XY_SCALE = 0.015
ILS_PERTURB_R_SCALE = 0.01
ILS_RANDOM_SEED_OFFSET = 100
SLSQP_MAXITER_ILS = 1500 # Further reduced max iterations for ILS SLSQP
SLSQP_FTOL_ILS = 1e-9

@njit(cache=True) # Cache the compiled function for faster subsequent calls
def _evaluate_constraints_numba(x_coords, y_coords, radii):
    """
    Numba-optimized function to evaluate all constraint violations for a given configuration.
    Returns:
        total_violation: Sum of all constraint violations.
        violations_array: Array of individual violations (for detailed analysis).
    """
    n = len(radii)
    
    # Max possible violations: N*5 (containment + r>=0) + N*(N-1)/2 (overlap)
    # 26*5 = 130, 26*25/2 = 325. Total = 455.
    max_violations = n * 5 + n * (n - 1) // 2
    violations = np.zeros(max_violations, dtype=np.float64)
    violation_idx = 0

    # 1. Containment constraints (4 per circle) and r >= 0
    for i in range(n):
        r_i = radii[i]
        x_i = x_coords[i]
        y_i = y_coords[i]

        violations[violation_idx] = max(0.0, r_i - x_i) # x_i >= r_i
        violation_idx += 1
        violations[violation_idx] = max(0.0, x_i + r_i - UNIT_SQUARE_SIDE) # x_i <= 1 - r_i
        violation_idx += 1
        violations[violation_idx] = max(0.0, r_i - y_i) # y_i >= r_i
        violation_idx += 1
        violations[violation_idx] = max(0.0, y_i + r_i - UNIT_SQUARE_SIDE) # y_i <= 1 - r_i
        violation_idx += 1
        violations[violation_idx] = max(0.0, -r_i) # radii must be non-negative
        violation_idx += 1

    # 2. Non-overlap constraints (N*(N-1)/2 pairs)
    for i in range(n):
        for j in range(i + 1, n):
            r_i = radii[i]
            r_j = radii[j]
            x_i = x_coords[i]
            y_i = y_coords[i]
            x_j = x_coords[j]
            y_j = y_coords[j]

            dist_sq = (x_i - x_j)**2 + (y_i - y_j)**2
            min_dist_sq = (r_i + r_j)**2
            violations[violation_idx] = max(0.0, min_dist_sq - dist_sq) # dist^2 >= (r_i+r_j)^2
            violation_idx += 1

    return np.sum(violations), violations

@njit(cache=True)
def _slsqp_constraints_numba(params):
    """
    Numba-optimized function to evaluate all SLSQP constraints at once.
    Returns a vector where each element must be >= 0 for the constraint to be met.
    """
    n = N_CIRCLES
    x_coords = params[0::3]
    y_coords = params[1::3]
    radii = params[2::3]

    # Total constraints: N*5 (containment, r>=0) + N*(N-1)/2 (overlap)
    max_constraints = n * 5 + n * (n - 1) // 2
    constraints = np.zeros(max_constraints, dtype=np.float64)
    constraint_idx = 0

    # 1. Containment and radius constraints (g(x) >= 0)
    for i in range(n):
        r_i = radii[i]
        x_i = x_coords[i]
        y_i = y_coords[i]
        
        constraints[constraint_idx] = x_i - r_i
        constraint_idx += 1
        constraints[constraint_idx] = UNIT_SQUARE_SIDE - x_i - r_i
        constraint_idx += 1
        constraints[constraint_idx] = y_i - r_i
        constraint_idx += 1
        constraints[constraint_idx] = UNIT_SQUARE_SIDE - y_i - r_i
        constraint_idx += 1
        constraints[constraint_idx] = r_i # r_i >= 0
        constraint_idx += 1

    # 2. Non-overlap constraints (g(x) >= 0)
    for i in range(n):
        for j in range(i + 1, n):
            r_i = radii[i]
            r_j = radii[j]
            x_i = x_coords[i]
            y_i = y_coords[i]
            x_j = x_coords[j]
            y_j = y_coords[j]

            dist_sq = (x_i - x_j)**2 + (y_i - y_j)**2
            min_dist_sq = (r_i + r_j)**2
            constraints[constraint_idx] = dist_sq - min_dist_sq
            constraint_idx += 1
            
    return constraints

def generate_initial_guess(n_circles, grid_cols=5):
    """
    Generates a structured initial guess by placing circles on a grid.
    Uses defined constants for initial radius and jitter.
    """
    params = np.zeros(n_circles * 3)
    grid_rows = int(np.ceil(n_circles / grid_cols))
    
    # Create grid points slightly inside the unit square to start
    x_steps = np.linspace(0.1, UNIT_SQUARE_SIDE - 0.1, grid_cols)
    y_steps = np.linspace(0.1, UNIT_SQUARE_SIDE - 0.1, grid_rows)
    
    initial_radius = INITIAL_RADIUS_GRID # Use constant
    
    idx = 0
    rng = np.random.default_rng(GLOBAL_RANDOM_SEED)
    for r in range(grid_rows):
        for c in range(grid_cols):
            if idx < n_circles:
                params[3*idx] = x_steps[c]
                params[3*idx+1] = y_steps[r]
                # Add a slightly larger jitter to radii to break symmetry and encourage exploration
                params[3*idx+2] = initial_radius + rng.uniform(INITIAL_RADIUS_JITTER_RANGE[0], INITIAL_RADIUS_JITTER_RANGE[1]) # Use constant
                idx += 1
    return params

def objective_with_penalty(params, penalty_factor=PENALTY_FACTOR, return_violations=False): # Use constant
    """
    Objective function for global optimization (e.g., Differential Evolution).
    Minimizes negative sum of radii, with a penalty for constraint violations.
    params: flattened array (x0, y0, r0, x1, y1, r1, ...)
    """
    x_coords = params[0::3]
    y_coords = params[1::3]
    radii = params[2::3]

    # Objective: Maximize sum of radii -> Minimize negative sum of radii
    sum_radii = np.sum(radii)
    objective_val = -sum_radii

    # Calculate violations using the Numba-optimized function
    total_violation, individual_violations = _evaluate_constraints_numba(x_coords, y_coords, radii)

    if return_violations:
        return objective_val, individual_violations
    
    # Add penalty for violations
    objective_val += penalty_factor * total_violation
    
    return objective_val

def get_slsqp_constraints():
    """
    Generates a single, vectorized constraint function for SLSQP.
    This is vastly more efficient than providing a list of individual lambda functions.
    """
    return [{'type': 'ineq', 'fun': _slsqp_constraints_numba}]

def circle_packing26() -> np.ndarray:
    """
    Places 26 non-overlapping circles in the unit square in order to maximize the sum of radii.
    Employs a three-phase optimization: Differential Evolution for global search,
    followed by SLSQP for initial local refinement, and then Iterated Local Search (ILS).

    Returns:
        circles: np.array of shape (26,3), where the i-th row (x,y,r) stores the (x,y) coordinates of the i-th circle of radius r.
    """
    start_time = time.time() # Start global timer
    
    np.random.seed(GLOBAL_RANDOM_SEED)
    rng = np.random.default_rng(GLOBAL_RANDOM_SEED)

    # Define bounds for each parameter (x, y, r). Use EPSILON_R_BOUND.
    bounds = [(0.0, UNIT_SQUARE_SIDE), (0.0, UNIT_SQUARE_SIDE), (EPSILON_R_BOUND, 0.5)] * N_CIRCLES

    # Create an initial population for DE, seeding with a good guess.
    initial_population = np.zeros((DE_POPSIZE, len(bounds))) # Use DE_POPSIZE
    for i in range(len(bounds)):
        low, high = bounds[i]
        initial_population[:, i] = rng.uniform(low, high, size=DE_POPSIZE)

    # Replace the first member of the population with our smart grid-based guess.
    grid_guess = generate_initial_guess(N_CIRCLES)
    initial_population[0, :] = grid_guess

    # Phase 1: Global Optimization with Differential Evolution
    print(f"Starting Differential Evolution (Phase 1) for {N_CIRCLES} circles (max_iter={DE_MAXITER}, popsize={DE_POPSIZE})...")
    de_result = differential_evolution(
        func=objective_with_penalty,
        bounds=bounds,
        strategy='best1bin',
        maxiter=DE_MAXITER,      # Use DE_MAXITER constant
        popsize=DE_POPSIZE,      # Use DE_POPSIZE constant
        tol=DE_TOL,              # Use DE_TOL constant
        mutation=DE_MUTATION,
        recombination=DE_RECOMBINATION,
        seed=GLOBAL_RANDOM_SEED,
        disp=True,
        workers=DE_WORKERS,      # Use DE_WORKERS constant
        init=initial_population
    )
    
    de_runtime = time.time() - start_time # Track DE runtime
    print(f"Differential Evolution finished in {de_runtime:.2f} seconds.")

    # Analyze DE results
    unpenalized_obj_de, violations_de = objective_with_penalty(de_result.x, penalty_factor=0, return_violations=True)
    print(f"Differential Evolution finished. Best objective (penalized): {de_result.fun:.4f}")
    print(f"Sum of radii after DE (unpenalized): {-unpenalized_obj_de:.12f}") # Increased precision for printing
    print(f"Total constraint violation after DE: {violations_de.sum():.10f}")

    # Use the result from DE as the initial guess for local optimization
    initial_guess_slsqp = de_result.x

    # Phase 2: Initial Local Refinement with SLSQP
    print("\nStarting SLSQP Initial Local Refinement (Phase 2)...")
    slsqp_bounds = bounds
    slsqp_constraints = get_slsqp_constraints()

    best_slsqp_result = minimize(
        fun=lambda p: -np.sum(p[2::3]),
        x0=initial_guess_slsqp,
        method='SLSQP',
        bounds=slsqp_bounds,
        constraints=slsqp_constraints,
        options={'maxiter': SLSQP_MAXITER_INITIAL, 'ftol': SLSQP_FTOL_INITIAL, 'disp': True} # Use constants
    )
    best_params = best_slsqp_result.x
    best_sum_radii = -best_slsqp_result.fun if best_slsqp_result.success else -np.inf

    print(f"Initial SLSQP finished. Sum of radii: {best_sum_radii:.12f}")

    # Phase 3: Iterated Local Search (ILS)
    print("\nStarting Iterated Local Search (Phase 3)...")
    
    for i in range(NUM_ILS_ITERATIONS): # Use NUM_ILS_ITERATIONS
        print(f"\nILS Iteration {i+1}/{NUM_ILS_ITERATIONS}...")
        
        # Perturb the current best solution using a unique seed for each iteration
        rng_perturb = np.random.default_rng(GLOBAL_RANDOM_SEED + ILS_RANDOM_SEED_OFFSET + i) # Use GLOBAL_RANDOM_SEED and ILS_RANDOM_SEED_OFFSET
        perturbed_guess = np.copy(best_params)
        
        # Apply noise based on current best solution
        perturbed_guess[0::3] += rng_perturb.normal(0, ILS_PERTURB_XY_SCALE, N_CIRCLES) # Use ILS_PERTURB_XY_SCALE
        perturbed_guess[1::3] += rng_perturb.normal(0, ILS_PERTURB_XY_SCALE, N_CIRCLES) # Use ILS_PERTURB_XY_SCALE
        perturbed_guess[2::3] += rng_perturb.normal(0, ILS_PERTURB_R_SCALE, N_CIRCLES)   # Use ILS_PERTURB_R_SCALE

        # Ensure the perturbed guess remains within the defined bounds
        for j in range(len(bounds)):
            low, high = bounds[j]
            perturbed_guess[j] = np.clip(perturbed_guess[j], low, high)

        # Run SLSQP again from the perturbed starting point
        current_slsqp_result = minimize(
            fun=lambda p: -np.sum(p[2::3]),
            x0=perturbed_guess,
            method='SLSQP',
            bounds=slsqp_bounds,
            constraints=slsqp_constraints,
            options={'maxiter': SLSQP_MAXITER_ILS, 'ftol': SLSQP_FTOL_ILS, 'disp': False} # Use constants
        )
        
        current_sum_radii = -current_slsqp_result.fun if current_slsqp_result.success else -np.inf

        if current_sum_radii > best_sum_radii:
            print(f"  Improved solution found! Sum of radii: {current_sum_radii:.12f}")
            best_sum_radii = current_sum_radii
            best_params = current_slsqp_result.x
            best_slsqp_result = current_slsqp_result # Keep track of the result object for final analysis
        else:
            print(f"  No improvement. Current best: {best_sum_radii:.12f}")

    final_result = best_slsqp_result # The best result found across all phases

    total_runtime = time.time() - start_time # Track total runtime
    print(f"\nOptimization finished. Total runtime: {total_runtime:.2f} seconds.")

    # Analyze final results
    final_sum_radii = -final_result.fun
    _, final_violations = objective_with_penalty(final_result.x, penalty_factor=0, return_violations=True)
    
    print(f"\nFinal Solution Summary:") # More descriptive final printout
    print(f"SLSQP success: {final_result.success}")
    print(f"SLSQP message: {final_result.message}")
    print(f"Final Sum of Radii: {final_sum_radii:.12f}")
    print(f"Total constraint violation after final SLSQP: {final_violations.sum():.10f}")

    # Extract circles from the optimized parameters
    optimized_params = final_result.x
    circles = np.zeros((N_CIRCLES, 3), dtype=np.float64)
    for i in range(N_CIRCLES):
        circles[i, 0] = optimized_params[3*i]
        circles[i, 1] = optimized_params[3*i + 1]
        circles[i, 2] = optimized_params[3*i + 2]

    # Final validation check: ensure radii are not below a minimal value
    circles[:, 2] = np.maximum(circles[:, 2], EPSILON_FINAL_CLIP) # Use EPSILON_FINAL_CLIP
    
    return circles


# EVOLVE-BLOCK-END
