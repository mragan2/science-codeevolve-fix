# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import differential_evolution, minimize
from numba import njit

# Constants for the problem
N_CIRCLES = 26
UNIT_SQUARE_SIDE = 1.0
GLOBAL_RANDOM_SEED = 42 # For reproducibility

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
    """Generates a structured initial guess by placing circles on a grid."""
    params = np.zeros(n_circles * 3)
    grid_rows = int(np.ceil(n_circles / grid_cols))
    
    # Create grid points slightly inside the unit square to start
    x_steps = np.linspace(0.1, UNIT_SQUARE_SIDE - 0.1, grid_cols)
    y_steps = np.linspace(0.1, UNIT_SQUARE_SIDE - 0.1, grid_rows)
    
    # Use a more aggressive initial radius, closer to the expected average for 26 circles.
    # This helps DE start with larger circles and potentially converge faster to a better sum.
    # For N=26, the average radius in optimal packings is around 0.1, so 0.095 is a good starting point (from Insp 2).
    initial_radius = 0.095

    idx = 0
    rng = np.random.default_rng(GLOBAL_RANDOM_SEED)
    for r in range(grid_rows):
        for c in range(grid_cols):
            if idx < n_circles:
                params[3*idx] = x_steps[c]
                params[3*idx+1] = y_steps[r]
                # Add a slightly larger jitter to radii to break symmetry and encourage exploration (from Insp 2)
                params[3*idx+2] = initial_radius + rng.uniform(-7e-4, 7e-4)
                idx += 1
    return params

def objective_with_penalty(params, penalty_factor=1e5, return_violations=False):
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
    Employs a two-phase optimization: Differential Evolution for global search,
    followed by SLSQP for local refinement.

    Returns:
        circles: np.array of shape (26,3), where the i-th row (x,y,r) stores the (x,y) coordinates of the i-th circle of radius r.
    """
    np.random.seed(GLOBAL_RANDOM_SEED)

    # Define bounds for each parameter (x, y, r). Reverted radius upper bound to 0.5 for flexibility.
    bounds = [(0.0, UNIT_SQUARE_SIDE), (0.0, UNIT_SQUARE_SIDE), (1e-6, 0.5)] * N_CIRCLES

    # Create a good initial guess and seed the DE population with it.
    popsize = 35 # Increased popsize for better global exploration (from Insp 2)
    rng = np.random.default_rng(GLOBAL_RANDOM_SEED)
    
    # Create an initial population using uniform random sampling within the bounds.
    # This corrected initialization ensures all random members are within their respective bounds.
    initial_population = np.zeros((popsize, len(bounds)))
    for i in range(len(bounds)):
        low, high = bounds[i]
        initial_population[:, i] = rng.uniform(low, high, size=popsize)

    # Replace the first member of the population with our smart grid-based guess.
    grid_guess = generate_initial_guess(N_CIRCLES)
    initial_population[0, :] = grid_guess

    # Phase 1: Global Optimization with Differential Evolution
    # DE explores the search space broadly, starting from a "warm" initial population.
    print("Starting Differential Evolution (Phase 1)...")
    de_result = differential_evolution(
        func=objective_with_penalty,
        bounds=bounds,
        strategy='best1bin',     # A robust strategy
        maxiter=5000,            # Increased maxiter for more thorough global search (from Insp 2)
        popsize=popsize,
        tol=0.001,               # Tolerance for convergence (matching Insp 2)
        mutation=(0.5, 1.0),     # Mutation range
        recombination=0.7,       # Crossover probability
        seed=GLOBAL_RANDOM_SEED,
        disp=True,               # Display progress
        workers=-1,              # Use all available CPU cores for parallelization
        init=initial_population  # Provide the smart initial population
    )
    
    # Analyze DE results
    unpenalized_obj_de, violations_de = objective_with_penalty(de_result.x, penalty_factor=0, return_violations=True)
    print(f"Differential Evolution finished. Best objective (penalized): {de_result.fun:.4f}")
    print(f"Sum of radii after DE (unpenalized): {-unpenalized_obj_de:.6f}")
    print(f"Total constraint violation after DE: {violations_de.sum():.6f}")

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
        options={'maxiter': 4000, 'ftol': 1e-9, 'disp': True} # Reduced maxiter and loosened ftol for initial refinement
    )
    best_params = best_slsqp_result.x
    best_sum_radii = -best_slsqp_result.fun if best_slsqp_result.success else -np.inf

    print(f"Initial SLSQP finished. Sum of radii: {best_sum_radii:.12f}")

    # Phase 3: Iterated Local Search (ILS)
    print("\nStarting Iterated Local Search (Phase 3)...")
    num_ils_iterations = 75 # Reduced number of perturbation-refinement cycles to balance speed and quality
    # IMPORTANT: Use a different seed for each ILS iteration to ensure diverse perturbations
    rng_perturb_base_seed = GLOBAL_RANDOM_SEED + 100 # Base seed for ILS perturbations

    noise_scale_xy = 0.015 # Retained aggressive perturbation for coordinates
    noise_scale_r = 0.01  # Retained aggressive perturbation for radii

    for i in range(num_ils_iterations):
        print(f"\nILS Iteration {i+1}/{num_ils_iterations}...")
        
        # Perturb the current best solution using a unique seed for each iteration
        rng_perturb = np.random.default_rng(rng_perturb_base_seed + i)
        perturbed_guess = np.copy(best_params)
        
        # Apply noise based on current best solution
        perturbed_guess[0::3] += rng_perturb.normal(0, noise_scale_xy, N_CIRCLES)
        perturbed_guess[1::3] += rng_perturb.normal(0, noise_scale_xy, N_CIRCLES)
        perturbed_guess[2::3] += rng_perturb.normal(0, noise_scale_r, N_CIRCLES)

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
            options={'maxiter': 2000, 'ftol': 1e-9, 'disp': False} # Reduced maxiter and loosened ftol for ILS SLSQP
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

    # Analyze final results
    final_sum_radii = -final_result.fun
    _, final_violations = objective_with_penalty(final_result.x, penalty_factor=0, return_violations=True)
    
    print(f"\nSLSQP finished. Final objective: {final_result.fun:.6f}")
    print(f"Final sum of radii: {final_sum_radii:.12f}")
    print(f"SLSQP success: {final_result.success}")
    print(f"SLSQP message: {final_result.message}")
    print(f"Total constraint violation after SLSQP: {final_violations.sum():.10f}")

    # Extract circles from the optimized parameters
    optimized_params = final_result.x
    circles = np.zeros((N_CIRCLES, 3), dtype=np.float64)
    for i in range(N_CIRCLES):
        circles[i, 0] = optimized_params[3*i]
        circles[i, 1] = optimized_params[3*i + 1]
        circles[i, 2] = optimized_params[3*i + 2]

    # Final validation check
    circles[:, 2] = np.maximum(circles[:, 2], 1e-9)
    
    return circles


# EVOLVE-BLOCK-END
