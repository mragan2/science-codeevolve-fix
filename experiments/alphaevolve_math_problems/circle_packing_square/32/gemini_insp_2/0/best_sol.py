# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize, differential_evolution, NonlinearConstraint
from numba import njit # Import numba for JIT compilation

@njit(cache=True, fastmath=True)
def _objective_jit(params: np.ndarray, n_circles: int) -> float:
    """
    Objective function to minimize: negative sum of radii.
    This JIT-compiled version directly sums radii from the flattened array.
    
    Args:
        params: A 1D numpy array representing all circle parameters.
        n_circles: The number of circles (unused, but kept for signature consistency).
    Returns:
        The negative sum of radii.
    """
    # Radii are at indices 2, 5, 8, ... in the flattened array
    return -np.sum(params[2::3]) # Direct slicing is efficient and doesn't require unpacking

@njit(cache=True, fastmath=True)
def _evaluate_constraints(circles_flat_array: np.ndarray, n: int) -> np.ndarray:
    """
    Evaluates all containment and non-overlap constraints.
    Returns an array of values, each of which should be >= 0 for feasibility.
    
    This function is JIT-compiled with Numba for performance.
    
    Args:
        circles_flat_array: A 1D numpy array representing all circle parameters
                            in the format [x1, y1, r1, x2, y2, r2, ...].
        n: The number of circles.
    Returns:
        A 1D numpy array where each element is the value of a constraint.
        For a feasible solution, all these values must be >= 0.
    """
    # Reshape the flat array back to (n, 3) for easier access
    circles = circles_flat_array.reshape(n, 3)

    num_overlap_constraints = n * (n - 1) // 2
    num_containment_constraints = n * 4 # x-r>=0, 1-x-r>=0, y-r>=0, 1-y-r>=0
    
    constraint_values = np.zeros(num_containment_constraints + num_overlap_constraints, dtype=np.float64)
    k = 0

    # Containment constraints: ri <= xi <= 1-ri and ri <= yi <= 1-ri
    # These translate to:
    # 1. xi - ri >= 0
    # 2. 1 - xi - ri >= 0
    # 3. yi - ri >= 0
    # 4. 1 - yi - ri >= 0
    for i in range(n):
        xi, yi, ri = circles[i]
        constraint_values[k] = xi - ri
        k += 1
        constraint_values[k] = 1 - xi - ri
        k += 1
        constraint_values[k] = yi - ri
        k += 1
        constraint_values[k] = 1 - yi - ri
        k += 1
    
    # Non-overlap constraints: (xi-xj)^2 + (yi-yj)^2 >= (ri + rj)^2
    # These translate to:
    # (xi-xj)^2 + (yi-yj)^2 - (ri + rj)^2 >= 0
    for i in range(n):
        for j in range(i + 1, n): # Iterate over unique pairs (i < j)
            xi, yi, ri = circles[i]
            xj, yj, rj = circles[j]
            dist_sq = (xi - xj)**2 + (yi - yj)**2
            min_dist_sq = (ri + rj)**2
            constraint_values[k] = dist_sq - min_dist_sq
            k += 1
    return constraint_values

def generate_hex_initial_guess(n_circles: int, rng: np.random.Generator) -> np.ndarray:
    """
    Generates an initial configuration of circles arranged in an approximate
    hexagonal packing pattern within the unit square. This provides a good
    starting point for the optimizer.
    
    For N=32 circles, a 6-row pattern with alternating 6, 5, 6, 5, 6, 4 circles
    is used to achieve a dense, valid starting configuration.

    Args:
        n_circles: The total number of circles to generate.
        rng: A numpy random number generator for perturbations.
    
    Returns:
        A 1D numpy array representing the initial circle parameters
        [x1, y1, r1, x2, y2, r2, ...].
    """
    circles_list = []
    
    # Define row structure for 32 circles to achieve a dense hexagonal packing
    # This pattern sums to 32 circles: 6 + 5 + 6 + 5 + 6 + 4 = 32
    row_counts = [6, 5, 6, 5, 6, 4]
    
    if n_circles != sum(row_counts):
        # Fallback to a simpler grid if the specific pattern isn't for N_CIRCLES
        print(f"Warning: Hexagonal pattern hardcoded for 32 circles, but {n_circles} requested. Falling back to simple grid.")
        grid_dim = int(np.ceil(np.sqrt(n_circles)))
        r_fallback = 0.5 / grid_dim if grid_dim > 0 else 0.25 # Ensure r_fallback is positive
        r_fallback = np.clip(r_fallback, 1e-6, 0.5)

        for i in range(n_circles):
            row = i // grid_dim
            col = i % grid_dim
            x = (col + 0.5) / grid_dim
            y = (row + 0.5) / grid_dim
            circles_list.append([x, y, r_fallback])
        
        fallback_guess_flat = np.array(circles_list).flatten()
        # Add small random perturbations to the fallback grid to break symmetry
        fallback_guess_flat += rng.normal(0, 0.0005, fallback_guess_flat.shape)
        # Ensure radii are positive and positions are within [0,1] after perturbation
        for i in range(n_circles):
            fallback_guess_flat[i*3+2] = np.clip(fallback_guess_flat[i*3+2], 1e-6, 0.5) # Radii
            fallback_guess_flat[i*3] = np.clip(fallback_guess_flat[i*3], 0.0, 1.0) # X
            fallback_guess_flat[i*3+1] = np.clip(fallback_guess_flat[i*3+1], 0.0, 1.0) # Y
        return fallback_guess_flat


    num_rows = len(row_counts)
    max_cols = max(row_counts) # Max circles in any row is 6

    # Calculate an initial radius (r_init) that allows the pattern to fit within the unit square
    r_h_limit = 1.0 / (2.0 * max_cols) 
    r_v_limit = 1.0 / (2.0 + (num_rows - 1) * np.sqrt(3)) 
    r_init = min(r_h_limit, r_v_limit)
    
    if r_init < 1e-4: # Ensure a reasonable starting radius
        r_init = 0.05 

    pattern_actual_width = max_cols * 2 * r_init
    pattern_actual_height = 2 * r_init + (num_rows - 1) * np.sqrt(3) * r_init

    # Center the pattern within the unit square
    global_x_offset = (1.0 - pattern_actual_width) / 2.0
    global_y_offset = (1.0 - pattern_actual_height) / 2.0
    
    current_y = r_init + global_y_offset

    circle_idx = 0
    for row_idx, num_in_row in enumerate(row_counts):
        # Shift odd rows horizontally for hexagonal packing
        row_x_start_offset = 0.0
        if row_idx % 2 != 0: # Odd-indexed rows (1, 3, 5) are shifted
            row_x_start_offset = r_init 
            
        # Adjust horizontal offset to center the current row relative to the max width
        row_width = num_in_row * 2 * r_init
        actual_row_x_offset = (max_cols * 2 * r_init - row_width) / 2.0
        
        current_x = r_init + global_x_offset + row_x_start_offset + actual_row_x_offset

        for _ in range(num_in_row):
            if circle_idx < n_circles:
                circles_list.append([current_x, current_y, r_init])
                circle_idx += 1
            current_x += 2 * r_init
        
        current_y += np.sqrt(3) * r_init

    # Add small random perturbation to break perfect symmetry for the optimizer
    initial_guess_flat = np.array(circles_list).flatten()
    initial_guess_flat += rng.normal(0, 0.0005, initial_guess_flat.shape) # Smaller perturbation for a good initial guess

    # Ensure radii are positive and positions are within [0,1]
    for i in range(n_circles):
        initial_guess_flat[i*3+2] = np.clip(initial_guess_flat[i*3+2], 1e-6, 0.5) # Radii
        initial_guess_flat[i*3] = np.clip(initial_guess_flat[i*3], 0.0, 1.0) # X
        initial_guess_flat[i*3+1] = np.clip(initial_guess_flat[i*3+1], 0.0, 1.0) # Y
    
    return initial_guess_flat


def circle_packing32() -> np.ndarray:
    """
    Generates an optimal arrangement of exactly 32 non-overlapping circles
    within a unit square, maximizing the sum of their radii.

    This function employs a three-stage optimization strategy:
    1. Global optimization using `differential_evolution` seeded with a dense
       hexagonal packing pattern to find a promising solution region.
    2. Local refinement using `minimize` ('SLSQP') to find the local optimum.
    3. Iterated Local Search, which repeatedly perturbs the best solution and
       re-runs the local optimizer to escape local optima and find better solutions.

    Returns:
        circles: np.array of shape (32,3), where the i-th row (x,y,r) stores
                 the (x,y) coordinates of the i-th circle of radius r.
    """
    n = 32
    
    # Use a fixed seed for reproducibility.
    rng = np.random.default_rng(seed=42)

    # Define bounds for x, y, r for each circle
    bounds = []
    for _ in range(n):
        bounds.append((0.0, 1.0))  # x coordinate
        bounds.append((0.0, 1.0))  # y coordinate
        bounds.append((1e-6, 0.5))  # r (radius) - using 1e-6 for smaller minimum
    
    # Define constraints using NonlinearConstraint for both DE and SLSQP
    num_overlap_constraints = n * (n - 1) // 2
    num_containment_constraints = n * 4
    total_constraints = num_containment_constraints + num_overlap_constraints
    lb_constraints = np.zeros(total_constraints)
    ub_constraints = np.full(total_constraints, np.inf)
    
    # Pass 'n' as an argument to the jitted constraint function
    constraints = (NonlinearConstraint(lambda params: _evaluate_constraints(params, n), lb_constraints, ub_constraints),)

    # Generate an initial population for DE, seeding it with a hexagonal guess.
    initial_population_size = 80 # Increased population size for broader exploration
    initial_population = np.zeros((initial_population_size, n * 3))
    
    # First member of population is the hexagonal guess
    initial_population[0] = generate_hex_initial_guess(n, rng)
    
    # Populate the rest with random variations, ensuring bounds are respected
    for i in range(1, initial_population_size):
        for j, (lower, upper) in enumerate(bounds):
            if j % 3 == 2:  # Radius
                initial_population[i, j] = rng.uniform(0.01, 0.5) # Radii can start relatively large
            else:  # x or y coordinate
                initial_population[i, j] = rng.uniform(lower, upper)
        # Add a small perturbation to random initial guesses too, ensuring determinism with rng
        initial_population[i] += rng.normal(0, 0.001, n*3)
        # Clip to bounds
        for j, (lower, upper) in enumerate(bounds):
            initial_population[i, j] = np.clip(initial_population[i, j], lower, upper)

    # --- Stage 1: Global Optimization with Differential Evolution ---
    print("Starting Differential Evolution (Stage 1)...")
    de_result = differential_evolution(
        func=_objective_jit, # Use JIT-compiled objective
        args=(n,), # Pass n as an argument to the objective function
        bounds=bounds,
        constraints=constraints,
        maxiter=2500, # Increased maxiter for better exploration
        popsize=initial_population_size, # popsize should match initial_population_size when init is provided
        seed=42,
        workers=-1, # Use all available CPU cores for parallelization
        disp=True,
        polish=False, # Polish will be done by SLSQP
        init=initial_population,
        tol=1e-4      # Add tolerance for earlier convergence if DE plateaus
    )
    print(f"Differential Evolution finished. Best negative sum of radii: {de_result.fun}")
    print(f"Initial sum of radii from DE: {-de_result.fun:.12f}")

    # --- Stage 2: Initial Local Refinement with SLSQP ---
    print("\nStarting initial SLSQP local refinement (Stage 2)...")
    slsqp_result = minimize(
        fun=_objective_jit, # Use JIT-compiled objective
        x0=de_result.x,
        args=(n,), # Pass n as an argument to the objective function
        method='SLSQP',
        bounds=bounds,
        constraints=constraints,
        options={'maxiter': 1500, 'ftol': 1e-10, 'disp': True, 'eps': 1e-8} # Tighter ftol, increased maxiter, slightly relaxed eps
    )
    
    best_solution = slsqp_result.x
    best_fun = slsqp_result.fun
    print(f"Initial SLSQP refined sum of radii: {-best_fun:.12f}")

    # --- Stage 3: Iterated Local Search with Enhanced Perturbation ---
    print("\nStarting Iterated Local Search (Stage 3)...")
    n_perturb_iterations = 80 # Slightly increased number of refinement cycles
    
    # Base noise scale: smaller noise for radii, larger for positions
    base_noise_scale = np.tile([0.002, 0.002, 0.001], n)
    # Stronger noise scale for selected circles
    strong_noise_scale = np.tile([0.02, 0.02, 0.01], n) # 10x stronger perturbation

    num_circles_to_strongly_perturb = 8 # Number of circles to apply stronger perturbation to

    for i in range(n_perturb_iterations):
        # Dynamically adjust noise scale over iterations using exponential decay
        # This keeps perturbation strength higher for longer compared to linear decay.
        decay_factor = 0.99
        current_noise_multiplier = decay_factor ** i
        current_base_noise_scale = base_noise_scale * current_noise_multiplier
        current_strong_noise_scale = strong_noise_scale * current_noise_multiplier

        perturbed_x0 = best_solution.copy()
        
        # Apply base noise to all circles
        perturbed_x0 += rng.normal(0, current_base_noise_scale)

        # Select a random subset of circles for stronger perturbation
        strong_perturb_indices = rng.choice(n, num_circles_to_strongly_perturb, replace=False)
        for circle_idx in strong_perturb_indices:
            start_idx = circle_idx * 3
            end_idx = start_idx + 3
            # Apply strong noise to the selected subset
            perturbed_x0[start_idx:end_idx] = best_solution[start_idx:end_idx] + rng.normal(0, current_strong_noise_scale[start_idx:end_idx])
        
        # Ensure the perturbed solution respects the bounds (clipping)
        for j, (lower, upper) in enumerate(bounds):
            perturbed_x0[j] = np.clip(perturbed_x0[j], lower, upper)

        # Re-run the local optimizer from the perturbed starting point
        refinement_result = minimize(
            fun=_objective_jit, # Use JIT-compiled objective
            x0=perturbed_x0,
            args=(n,), # Pass n as an argument to the objective function
            method='SLSQP',
            bounds=bounds,
            constraints=constraints,
            options={'maxiter': 1200, 'ftol': 1e-11, 'disp': False, 'eps': 1e-9} # Tighter ftol, slightly increased maxiter, slightly relaxed eps
        )
        
        # If the new result is a valid improvement, accept it
        if refinement_result.success and refinement_result.fun < best_fun:
            # Additionally, check constraint satisfaction for the new best solution more rigorously
            constraint_check_tolerance = 1e-7
            final_constraint_values = _evaluate_constraints(refinement_result.x, n)
            if np.all(final_constraint_values >= -constraint_check_tolerance):
                best_fun = refinement_result.fun
                best_solution = refinement_result.x
                print(f"    Found new best solution! Sum of radii: {-best_fun:.12f} (Iter {i+1}/{n_perturb_iterations})")

    print(f"\nIterated Local Search finished. Best sum of radii found: {-best_fun:.12f}")

    # --- Stage 4: Final Polish ---
    # Run one last, highly precise local optimization from the best solution found so far.
    print("\nStarting final high-precision polish (Stage 4)...")
    final_polish_result = minimize(
        fun=_objective_jit,
        x0=best_solution,
        args=(n,),
        method='SLSQP',
        bounds=bounds,
        constraints=constraints,
        options={'maxiter': 2000, 'ftol': 1e-12, 'disp': True, 'eps': 1e-10} # Very strict parameters, slightly relaxed eps
    )

    if final_polish_result.success and final_polish_result.fun < best_fun:
        best_solution = final_polish_result.x
        best_fun = final_polish_result.fun
        print(f"Final polish improved solution. Final sum of radii: {-best_fun:.12f}")
    else:
        print(f"Final polish did not improve solution. Using ILS result. Final sum of radii: {-best_fun:.12f}")
    
    # Reshape the final optimized parameters into the (n, 3) format (x, y, r)
    circles = best_solution.reshape(n, 3)

    # Ensure radii are strictly non-negative after optimization (can sometimes be tiny negative due to float precision)
    circles[:, 2] = np.maximum(1e-6, circles[:, 2]) # Ensure minimum radius of 1e-6
    
    # Final clamping to ensure all circles are strictly within bounds after optimization
    for i in range(n):
        r = circles[i, 2]
        circles[i, 0] = np.clip(circles[i, 0], r, 1 - r)
        circles[i, 1] = np.clip(circles[i, 1], r, 1 - r)

    return circles


# EVOLVE-BLOCK-END
