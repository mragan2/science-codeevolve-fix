# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize, Bounds, NonlinearConstraint, dual_annealing # Added dual_annealing
from numba import njit
# import time # Keep time for potential future performance profiling, though not used in current logic.

# --- Constants ---
N_CIRCLES = 32
BENCHMARK_SUM_RADII = 2.937944526205518
RANDOM_SEED = 42
MIN_RADIUS = 1e-7  # Minimum radius to avoid numerical issues and ensure positive radii
MAX_RADIUS = 0.5   # Maximum possible radius for a single circle in a unit square (0.5 if centered)

# --- Objective Function and its Gradient ---
def objective(params: np.ndarray) -> float:
    """
    Objective function to minimize: negative sum of radii.
    params: flattened array [x1, y1, r1, x2, y2, r2, ...]
    """
    radii = params[2::3]
    return -np.sum(radii)

@njit(cache=True)
def objective_grad(params: np.ndarray) -> np.ndarray:
    """
    Gradient of the objective function.
    The objective is -sum(r_i).
    Gradient w.r.t. x_i is 0, w.r.t. y_i is 0, w.r.t. r_i is -1.
    """
    grad = np.zeros_like(params, dtype=np.float64)
    grad[2::3] = -1.0
    return grad

# --- Numba-optimized Constraint Functions and their Jacobian ---
@njit(cache=True)
def _compute_boundary_constraints(x: np.ndarray, y: np.ndarray, r: np.ndarray, n: int) -> np.ndarray:
    """
    Computes boundary containment constraints:
    r_i <= x_i <= 1-r_i  => x_i - r_i >= 0  and  1 - x_i - r_i >= 0
    r_i <= y_i <= 1-r_i  => y_i - r_i >= 0  and  1 - y_i - r_i >= 0
    """
    cons_values = np.empty(4 * n, dtype=np.float64)
    for i in range(n):
        cons_values[i] = x[i] - r[i]          # x_i - r_i >= 0
        cons_values[n + i] = 1 - x[i] - r[i]  # 1 - x_i - r_i >= 0
        cons_values[2 * n + i] = y[i] - r[i]  # y_i - r_i >= 0
        cons_values[3 * n + i] = 1 - y[i] - r[i] # 1 - y_i - r_i >= 0
    return cons_values

@njit(cache=True)
def _compute_boundary_jacobian(x: np.ndarray, y: np.ndarray, r: np.ndarray, n: int) -> np.ndarray:
    """
    Computes the Jacobian for boundary containment constraints.
    """
    num_params = 3 * n
    num_boundary_cons = 4 * n
    jac = np.zeros((num_boundary_cons, num_params), dtype=np.float64)

    for i in range(n):
        # Constraint i: x_i - r_i >= 0
        jac[i, 3*i] = 1.0     # d(x_i - r_i)/dx_i
        jac[i, 3*i+2] = -1.0  # d(x_i - r_i)/dr_i

        # Constraint n + i: 1 - x_i - r_i >= 0
        jac[n + i, 3*i] = -1.0
        jac[n + i, 3*i+2] = -1.0

        # Constraint 2n + i: y_i - r_i >= 0
        jac[2*n + i, 3*i+1] = 1.0
        jac[2*n + i, 3*i+2] = -1.0

        # Constraint 3n + i: 1 - y_i - r_i >= 0
        jac[3*n + i, 3*i+1] = -1.0
        jac[3*n + i, 3*i+2] = -1.0
    return jac

@njit(cache=True)
def _compute_overlap_constraints(x: np.ndarray, y: np.ndarray, r: np.ndarray, n: int) -> np.ndarray:
    """
    Computes non-overlap constraints:
    (x_i - x_j)^2 + (y_i - y_j)^2 >= (r_i + r_j)^2
    => (x_i - x_j)^2 + (y_i - y_j)^2 - (r_i + r_j)^2 >= 0
    """
    num_pairs = n * (n - 1) // 2
    cons_values = np.empty(num_pairs, dtype=np.float64)
    k = 0
    for i in range(n):
        for j in range(i + 1, n):
            dist_sq = (x[i] - x[j])**2 + (y[i] - y[j])**2
            min_dist_sq = (r[i] + r[j])**2
            cons_values[k] = dist_sq - min_dist_sq
            k += 1
    return cons_values

@njit(cache=True)
def _compute_overlap_jacobian(x: np.ndarray, y: np.ndarray, r: np.ndarray, n: int) -> np.ndarray:
    """
    Computes the Jacobian for non-overlap constraints.
    """
    num_params = 3 * n
    num_pairs = n * (n - 1) // 2
    jac = np.zeros((num_pairs, num_params), dtype=np.float64)
    
    k = 0
    for i in range(n):
        for j in range(i + 1, n):
            # Constraint k: (x_i - x_j)^2 + (y_i - y_j)^2 - (r_i + r_j)^2 >= 0
            
            # Derivatives w.r.t. x_i, y_i, r_i
            jac[k, 3*i] = 2 * (x[i] - x[j])
            jac[k, 3*i+1] = 2 * (y[i] - y[j])
            jac[k, 3*i+2] = -2 * (r[i] + r[j])
            
            # Derivatives w.r.t. x_j, y_j, r_j
            jac[k, 3*j] = -2 * (x[i] - x[j]) # = 2 * (x_j - x_i)
            jac[k, 3*j+1] = -2 * (y[i] - y[j]) # = 2 * (y_j - y_i)
            jac[k, 3*j+2] = -2 * (r[i] + r[j])
            k += 1
    return jac

def combined_constraints_numba(params: np.ndarray) -> np.ndarray:
    """
    Combines all constraints into a single array for scipy.optimize.NonlinearConstraint.
    """
    n = N_CIRCLES
    x = params[0::3]
    y = params[1::3]
    r = params[2::3]

    boundary_cons = _compute_boundary_constraints(x, y, r, n)
    overlap_cons = _compute_overlap_constraints(x, y, r, n)

    return np.concatenate((boundary_cons, overlap_cons))

def combined_constraints_jacobian(params: np.ndarray) -> np.ndarray:
    """
    Combines Jacobians of all constraints into a single matrix.
    """
    n = N_CIRCLES
    x = params[0::3]
    y = params[1::3]
    r = params[2::3]

    boundary_jac = _compute_boundary_jacobian(x, y, r, n)
    overlap_jac = _compute_overlap_jacobian(x, y, r, n)

    return np.vstack((boundary_jac, overlap_jac))

# --- Initial Guess Generation ---
def generate_initial_guess_grid_perturbed(n: int, seed: int = None) -> np.ndarray:
    """
    Generates an initial guess based on a perturbed grid arrangement.
    """
    if seed is not None:
        np.random.seed(seed)

    # For N=32, a 4x8 grid is efficient
    rows = 4
    cols = 8
    
    # Calculate initial radius based on fitting to the grid dimensions
    # This ensures initial non-overlap for the grid points
    # Use 90% of the maximum possible radius for a grid to allow for perturbation and expansion
    initial_r = min(1.0 / (2.0 * cols), 1.0 / (2.0 * rows)) * 0.9 

    x_coords = np.linspace(initial_r, 1 - initial_r, cols)
    y_coords = np.linspace(initial_r, 1 - initial_r, rows)
    
    initial_params_list = []
    count = 0
    for j in range(rows):
        for i in range(cols):
            if count < n:
                initial_params_list.extend([x_coords[i], y_coords[j], initial_r])
                count += 1
    
    initial_params = np.array(initial_params_list, dtype=np.float64)

    # Add small random perturbation to break symmetry and aid optimization
    perturb_scale_xy = 0.01
    perturb_scale_r = 0.001
    
    initial_params[0::3] += np.random.uniform(-perturb_scale_xy, perturb_scale_xy, size=n) # Perturb x
    initial_params[1::3] += np.random.uniform(-perturb_scale_xy, perturb_scale_xy, size=n) # Perturb y
    initial_params[2::3] += np.random.uniform(-perturb_scale_r, perturb_scale_r, size=n)   # Perturb r
    
    # Ensure radii stay positive and within bounds after perturbation
    initial_params[2::3] = np.clip(initial_params[2::3], MIN_RADIUS, MAX_RADIUS)
    
    # Ensure centers stay within [r, 1-r] to satisfy initial boundary constraints
    for k in range(n):
        r_k = initial_params[k*3+2]
        initial_params[k*3] = np.clip(initial_params[k*3], r_k, 1 - r_k)
        initial_params[k*3+1] = np.clip(initial_params[k*3+1], r_k, 1 - r_k)

    return initial_params

def generate_initial_guess_hexagonal(n: int, seed: int = None) -> np.ndarray:
    """
    Generates an initial guess based on a hexagonal-like packing arrangement.
    Aims to place N circles in a roughly hexagonal grid, prioritizing central positions.
    """
    if seed is not None:
        np.random.seed(seed)

    # Estimate a base radius that would allow N circles in a hexagonal pattern.
    # For N=32, a radius around 0.07-0.08 is a reasonable starting point for packing.
    # Increased initial_r_base for a denser initial packing
    initial_r_base = 0.09 # Increased from 0.075, aiming for a denser initial packing
    
    candidate_x = []
    candidate_y = []
    candidate_r = []

    # Generate points in a hexagonal grid pattern over a slightly larger area than [0,1]
    # to ensure we capture enough points, especially near boundaries.
    # Use a slightly smaller radius for initial grid generation to allow for tight packing.
    r_gen = initial_r_base * 0.95 
    
    # Define a range slightly larger than [0,1] to generate candidate points
    # This helps ensure we can select N points that are fully within [0,1]
    x_gen_start = -r_gen * 2
    x_gen_end = 1 + r_gen * 2
    y_gen_start = -r_gen * 2
    y_gen_end = 1 + r_gen * 2

    row_idx = 0
    while True:
        y_center = r_gen + row_idx * r_gen * np.sqrt(3)
        if y_center > y_gen_end:
            break
        
        # Alternate x-offset for hexagonal pattern
        x_start_offset = 0.0 if row_idx % 2 == 0 else r_gen
        
        col_idx = 0
        while True:
            x_center = r_gen + x_start_offset + col_idx * (2 * r_gen)
            if x_center > x_gen_end:
                break
            
            candidate_x.append(x_center)
            candidate_y.append(y_center)
            candidate_r.append(r_gen) # All start with the same radius for initial grid
            col_idx += 1
        row_idx += 1
    
    # Filter candidates to be those where a circle of radius r_gen could fit within [0,1]
    valid_indices = []
    for i in range(len(candidate_x)):
        if (candidate_x[i] >= r_gen and candidate_x[i] <= 1 - r_gen and
            candidate_y[i] >= r_gen and candidate_y[i] <= 1 - r_gen):
            valid_indices.append(i)
            
    # If we have more valid candidates than needed, select the N_CIRCLES closest to the center.
    # This prioritizes a compact, central packing.
    if len(valid_indices) > n:
        distances_to_center = np.sqrt((np.array(candidate_x)[valid_indices] - 0.5)**2 + 
                                       (np.array(candidate_y)[valid_indices] - 0.5)**2)
        sorted_indices = np.argsort(distances_to_center)
        selected_indices = np.array(valid_indices)[sorted_indices[:n]]
    else:
        selected_indices = np.array(valid_indices)
    
    # Initialize params with selected candidates
    initial_params = np.empty(n * 3, dtype=np.float64)
    if len(selected_indices) > 0:
        initial_params[0::3][:len(selected_indices)] = np.array(candidate_x)[selected_indices]
        initial_params[1::3][:len(selected_indices)] = np.array(candidate_y)[selected_indices]
        initial_params[2::3][:len(selected_indices)] = np.array(candidate_r)[selected_indices]
    
    # If fewer than N circles were generated (e.g., r_gen was too large for the square),
    # fill the remaining spots with small random circles.
    if len(selected_indices) < n:
        for k in range(len(selected_indices), n):
            r_fill = MIN_RADIUS * 5 # Use a small radius for filler circles
            x_fill = np.random.uniform(r_fill, 1 - r_fill)
            y_fill = np.random.uniform(r_fill, 1 - r_fill)
            initial_params[k*3] = x_fill
            initial_params[k*3+1] = y_fill
            initial_params[k*3+2] = r_fill

    # Add small random perturbation to break symmetry and aid optimization
    perturb_scale_xy = 0.01 * initial_r_base # Relative perturbation
    perturb_scale_r = 0.005 * initial_r_base # Increased from 0.001 * initial_r_base for more radius variation
    
    initial_params[0::3] += np.random.uniform(-perturb_scale_xy, perturb_scale_xy, size=n)
    initial_params[1::3] += np.random.uniform(-perturb_scale_xy, perturb_scale_xy, size=n)
    initial_params[2::3] += np.random.uniform(-perturb_scale_r, perturb_scale_r, size=n)
    
    # Ensure radii stay positive and within overall bounds after perturbation
    initial_params[2::3] = np.clip(initial_params[2::3], MIN_RADIUS, MAX_RADIUS)
    
    # Ensure centers stay within [r, 1-r] to satisfy initial boundary constraints
    for k in range(n):
        r_k = initial_params[k*3+2]
        initial_params[k*3] = np.clip(initial_params[k*3], r_k, 1 - r_k)
        initial_params[k*3+1] = np.clip(initial_params[k*3+1], r_k, 1 - r_k)

    return initial_params


def generate_initial_guess_random_perturbed(n: int, seed: int = None) -> np.ndarray:
    """
    Generates a random initial guess with small radii and perturbation.
    """
    if seed is not None:
        np.random.seed(seed)
    
    initial_params = np.empty(n * 3, dtype=np.float64)
    initial_r_base = 0.01 # Start with a small base radius
    
    for k in range(n):
        # Random x, y coordinates
        x_k = np.random.uniform(0.0, 1.0)
        y_k = np.random.uniform(0.0, 1.0)
        # Random radius around initial_r_base, ensuring positivity
        r_k = initial_r_base + np.random.uniform(-initial_r_base/2, initial_r_base/2)
        r_k = np.clip(r_k, MIN_RADIUS, MAX_RADIUS)

        # Ensure x, y are within bounds for the given r
        initial_params[k*3] = np.clip(x_k, r_k, 1 - r_k)
        initial_params[k*3+1] = np.clip(y_k, r_k, 1 - r_k)
        initial_params[k*3+2] = r_k
    
    return initial_params


# --- Main Constructor Function ---
def circle_packing32() -> np.ndarray:
    """
    Places 32 non-overlapping circles in the unit square in order to maximize the sum of radii.
    Employs a multi-start local optimization strategy using SLSQP with Numba-optimized constraints.

    Returns:
        circles: np.array of shape (32,3), where the i-th row (x,y,r) stores the (x,y) coordinates of the i-th circle of radius r.
    """
    
    # Define general bounds for x, y, r for the optimization variables
    # x, y can be from 0 to 1, r from MIN_RADIUS to MAX_RADIUS.
    # The NonlinearConstraint will enforce tighter bounds (e.g., x >= r).
    lower_bounds_array = np.zeros(N_CIRCLES * 3, dtype=np.float64)
    upper_bounds_array = np.ones(N_CIRCLES * 3, dtype=np.float64)
    lower_bounds_array[2::3] = MIN_RADIUS # Radii must be positive
    upper_bounds_array[2::3] = MAX_RADIUS # Radii cannot exceed 0.5
    param_bounds = Bounds(lower_bounds_array, upper_bounds_array)

    # Define the NonlinearConstraint object for all boundary and non-overlap conditions.
    # All constraint values must be >= 0.
    num_boundary_constraints = 4 * N_CIRCLES
    num_overlap_constraints = N_CIRCLES * (N_CIRCLES - 1) // 2
    total_constraints = num_boundary_constraints + num_overlap_constraints
    
    nonlinear_constraint = NonlinearConstraint(
        combined_constraints_numba,
        lb=np.zeros(total_constraints, dtype=np.float64), # Lower bound for constraints
        ub=np.full(total_constraints, np.inf, dtype=np.float64), # Upper bound for constraints (no upper limit)
        jac=combined_constraints_jacobian # Provide the analytical Jacobian
    )

    best_sum_radii = -np.inf
    best_circles = np.zeros((N_CIRCLES, 3), dtype=np.float64) # Initialize with zeros as fallback

    np.random.seed(RANDOM_SEED) # Seed for overall random operations

    # --- Stage 1: Global Optimization using Dual Annealing ---
    # Dual Annealing is known for robust global search and can integrate local search with gradients.
    
    # Dual Annealing options
    da_options = {
        'maxiter': 1000, # Max global search iterations. Reduced slightly from DE's 1500, as DA's local searches are more efficient.
        'initial_temp': 9600.0, # Heuristic: 100 * D, where D=96 parameters. Higher for more exploration.
        'restart_temp_ratio': 2e-5, # Controls how quickly temperature drops
        'visit_limit': 100, # Number of times to visit the same point
        'seed': RANDOM_SEED,
        'no_local_search': False, # Enable local search within DA
    }
    
    # Minimizer arguments for the local search step within dual_annealing
    minimizer_kwargs = {
        'method': 'SLSQP',
        'bounds': param_bounds,
        'constraints': nonlinear_constraint,
        'jac': objective_grad, # Provide objective gradient to local SLSQP
        'options': {'maxiter': 500, 'ftol': 1e-6, 'disp': False} # Local search options
    }
    
    # print("Starting Dual Annealing for global search...")
    da_result = None
    try:
        da_result = dual_annealing(
            objective,
            param_bounds,
            jac=objective_grad, # Provide objective gradient to dual_annealing
            minimizer_kwargs=minimizer_kwargs,
            **da_options
        )
        # print(f"Dual Annealing finished. Success: {da_result.success}, Sum Radii: {-da_result.fun:.6f}")
        
        if da_result.success:
            best_sum_radii = -da_result.fun
            best_circles = da_result.x.reshape(N_CIRCLES, 3)
        # elif not da_result.success:
            # print(f"Dual Annealing did not converge successfully: {da_result.message}")

    except Exception as e:
        # print(f"Error during Dual Annealing: {e}")
        da_result = None # Ensure da_result is None if an error occurs

    # --- Stage 2: Local Refinement using Multi-start SLSQP ---
    # If DA provided a good starting point, use it. Otherwise, rely on improved initial guess strategies.
    
    num_slsqp_runs = 15 # Number of independent SLSQP runs for local refinement
    
    # Add the new hexagonal initial guess strategy
    initial_guess_strategies = [
        generate_initial_guess_grid_perturbed,
        generate_initial_guess_hexagonal, # New hexagonal initial guess
        generate_initial_guess_random_perturbed,
    ]

    # Initialize SLSQP starting points: include DA's best result if successful
    slsqp_initial_guesses = []
    if da_result is not None and da_result.success:
        slsqp_initial_guesses.append(best_circles.flatten())
    
    # Generate additional initial guesses to fill up to num_slsqp_runs
    for run_idx in range(num_slsqp_runs - len(slsqp_initial_guesses)):
        strategy_func = initial_guess_strategies[run_idx % len(initial_guess_strategies)]
        # Use a unique seed for each run to ensure diversity in initial perturbations
        current_seed = RANDOM_SEED + run_idx + 100 # Offset seed for SLSQP runs
        slsqp_initial_guesses.append(strategy_func(N_CIRCLES, seed=current_seed))
        
    # SLSQP options - even more aggressive for fine-tuning
    slsqp_options = {'maxiter': 5000, 'ftol': 1e-7, 'disp': False} # Increased maxiter, adjusted ftol for precision

    for run_idx, x0_slsqp in enumerate(slsqp_initial_guesses):
        try:
            res_slsqp = minimize(
                objective,
                x0_slsqp,
                method='SLSQP',
                bounds=param_bounds,
                constraints=[nonlinear_constraint],
                jac=objective_grad, # Provide objective gradient to SLSQP
                options=slsqp_options
            )

            if res_slsqp.success and -res_slsqp.fun > best_sum_radii:
                best_sum_radii = -res_slsqp.fun
                best_circles = res_slsqp.x.reshape(N_CIRCLES, 3)
            # elif not res_slsqp.success:
                # print(f"SLSQP Run {run_idx} (Strategy {strategy_func.__name__}): Optimization failed: {res_slsqp.message}")

        except Exception as e:
            # print(f"SLSQP Run {run_idx}: An error occurred: {e}")
            pass # Continue to the next run

    # Fallback: If no successful optimization was found (e.g., all runs failed or best_sum_radii was never updated),
    # return a reasonable default, like the hexagonal configuration.
    if best_sum_radii == -np.inf:
        # print("Warning: No successful optimization found, returning default hexagonal configuration.")
        fallback_params = generate_initial_guess_hexagonal(N_CIRCLES, seed=RANDOM_SEED)
        best_circles = fallback_params.reshape(N_CIRCLES, 3)
        # Ensure radii are strictly positive for the fallback
        best_circles[:, 2] = np.maximum(best_circles[:, 2], MIN_RADIUS)

    return best_circles


# EVOLVE-BLOCK-END
