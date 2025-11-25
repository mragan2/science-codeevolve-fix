# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize, NonlinearConstraint
import joblib
import matplotlib.pyplot as plt
import time

N_CIRCLES = 26
GLOBAL_SEED = 42 # For reproducibility

# --- Helper functions to extract parameters from the flattened state vector ---
def get_x(params):
    return params[0::3]

def get_y(params):
    return params[1::3]

def get_r(params):
    return params[2::3]

# --- Objective function to minimize ---
def objective(params):
    """
    Objective function: negative sum of radii.
    We minimize this to maximize the sum of radii.
    """
    radii = get_r(params)
    return -np.sum(radii)

# --- Constraint functions, all of which must be >= 0 ---
def constraints_fun(params):
    """
    Returns an array of all constraint values. All values must be >= 0 for feasibility.
    1. Containment constraints: ri <= xi <= 1-ri, ri <= yi <= 1-ri
    2. Non-overlap constraints: distance(i,j) >= ri + rj
    """
    x = get_x(params)
    y = get_y(params)
    r = get_r(params)
    n = len(x)

    # 1. Containment constraints (4*N constraints) - fully vectorized
    # Ensure x_i - r_i >= 0, 1 - x_i - r_i >= 0, y_i - r_i >= 0, 1 - y_i - r_i >= 0
    containment_constraints = np.concatenate([
        x - r,
        1 - x - r,
        y - r,
        1 - y - r
    ])

    # 2. Non-overlap constraints (N*(N-1)/2 constraints) - vectorized
    # Formulate as (x_i - x_j)^2 + (y_i - y_j)^2 - (r_i + r_j)^2 >= 0
    
    # Reshape x, y, r to (N, 1) for broadcasting to create all-pairs differences/sums
    x_col = x[:, np.newaxis]
    y_col = y[:, np.newaxis]
    r_col = r[:, np.newaxis]

    # Compute all-pairs differences for x and y coordinates
    dx = x_col - x_col.T
    dy = y_col - y_col.T

    # Compute squared Euclidean distances between all centers
    dist_sq_matrix = dx**2 + dy**2

    # Compute squared sum of radii for all pairs (r_i + r_j)^2
    sum_r_sq_matrix = (r_col + r_col.T)**2

    # Extract unique pairs (upper triangle excluding the diagonal)
    idx = np.triu_indices(n, k=1) # k=1 excludes diagonal to avoid self-comparison
    non_overlap_constraints = dist_sq_matrix[idx] - sum_r_sq_matrix[idx]
    
    # Combine all constraint values into a single array
    return np.concatenate([containment_constraints, non_overlap_constraints])

# --- Bounds for individual parameters (x, y, r) ---
def get_parameter_bounds(n_circles):
    """
    Returns a list of (min, max) tuples for each parameter (x, y, r) of all circles.
    x in [0, 1], y in [0, 1], r in [1e-6, 0.5].
    """
    bounds = []
    for _ in range(n_circles):
        bounds.append((0.0, 1.0)) # x-coordinate bounds
        bounds.append((0.0, 1.0)) # y-coordinate bounds
        # Radius bounds: must be positive (1e-6 to avoid numerical issues with 0) and <= 0.5 (max for unit square)
        bounds.append((1e-6, 0.5)) 
    return bounds

# --- Initial guess generation for the optimizer ---
def generate_initial_guess(n_circles, seed=None):
    """
    Generates a structured initial guess based on a grid, with random perturbations.
    For N=26, it starts with a 5x5 grid and adds one more circle.
    This provides a more ordered starting point than pure random placement, which is
    more likely to lead the optimizer towards a high-quality structured packing.
    """
    if seed is not None:
        np.random.seed(seed)
    
    params = np.zeros(n_circles * 3)
    
    if n_circles == 26:
        # Create a 5x5 grid for the first 25 circles
        grid_size = 5
        indices = np.arange(grid_size * grid_size)
        x_grid = (indices % grid_size + 0.5) / grid_size
        y_grid = (indices // grid_size + 0.5) / grid_size

        # For the 26th circle, place it randomly within a central region to enhance diversity
        x_coords = np.append(x_grid, np.random.uniform(0.2, 0.8))
        y_coords = np.append(y_grid, np.random.uniform(0.2, 0.8))
        
        # Add small random perturbations to break symmetry and explore the local neighborhood.
        # Each restart gets a different perturbation because the seed is unique for each run.
        params[0::3] = x_coords + np.random.uniform(-0.05, 0.05, n_circles)
        params[1::3] = y_coords + np.random.uniform(-0.05, 0.05, n_circles)
        
        # Ensure points stay within reasonable bounds after perturbation
        params[0::3] = np.clip(params[0::3], 0.01, 0.99)
        params[1::3] = np.clip(params[1::3], 0.01, 0.99)
    else:
        # Fallback to random for other numbers of circles
        params[0::3] = np.random.uniform(0.05, 0.95, n_circles)
        params[1::3] = np.random.uniform(0.05, 0.95, n_circles)

    # Radii are still small random values to avoid initial overlaps
    params[2::3] = np.random.uniform(0.01, 0.05, n_circles)

    return params

# --- Validation function to check if a packing is valid ---
def validate_packing(circles_array, tolerance=1e-6):
    """
    Validates a given packing of circles against containment and non-overlap rules.
    circles_array: np.array of shape (N, 3) where each row is (x, y, r).
    Returns (True, ["All checks passed"]) or (False, list_of_error_messages).
    """
    n = circles_array.shape[0]
    if circles_array.shape != (n, 3):
        return False, [f"Invalid input shape: expected ({n}, 3), got {circles_array.shape}"]

    x = circles_array[:, 0]
    y = circles_array[:, 1]
    r = circles_array[:, 2]
    
    errors = []

    # 1. Positive Radii Check
    non_positive_radii_indices = np.where(r <= 0 - tolerance)[0]
    if len(non_positive_radii_indices) > 0:
        errors.append(f"Non-positive radii found at indices: {non_positive_radii_indices}")

    # 2. Containment within Unit Square Check
    # x_i - r_i >= 0, 1 - x_i - r_i >= 0, y_i - r_i >= 0, 1 - y_i - r_i >= 0
    if np.any(x - r < -tolerance): errors.append("Containment violation: x - r < 0")
    if np.any(1 - x - r < -tolerance): errors.append("Containment violation: 1 - x - r < 0")
    if np.any(y - r < -tolerance): errors.append("Containment violation: y - r < 0")
    if np.any(1 - y - r < -tolerance): errors.append("Containment violation: 1 - y - r < 0")

    # 3. Non-overlapping Check - vectorized
    x_col = x[:, np.newaxis]
    y_col = y[:, np.newaxis]
    r_col = r[:, np.newaxis]

    dx = x_col - x_col.T
    dy = y_col - y_col.T
    dist_sq_matrix = dx**2 + dy**2
    
    sum_r_sq_matrix = (r_col + r_col.T)**2

    idx = np.triu_indices(n, k=1) # Upper triangle, excluding diagonal for unique pairs
    
    # Check if any pair violates the non-overlap condition (distance_squared < (r_i + r_j)^2 - tolerance)
    violations = dist_sq_matrix[idx] < sum_r_sq_matrix[idx] - tolerance
    if np.any(violations):
        violation_indices = np.where(violations)[0]
        # Report up to the first 3 overlapping pairs for brevity
        overlap_pairs = [(idx[0][k], idx[1][k]) for k in violation_indices[:3]] 
        errors.append(f"Overlap violations detected ({len(violation_indices)} pairs). First few: {overlap_pairs}")

    if errors:
        return False, errors
    else:
        return True, ["All checks passed"]

# --- Visualization function (optional, for debugging/analysis) ---
def plot_packing(circles_array, title="", filename=None):
    """
    Plots the circles within the unit square using matplotlib.
    """
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.set_aspect('equal', adjustable='box') # Ensure circles appear circular
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_title(title)

    for x, y, r in circles_array:
        circle = plt.Circle((x, y), r, color='blue', alpha=0.6, ec='black') # ec for edge color
        ax.add_patch(circle)
    
    if filename:
        plt.savefig(filename)
    plt.close(fig) # Close the plot to free memory, especially important in loops or automated runs

# --- Main optimization function ---
def circle_packing26() -> np.ndarray:
    """
    Places 26 non-overlapping circles in the unit square, maximizing the sum of radii.
    Employs a multi-start SLSQP optimization strategy with parallel processing.

    Returns:
        circles: np.array of shape (26,3), where the i-th row (x,y,r) stores 
                 the (x,y) coordinates of the i-th circle and its radius r.
    """
    start_time = time.time()
    n = N_CIRCLES
    
    # Define the nonlinear constraint object for scipy.optimize.minimize
    num_containment_constraints = 4 * n
    num_non_overlap_constraints = n * (n - 1) // 2
    total_constraints = num_containment_constraints + num_non_overlap_constraints
    
    # All constraint values must be greater than or equal to 0
    nlc = NonlinearConstraint(constraints_fun, 
                              lb=np.zeros(total_constraints), 
                              ub=np.full(total_constraints, np.inf))

    # Define parameter bounds for (x, y, r) for each circle
    bounds = get_parameter_bounds(n)

    # Multi-start optimization parameters
    # Increase the number of restarts significantly to more thoroughly search the solution space.
    # With a better initial guess strategy, more restarts increase the chance of finding the global optimum.
    num_restarts = 500 # Number of independent optimization runs from random initial guesses, further increased for a more exhaustive search
    # Increase max iterations per run to allow for full convergence from diverse initial points.
    max_iter_per_run = 5000 # Maximum iterations for the SLSQP algorithm in each run

    # Use joblib to run multiple optimization processes in parallel
    # n_jobs=-1 uses all available CPU cores
    results = joblib.Parallel(n_jobs=-1)(
        joblib.delayed(minimize)(
            objective,
            generate_initial_guess(n, seed=GLOBAL_SEED + i), # Each run gets a unique seed
            method='SLSQP', # Sequential Least Squares Programming, suitable for non-linear constraints
            bounds=bounds,
            constraints=[nlc],
            options={'ftol': 1e-9, 'maxiter': max_iter_per_run, 'disp': False} # Use a tighter function tolerance for higher precision.
        ) for i in range(num_restarts)
    )

    best_sum_radii = -np.inf # Initialize with negative infinity for maximization
    best_circles_params = None
    
    # Iterate through results from all parallel runs to find the best one
    for res in results:
        # Check if the optimization run was successful and if it yielded a better sum of radii
        if res.success and -res.fun > best_sum_radii: 
            best_sum_radii = -res.fun # -res.fun because objective minimizes negative sum
            best_circles_params = res.x

    # Handle cases where no successful optimization run was found
    if best_circles_params is None:
        print("Warning: No successful optimization run found. Returning a default zero-radius packing.")
        return np.zeros((n, 3)) 

    # Reshape the best found parameters into the (N, 3) format (x, y, r)
    best_circles = np.reshape(best_circles_params, (n, 3))

    # Final validation of the best result found
    is_valid, validation_messages = validate_packing(best_circles)
    if not is_valid:
        print(f"Warning: The best packing found is technically invalid. Sum of radii: {best_sum_radii:.6f}")
        for msg in validation_messages:
            print(f"  - {msg}")
        # Note: SLSQP might converge to slightly infeasible points due to numerical tolerances.
        # We return the solution as is, with the warning.
    
    # Optional: Plot the best packing. Commented out by default for automated evaluation environments.
    # plot_packing(best_circles, title=f"Optimal Packing N={n}, Sum R={best_sum_radii:.4f}", filename="packing_26_optimal.png")

    end_time = time.time()
    # The evaluation framework will measure and report performance metrics.
    # For local debugging/analysis, these can be printed:
    # print(f"Optimization finished in {end_time - start_time:.2f} seconds.")
    # print(f"Best sum of radii: {best_sum_radii:.6f}")
    # print(f"Benchmark ratio: {best_sum_radii / 2.6358627564136983:.4f}")

    return best_circles


# EVOLVE-BLOCK-END
