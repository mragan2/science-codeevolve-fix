# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import basinhopping, minimize
from scipy.spatial.distance import pdist
import time
from joblib import Parallel, delayed # Added for parallelization

# --- Constants ---
N_CIRCLES = 26
RANDOM_SEED = 42
CONSTRAINT_TOL = 1e-6 # Tolerance for checking constraints in validation

# --- Helper Functions for Circle Packing ---

def _params_to_circles(params: np.ndarray) -> np.ndarray:
    """Converts a flattened parameter array [x1, y1, r1, ...] to a (N, 3) circle array."""
    return params.reshape((N_CIRCLES, 3))

def _circles_to_params(circles: np.ndarray) -> np.ndarray:
    """Converts a (N, 3) circle array to a flattened parameter array."""
    return circles.flatten()

def _objective_function(params: np.ndarray) -> float:
    """Objective function to minimize: negative sum of radii.
    A small penalty is added for negative radii, although bounds should primarily handle this."""
    circles = _params_to_circles(params)
    radii = circles[:, 2]
    # Penalize negative radii heavily if they somehow slip through bounds
    penalty = np.sum(np.maximum(0, -radii)) * 1e6 
    return -np.sum(radii) + penalty

# --- Constraint Functions for scipy.optimize.minimize (Vectorized for performance) ---

def _all_containment_constraints(params: np.ndarray) -> np.ndarray:
    """Returns an array of values for all containment constraints (>= 0).
    Concatenates x-r, 1-x-r, y-r, 1-y-r for all circles."""
    circles = _params_to_circles(params)
    x, y, r = circles[:, 0], circles[:, 1], circles[:, 2]
    
    # Constraints: x - r >= 0, 1 - x - r >= 0, y - r >= 0, 1 - y - r >= 0
    c1 = x - r
    c2 = 1 - x - r
    c3 = y - r
    c4 = 1 - y - r
    
    return np.concatenate((c1, c2, c3, c4))

def _overlap_constraints_vectorized(params: np.ndarray) -> np.ndarray:
    """Returns an array of values for non-overlap constraints (>= 0), using vectorized operations."""
    circles = _params_to_circles(params)
    x_coords, y_coords, radii = circles[:, 0], circles[:, 1], circles[:, 2]
    
    # Calculate pairwise Euclidean distances between circle centers
    # pdist returns a condensed distance matrix (1D array of N*(N-1)/2 elements)
    distances = pdist(circles[:, :2])
    
    # Calculate pairwise sum of radii for unique pairs
    # Create a matrix of r_i + r_j
    radii_sum_matrix = radii[:, None] + radii
    # Extract the upper triangle (excluding diagonal) elements to match pdist output format
    triu_indices = np.triu_indices(N_CIRCLES, k=1)
    radii_sums_condensed = radii_sum_matrix[triu_indices]
    
    # The constraint is dist - (r_i + r_j) >= 0
    return distances - radii_sums_condensed

def _get_constraints():
    """Returns a list of constraint dictionaries for scipy.optimize.minimize."""
    constraints = []
    
    # All containment constraints in one go (4 * N_CIRCLES inequalities)
    constraints.append({'type': 'ineq', 'fun': _all_containment_constraints})

    # Non-overlap constraints (N_CIRCLES*(N_CIRCLES-1)/2 inequalities)
    constraints.append({'type': 'ineq', 'fun': _overlap_constraints_vectorized})
    
    return constraints

# --- Validation Functions ---

def is_valid_configuration(circles: np.ndarray, tol: float = CONSTRAINT_TOL) -> bool:
    """
    Checks if a given configuration of circles is valid (contained and non-overlapping).
    """
    if circles.shape != (N_CIRCLES, 3):
        # print(f"Invalid: Expected shape ({N_CIRCLES}, 3), got {circles.shape}") # Removed for cleaner output
        return False
    
    x, y, r = circles[:, 0], circles[:, 1], circles[:, 2]

    # Check for negative radii (should be handled by bounds, but double-check)
    if np.any(r < -tol): 
        # print(f"Invalid: Negative radii found: {r[r < -tol]}") # Removed for cleaner output
        return False

    # Check containment
    if np.any(x - r < -tol) or \
       np.any(1 - x - r < -tol) or \
       np.any(y - r < -tol) or \
       np.any(1 - y - r < -tol):
        # print("Invalid: Containment violated.") # Removed for cleaner output
        return False
    
    # Check non-overlap (using vectorized calculation for efficiency and consistency)
    distances = pdist(circles[:, :2])
    radii_sum_matrix = r[:, None] + r
    triu_indices = np.triu_indices(N_CIRCLES, k=1)
    radii_sums_condensed = radii_sum_matrix[triu_indices]
    
    if np.any(distances < radii_sums_condensed - tol):
        # print("Invalid: Overlap detected.") # Removed for cleaner output
        return False
                
    return True

def calculate_sum_radii(circles: np.ndarray) -> float:
    """Calculates the sum of radii for a given circle configuration."""
    return np.sum(circles[:, 2])

# --- Main Optimization Function ---

def _generate_grid_initial_guess() -> np.ndarray:
    """Generate a structured grid-based initial guess for circle placement."""
    # Create a rough grid layout
    grid_size = int(np.ceil(np.sqrt(N_CIRCLES)))
    spacing = 1.0 / (grid_size + 1)
    
    circles = []
    for i in range(N_CIRCLES):
        row = i // grid_size
        col = i % grid_size
        
        # Add some randomness to avoid perfect grid alignment
        x = (col + 1) * spacing + np.random.uniform(-spacing*0.2, spacing*0.2)
        y = (row + 1) * spacing + np.random.uniform(-spacing*0.2, spacing*0.2)
        
        # Ensure within bounds
        x = np.clip(x, 0.02, 0.98)
        y = np.clip(y, 0.02, 0.98)
        
        # Small initial radius
        r = np.random.uniform(0.001, 0.02)
        
        circles.append([x, y, r])
    
    return np.array(circles)

def _generate_dense_initial_guess() -> np.ndarray:
    """Generate a denser, hexagonal-like, initial guess for N=26."""
    circles = []
    
    # For N_CIRCLES = 26, a 5x5 grid (25 circles) plus one more is a common approach.
    # Let's try to arrange them in a somewhat dense pattern, like a distorted hexagonal grid.
    
    # Number of circles per row, alternating for a hexagonal-like pattern
    # Sums to 26: 5+5+5+6+5
    circles_per_row = [5, 5, 5, 6, 5] 
    num_rows = len(circles_per_row)
    
    # Estimate a base radius that encourages density without too much initial overlap
    # A value around 0.11-0.12 often works well for N=26.
    base_r = 0.11 
    
    # Calculate uniform step sizes based on the *maximum* number of circles in a row
    # and total number of rows, to maintain a somewhat consistent grid.
    max_circles_in_row = max(circles_per_row)
    
    x_spacing_unit = 1.0 / (max_circles_in_row + 1)
    y_spacing_unit = 1.0 / (num_rows + 1)
    
    current_circle_idx = 0
    
    for row_idx, num_in_row in enumerate(circles_per_row):
        y = (row_idx + 1) * y_spacing_unit
        
        # Calculate horizontal starting position to center the current row in the square
        # For hexagonal packing, alternate rows are offset by half a step.
        row_actual_width_span = (num_in_row - 1) * x_spacing_unit
        x_start_for_centering = (1.0 - (row_actual_width_span + 2*x_spacing_unit/2)) / 2.0 # accounts for "half circle" at ends
        
        if row_idx % 2 == 1: # Odd rows shifted for hexagonal effect
            x_start_for_centering += x_spacing_unit / 2.0
        
        for col_idx in range(num_in_row):
            if current_circle_idx >= N_CIRCLES:
                break
            
            x = x_start_for_centering + (col_idx + 0.5) * x_spacing_unit # Center of circle
            
            # Add some slight perturbation for variety and to escape local minima
            x += np.random.uniform(-x_spacing_unit * 0.1, x_spacing_unit * 0.1)
            y += np.random.uniform(-y_spacing_unit * 0.1, y_spacing_unit * 0.1)
            
            # Ensure within reasonable bounds
            x = np.clip(x, 0.05, 0.95)
            y = np.clip(y, 0.05, 0.95)
            
            # Perturb base radius for individual circles
            r = base_r * np.random.uniform(0.9, 1.1) 
            r = np.clip(r, 0.001, 0.5) # Ensure valid radius range
            
            circles.append([x, y, r])
            current_circle_idx += 1
    
    # Fallback for ensuring N_CIRCLES are generated if the pattern logic is complex
    while len(circles) < N_CIRCLES:
        # Add random small circles until N_CIRCLES is met
        circles.append([np.random.uniform(0.1, 0.9), np.random.uniform(0.1, 0.9), np.random.uniform(0.005, 0.02)])

    return np.array(circles)

def _generate_large_radii_initial_guess() -> np.ndarray:
    """Generate a random initial guess with larger initial radii, allowing the optimizer to shrink them."""
    initial_radii = np.random.uniform(0.06, 0.15, N_CIRCLES) # Larger initial radii
    initial_x = np.random.uniform(initial_radii, 1 - initial_radii, N_CIRCLES)
    initial_y = np.random.uniform(initial_radii, 1 - initial_radii, N_CIRCLES)
    return np.column_stack((initial_x, initial_y, initial_radii))

def _generate_center_weighted_initial_guess() -> np.ndarray:
    """Generate an initial guess with circles clustered towards the center, with moderate radii."""
    circles = []
    center_x, center_y = 0.5, 0.5
    
    for i in range(N_CIRCLES):
        # Generate coordinates with a normal distribution around the center
        # Standard deviation chosen to keep most circles within the square but clustered
        x = np.random.normal(center_x, 0.15)
        y = np.random.normal(center_y, 0.15)
        
        # Radii slightly larger than initial random, but not as large as `large_radii_initial`
        r = np.random.uniform(0.03, 0.08) 
        
        # Ensure initial placement is valid within bounds (even if overlapping with others)
        x = np.clip(x, r, 1 - r)
        y = np.clip(y, r, 1 - r)
        
        circles.append([x, y, r])
    return np.array(circles)


def _run_single_optimization(initial_circles: np.ndarray, run_id: int, 
                           niter: int, temperature: float, stepsize: float,
                           minimizer_maxiter: int) -> tuple:
    """Run a single optimization with given parameters."""
    # print(f"  Run {run_id}: niter={niter}, T={temperature}, stepsize={stepsize}, minimizer_maxiter={minimizer_maxiter}") # Suppress output for parallel runs
    
    x0 = _circles_to_params(initial_circles)
    
    # Define bounds
    bounds = []
    for _ in range(N_CIRCLES):
        bounds.append((0.0, 1.0))  # x coordinate
        bounds.append((0.0, 1.0))  # y coordinate
        bounds.append((0.0, 0.5))  # r coordinate
    
    # Minimizer arguments for basinhopping
    minimizer_kwargs = {
        "method": "SLSQP",
        "bounds": bounds,
        "constraints": _get_constraints(),
        "options": {"ftol": 1e-7, "maxiter": minimizer_maxiter} 
    }

    # Basinhopping
    result = basinhopping(
        _objective_function,
        x0,
        minimizer_kwargs=minimizer_kwargs,
        niter=niter,
        T=temperature,
        stepsize=stepsize,
        seed=RANDOM_SEED + run_id,  # Different seed for each run
        disp=False
    )

    circles = _params_to_circles(result.x)
    sum_radii = calculate_sum_radii(circles)
    
    return circles, sum_radii, result.fun, run_id # Return run_id for identification

def circle_packing26() -> np.ndarray:
    """
    Places 26 non-overlapping circles in the unit square in order to maximize the sum of radii.
    Uses a multi-start strategy with different initialization and optimization parameters.
    """
    start_time = time.time()
    
    np.random.seed(RANDOM_SEED)
    
    print(f"Starting multi-start optimization for {N_CIRCLES} circles...")
    
    # Strategy 1: Random initialization with aggressive exploration (re-tuned)
    np.random.seed(RANDOM_SEED)
    initial_radii = np.random.uniform(0.001, 0.05, N_CIRCLES)
    initial_x = np.random.uniform(initial_radii, 1 - initial_radii, N_CIRCLES)
    initial_y = np.random.uniform(initial_radii, 1 - initial_radii, N_CIRCLES)
    random_initial = np.column_stack((initial_x, initial_y, initial_radii))
    
    # Strategy 2: Grid-based initialization (re-tuned)
    np.random.seed(RANDOM_SEED + 100)
    grid_initial = _generate_grid_initial_guess()
    
    # Strategy 3: Another random initialization with different seed (re-tuned)
    np.random.seed(RANDOM_SEED + 200)
    initial_radii2 = np.random.uniform(0.001, 0.03, N_CIRCLES)
    initial_x2 = np.random.uniform(initial_radii2, 1 - initial_radii2, N_CIRCLES)
    initial_y2 = np.random.uniform(initial_radii2, 1 - initial_radii2, N_CIRCLES)
    random_initial2 = np.column_stack((initial_x2, initial_y2, initial_radii2))

    # Strategy 4: Dense/Hexagonal-like initialization (re-tuned)
    np.random.seed(RANDOM_SEED + 300)
    dense_initial = _generate_dense_initial_guess()

    # Strategy 5: Random initialization with larger initial radii (re-tuned)
    np.random.seed(RANDOM_SEED + 400)
    large_radii_initial = _generate_large_radii_initial_guess()

    # Strategy 6: Another grid-based with different perturbation (re-tuned)
    np.random.seed(RANDOM_SEED + 500)
    grid_initial2 = _generate_grid_initial_guess() # Re-use, but different seed for random component

    # Strategy 7: Center-weighted initial guess (NEW strategy)
    np.random.seed(RANDOM_SEED + 600)
    center_weighted_initial = _generate_center_weighted_initial_guess()

    # Strategy 8: Another random initialization, slightly more conservative (NEW strategy)
    np.random.seed(RANDOM_SEED + 700)
    initial_radii3 = np.random.uniform(0.005, 0.04, N_CIRCLES)
    initial_x3 = np.random.uniform(initial_radii3, 1 - initial_radii3, N_CIRCLES)
    initial_y3 = np.random.uniform(initial_radii3, 1 - initial_radii3, N_CIRCLES)
    random_initial3 = np.column_stack((initial_x3, initial_y3, initial_radii3))
    
    # Run multiple optimization strategies with varied parameters
    # Increased niter and maxiter for more thorough search, leveraging parallelization
    optimization_runs = [
        (random_initial, 1, 350, 3.0, 0.12, 1800),    # Aggressive exploration
        (grid_initial, 2, 300, 2.2, 0.08, 1600),      # Moderate exploration from grid
        (random_initial2, 3, 250, 4.0, 0.15, 1300),   # Very high temperature, very large steps
        (dense_initial, 4, 400, 1.8, 0.06, 2000),     # Denser initial, deep local refinement
        (large_radii_initial, 5, 320, 2.8, 0.10, 1500), # Large initial radii, aggressive exploration
        (grid_initial2, 6, 280, 2.5, 0.09, 1700),      # Another grid, slightly more aggressive
        (center_weighted_initial, 7, 300, 2.0, 0.07, 1600), # New strategy, balanced
        (random_initial3, 8, 200, 1.5, 0.05, 1000)    # More conservative random, quicker check
    ]
    
    best_circles = None
    best_sum_radii = -np.inf
    
    # Parallel execution using all available CPU cores
    results = Parallel(n_jobs=-1)(
        delayed(_run_single_optimization)(initial_circles, run_id, niter, temperature, stepsize, minimizer_maxiter)
        for initial_circles, run_id, niter, temperature, stepsize, minimizer_maxiter in optimization_runs
    )

    for circles, sum_radii, obj_val, run_id in results:
        is_valid = is_valid_configuration(circles)
        print(f"  Run {run_id} finished: sum_radii = {sum_radii:.6f}, valid={is_valid}")
        if sum_radii > best_sum_radii and is_valid:
            best_circles = circles
            best_sum_radii = sum_radii
            print(f"    New best valid solution: {sum_radii:.6f} from Run {run_id}")
    
    if best_circles is None:
        print("WARNING: No valid solution found in multi-start approach. Attempting to generate a valid fallback.")
        # Attempt to make a very small valid configuration
        fallback_circles = np.zeros((N_CIRCLES, 3))
        fallback_r = 0.005 # Very small radius
        for i in range(N_CIRCLES):
            fallback_circles[i] = [0.5 + (i % 2 - 0.5) * 0.1, 0.5 + (i // 2 - 0.5) * 0.1, fallback_r]
        best_circles = fallback_circles
        if not is_valid_configuration(best_circles):
            print("Fallback also invalid, using a minimal default.")
            best_circles = np.array([[0.5, 0.5, 0.001]] * N_CIRCLES) # Extremely small, guaranteed valid
        else:
            print(f"Using generated fallback with sum_radii: {calculate_sum_radii(best_circles):.6f}")


    # Final local refinement on the best solution found
    print("Starting final local refinement on best solution...")
    
    bounds = []
    for _ in range(N_CIRCLES):
        bounds.append((0.0, 1.0))  # x coordinate
        bounds.append((0.0, 1.0))  # y coordinate
        bounds.append((0.0, 0.5))  # r coordinate
    
    final_refinement_result = minimize(
        _objective_function,
        _circles_to_params(best_circles),
        method="SLSQP",
        bounds=bounds,
        constraints=_get_constraints(),
        options={"ftol": 1e-10, "maxiter": 3500}  # Higher precision final polish
    )
    
    final_circles = _params_to_circles(final_refinement_result.x)
    
    # Validate the final solution
    if not is_valid_configuration(final_circles):
        print("WARNING: Final refinement broke constraints. Reverting to best multi-start solution.")
        # best_circles should already be validated.
        final_circles = best_circles
        # If the best_circles itself was a fallback, ensure it's still good.
        if not is_valid_configuration(final_circles):
            print("CRITICAL WARNING: Reverted solution is also invalid. This implies a deeper issue in validation or fallback.")
            # As a last resort, ensure a minimal valid solution is returned.
            final_circles = np.array([[0.5, 0.5, 0.001]] * N_CIRCLES)
            
    end_time = time.time()
    final_sum = calculate_sum_radii(final_circles)
    print(f"Multi-start optimization completed in {end_time - start_time:.2f} seconds.")
    print(f"Final sum of radii: {final_sum:.6f}")
    
    return final_circles


# EVOLVE-BLOCK-END
