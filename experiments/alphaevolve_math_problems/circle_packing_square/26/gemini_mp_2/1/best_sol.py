# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import basinhopping, minimize
from scipy.spatial.distance import pdist # Added for vectorized distance calculation
import time

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

# --- Constraint Functions for scipy.optimize.minimize ---

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
        print(f"Invalid: Expected shape ({N_CIRCLES}, 3), got {circles.shape}")
        return False
    
    x, y, r = circles[:, 0], circles[:, 1], circles[:, 2]

    # Check for negative radii (should be handled by bounds, but double-check)
    if np.any(r < -tol): 
        print(f"Invalid: Negative radii found: {r[r < -tol]}")
        return False

    # Check containment
    if np.any(x - r < -tol) or \
       np.any(1 - x - r < -tol) or \
       np.any(y - r < -tol) or \
       np.any(1 - y - r < -tol):
        print("Invalid: Containment violated.")
        # Optionally print details for violated circles
        # for i in range(N_CIRCLES):
        #     if x[i] - r[i] < -tol or 1 - x[i] - r[i] < -tol or y[i] - r[i] < -tol or 1 - y[i] - r[i] < -tol:
        #         print(f"  Circle {i}: x={x[i]:.4f}, y={y[i]:.4f}, r={r[i]:.4f} - violates containment.")
        return False
    
    # Check non-overlap (using vectorized calculation for efficiency and consistency)
    distances = pdist(circles[:, :2])
    radii_sum_matrix = r[:, None] + r
    triu_indices = np.triu_indices(N_CIRCLES, k=1)
    radii_sums_condensed = radii_sum_matrix[triu_indices]
    
    if np.any(distances < radii_sums_condensed - tol):
        print("Invalid: Overlap detected.")
        # For detailed debugging of overlaps, the commented out loop based approach
        # or more complex mapping from condensed index would be needed.
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

def _run_single_optimization(initial_circles: np.ndarray, run_id: int, 
                           niter: int, temperature: float, stepsize: float) -> tuple:
    """Run a single optimization with given parameters."""
    print(f"  Run {run_id}: niter={niter}, T={temperature}, stepsize={stepsize}")
    
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
        "options": {"ftol": 1e-7, "maxiter": 800}  # Reduced for multi-start approach
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
    
    print(f"    Result: sum_radii = {sum_radii:.6f}")
    return circles, sum_radii, result.fun

def circle_packing26() -> np.ndarray:
    """
    Places 26 non-overlapping circles in the unit square in order to maximize the sum of radii.
    Uses a multi-start strategy with different initialization and optimization parameters.
    """
    start_time = time.time()
    
    np.random.seed(RANDOM_SEED)
    
    print(f"Starting multi-start optimization for {N_CIRCLES} circles...")
    
    # Strategy 1: Random initialization with aggressive exploration
    np.random.seed(RANDOM_SEED)
    initial_radii = np.random.uniform(0.001, 0.05, N_CIRCLES)
    initial_x = np.random.uniform(initial_radii, 1 - initial_radii, N_CIRCLES)
    initial_y = np.random.uniform(initial_radii, 1 - initial_radii, N_CIRCLES)
    random_initial = np.column_stack((initial_x, initial_y, initial_radii))
    
    # Strategy 2: Grid-based initialization
    np.random.seed(RANDOM_SEED + 100)
    grid_initial = _generate_grid_initial_guess()
    
    # Strategy 3: Another random initialization with different seed
    np.random.seed(RANDOM_SEED + 200)
    initial_radii2 = np.random.uniform(0.001, 0.03, N_CIRCLES)
    initial_x2 = np.random.uniform(initial_radii2, 1 - initial_radii2, N_CIRCLES)
    initial_y2 = np.random.uniform(initial_radii2, 1 - initial_radii2, N_CIRCLES)
    random_initial2 = np.column_stack((initial_x2, initial_y2, initial_radii2))
    
    # Run multiple optimization strategies
    optimization_runs = [
        (random_initial, 1, 120, 2.0, 0.07),    # Aggressive exploration
        (grid_initial, 2, 100, 1.5, 0.05),      # Moderate exploration from grid
        (random_initial2, 3, 80, 2.5, 0.08),    # High temperature exploration
    ]
    
    best_circles = None
    best_sum_radii = -np.inf
    
    for initial_circles, run_id, niter, temperature, stepsize in optimization_runs:
        circles, sum_radii, obj_val = _run_single_optimization(
            initial_circles, run_id, niter, temperature, stepsize
        )
        
        if sum_radii > best_sum_radii and is_valid_configuration(circles):
            best_circles = circles
            best_sum_radii = sum_radii
            print(f"    New best valid solution: {sum_radii:.6f}")
    
    if best_circles is None:
        print("WARNING: No valid solution found in multi-start approach. Using fallback.")
        best_circles = random_initial
    
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
        options={"ftol": 1e-8, "maxiter": 1500}  # High precision final polish
    )
    
    final_circles = _params_to_circles(final_refinement_result.x)
    
    # Validate the final solution
    if not is_valid_configuration(final_circles):
        print("WARNING: Final refinement broke constraints. Reverting to best multi-start solution.")
        final_circles = best_circles
    
    end_time = time.time()
    final_sum = calculate_sum_radii(final_circles)
    print(f"Multi-start optimization completed in {end_time - start_time:.2f} seconds.")
    print(f"Final sum of radii: {final_sum:.6f}")
    
    return final_circles


# EVOLVE-BLOCK-END
