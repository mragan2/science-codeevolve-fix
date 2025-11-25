# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize, Bounds, NonlinearConstraint, basinhopping
from numba import njit
# import time # Keep time for potential future performance profiling, though not used in current logic.

# --- Constants ---
N_CIRCLES = 32
BENCHMARK_SUM_RADII = 2.937944526205518
RANDOM_SEED = 42
MIN_RADIUS = 1e-7  # Minimum radius to avoid numerical issues and ensure positive radii
MAX_RADIUS = 0.5   # Maximum possible radius for a single circle in a unit square (0.5 if centered)

# --- Objective Function ---
def objective(params: np.ndarray) -> float:
    """
    Objective function to minimize: negative sum of radii.
    params: flattened array [x1, y1, r1, x2, y2, r2, ...]
    """
    radii = params[2::3]
    return -np.sum(radii)

# --- Numba-optimized Constraint Functions ---
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


def generate_initial_guess_hexagonal_perturbed(n: int, seed: int = None) -> np.ndarray:
    """
    Generates an initial guess based on a perturbed hexagonal lattice.
    This lattice provides a denser initial packing than a square grid.
    """
    if seed is not None:
        np.random.seed(seed)

    # A 6x6 hex grid gives 36 points, which is close to 32.
    cols, rows = 6, 6
    
    a = 1.0 / cols
    initial_r = a / 2.0 * 0.95 # Use 95% of radius to allow expansion

    dx = a
    dy = a * np.sqrt(3) / 2.0

    points = []
    for row in range(rows):
        for col in range(cols):
            x = col * dx
            if row % 2 == 1:
                x += dx / 2.0  # Offset odd rows
            y = row * dy
            points.append([x, y])
    
    points = np.array(points)
    
    # Center the generated grid within the unit square
    min_x, max_x = np.min(points[:, 0]), np.max(points[:, 0])
    min_y, max_y = np.min(points[:, 1]), np.max(points[:, 1])
    points[:, 0] += (1.0 - (max_x - min_x)) / 2.0 - min_x
    points[:, 1] += (1.0 - (max_y - min_y)) / 2.0 - min_y

    # Select the n points closest to the center of the square
    center = np.array([0.5, 0.5])
    distances_to_center = np.linalg.norm(points - center, axis=1)
    sorted_indices = np.argsort(distances_to_center)
    selected_points = points[sorted_indices[:n]]

    # Assemble the parameter vector
    initial_params = np.empty(n * 3, dtype=np.float64)
    initial_params[0::3] = selected_points[:, 0]
    initial_params[1::3] = selected_points[:, 1]
    initial_params[2::3] = initial_r

    # Add small random perturbation
    perturb_scale_xy = 0.01
    perturb_scale_r = 0.001
    
    initial_params[0::3] += np.random.uniform(-perturb_scale_xy, perturb_scale_xy, size=n)
    initial_params[1::3] += np.random.uniform(-perturb_scale_xy, perturb_scale_xy, size=n)
    initial_params[2::3] += np.random.uniform(-perturb_scale_r, perturb_scale_r, size=n)
    
    # Clip to ensure validity after perturbation
    initial_params[2::3] = np.clip(initial_params[2::3], MIN_RADIUS, MAX_RADIUS)
    for k in range(n):
        r_k = initial_params[k*3+2]
        initial_params[k*3] = np.clip(initial_params[k*3], r_k, 1 - r_k)
        initial_params[k*3+1] = np.clip(initial_params[k*3+1], r_k, 1 - r_k)

    return initial_params


def generate_initial_guess_corner_anchored(n: int, seed: int = None) -> np.ndarray:
    """
    Generates an initial guess with 4 large circles anchored in the corners,
    and the remaining circles placed randomly in the interior.
    """
    if seed is not None:
        np.random.seed(seed)

    num_anchors = 4
    if n < num_anchors: # Handle cases where N is too small for this strategy
        return generate_initial_guess_random_perturbed(n, seed)

    initial_params = np.empty(n * 3, dtype=np.float64)
    r_anchor = 0.15 # Heuristic radius for corner circles

    # Place anchor circles
    anchor_coords = [
        (r_anchor, r_anchor),
        (1 - r_anchor, r_anchor),
        (r_anchor, 1 - r_anchor),
        (1 - r_anchor, 1 - r_anchor)
    ]
    for i in range(num_anchors):
        initial_params[i*3] = anchor_coords[i][0]
        initial_params[i*3+1] = anchor_coords[i][1]
        initial_params[i*3+2] = r_anchor

    # Place remaining circles randomly in the central region
    remaining_n = n - num_anchors
    initial_r_filler_base = 0.03 # Smaller base radius for filler circles
    
    # Define a reduced square for filler circles to avoid immediate overlap with anchors
    min_coord_filler = r_anchor + initial_r_filler_base * 2 # Offset to give space
    max_coord_filler = 1 - r_anchor - initial_r_filler_base * 2
    
    for k in range(remaining_n):
        idx = num_anchors + k
        x_k = np.random.uniform(min_coord_filler, max_coord_filler)
        y_k = np.random.uniform(min_coord_filler, max_coord_filler)
        r_k = initial_r_filler_base + np.random.uniform(-initial_r_filler_base/2, initial_r_filler_base/2)
        r_k = np.clip(r_k, MIN_RADIUS, MAX_RADIUS)

        initial_params[idx*3] = np.clip(x_k, r_k, 1 - r_k)
        initial_params[idx*3+1] = np.clip(y_k, r_k, 1 - r_k)
        initial_params[idx*3+2] = r_k
    
    # Add small random perturbation to all circles to break symmetry and aid optimization
    perturb_scale_xy = 0.005 # Slightly smaller perturbation for more structured initial guess
    perturb_scale_r = 0.0005
    
    initial_params[0::3] += np.random.uniform(-perturb_scale_xy, perturb_scale_xy, size=n)
    initial_params[1::3] += np.random.uniform(-perturb_scale_xy, perturb_scale_xy, size=n)
    initial_params[2::3] += np.random.uniform(-perturb_scale_r, perturb_scale_r, size=n)
    
    # Ensure validity after perturbation
    initial_params[2::3] = np.clip(initial_params[2::3], MIN_RADIUS, MAX_RADIUS)
    for k in range(n):
        r_k = initial_params[k*3+2]
        initial_params[k*3] = np.clip(initial_params[k*3], r_k, 1 - r_k)
        initial_params[k*3+1] = np.clip(initial_params[k*3+1], r_k, 1 - r_k)

    return initial_params


def generate_initial_guess_central_anchored(n: int, seed: int = None) -> np.ndarray:
    """
    Generates an initial guess with 4 larger circles anchored centrally,
    and the remaining circles placed randomly (with small radii) in the square.
    """
    if seed is not None:
        np.random.seed(seed)

    num_anchors = 4
    if n < num_anchors:
        return generate_initial_guess_random_perturbed(n, seed)

    initial_params = np.empty(n * 3, dtype=np.float64)
    r_center_anchor = 0.12 # Heuristic radius for central circles

    # Place 4 anchor circles in a small square pattern in the center
    # The centers will be (0.5 +/- d, 0.5 +/- d)
    d_center_offset = r_center_anchor * 1.05 # Distance from 0.5 for centers, creating a small gap
    center_coords = [
        (0.5 - d_center_offset, 0.5 - d_center_offset),
        (0.5 + d_center_offset, 0.5 - d_center_offset),
        (0.5 - d_center_offset, 0.5 + d_center_offset),
        (0.5 + d_center_offset, 0.5 + d_center_offset)
    ]
    for i in range(num_anchors):
        initial_params[i*3] = center_coords[i][0]
        initial_params[i*3+1] = center_coords[i][1]
        initial_params[i*3+2] = r_center_anchor

    # Place remaining circles randomly in the entire square with small radii
    remaining_n = n - num_anchors
    initial_r_filler_base = 0.02 # Small base radius for filler circles
    
    for k in range(remaining_n):
        idx = num_anchors + k
        x_k = np.random.uniform(0.0, 1.0)
        y_k = np.random.uniform(0.0, 1.0)
        r_k = initial_r_filler_base + np.random.uniform(-initial_r_filler_base/2, initial_r_filler_base/2)
        r_k = np.clip(r_k, MIN_RADIUS, MAX_RADIUS)

        # Ensure x, y are within bounds for the given r
        initial_params[idx*3] = np.clip(x_k, r_k, 1 - r_k)
        initial_params[idx*3+1] = np.clip(y_k, r_k, 1 - r_k)
        initial_params[idx*3+2] = r_k
    
    # Add small random perturbation to all circles to break symmetry and aid optimization
    perturb_scale_xy = 0.005
    perturb_scale_r = 0.0005
    
    initial_params[0::3] += np.random.uniform(-perturb_scale_xy, perturb_scale_xy, size=n)
    initial_params[1::3] += np.random.uniform(-perturb_scale_xy, perturb_scale_xy, size=n)
    initial_params[2::3] += np.random.uniform(-perturb_scale_r, perturb_scale_r, size=n)
    
    # Ensure validity after perturbation
    initial_params[2::3] = np.clip(initial_params[2::3], MIN_RADIUS, MAX_RADIUS)
    for k in range(n):
        r_k = initial_params[k*3+2]
        initial_params[k*3] = np.clip(initial_params[k*3], r_k, 1 - r_k)
        initial_params[k*3+1] = np.clip(initial_params[k*3+1], r_k, 1 - r_k)

    return initial_params


# --- Custom Perturbation for Basin Hopping ---
class CustomTakeStep:
    """Custom step-taking routine for basinhopping to perturb x, y, and r differently."""
    def __init__(self, stepsize_xy: float = 0.05, stepsize_r: float = 0.01, random_seed: int = None):
        self.stepsize_xy = stepsize_xy
        self.stepsize_r = stepsize_r
        # Use a dedicated RandomState for reproducibility within the class
        self.rng = np.random.RandomState(random_seed)

    def __call__(self, x: np.ndarray) -> np.ndarray:
        """Applies a random perturbation to the parameter vector."""
        n = len(x) // 3
        
        # Perturb x and y coordinates
        x[0::3] += self.rng.uniform(-self.stepsize_xy, self.stepsize_xy, n)
        x[1::3] += self.rng.uniform(-self.stepsize_xy, self.stepsize_xy, n)
        
        # Perturb radii
        x[2::3] += self.rng.uniform(-self.stepsize_r, self.stepsize_r, n)
        
        # We don't clip here; the minimizer's bounds will handle invalid values.
        # This allows the search to explore near the boundaries more freely.
        return x


# --- Main Constructor Function ---
def circle_packing32() -> np.ndarray:
    """
    Places 32 non-overlapping circles in the unit square to maximize the sum of radii.
    Employs basin-hopping with a custom step function and SLSQP as the local minimizer.

    Returns:
        circles: np.array of shape (32,3), where the i-th row (x,y,r) stores the 
                 (x,y) coordinates of the i-th circle of radius r.
    """
    # Define general bounds for x, y, r for the optimization variables.
    lower_bounds_array = np.zeros(N_CIRCLES * 3, dtype=np.float64)
    upper_bounds_array = np.ones(N_CIRCLES * 3, dtype=np.float64)
    lower_bounds_array[2::3] = MIN_RADIUS # Radii must be positive
    upper_bounds_array[2::3] = MAX_RADIUS # Radii cannot exceed 0.5
    param_bounds = Bounds(lower_bounds_array, upper_bounds_array)

    # Define the NonlinearConstraint object for all boundary and non-overlap conditions.
    num_boundary_constraints = 4 * N_CIRCLES
    num_overlap_constraints = N_CIRCLES * (N_CIRCLES - 1) // 2
    total_constraints = num_boundary_constraints + num_overlap_constraints
    
    nonlinear_constraint = NonlinearConstraint(
        combined_constraints_numba,
        lb=np.zeros(total_constraints, dtype=np.float64),
        ub=np.full(total_constraints, np.inf, dtype=np.float64)
    )

    # Use the best heuristic initializer for a strong starting point for basin-hopping.
    x0 = generate_initial_guess_hexagonal_perturbed(N_CIRCLES, seed=RANDOM_SEED)

    # Configure the local minimizer (SLSQP) for basinhopping.
    minimizer_kwargs = {
        "method": "SLSQP",
        "bounds": param_bounds,
        "constraints": [nonlinear_constraint],
        "options": {'maxiter': 1500, 'ftol': 1e-6, 'disp': False}
    }

    # Instantiate the custom step-taking class for intelligent perturbations.
    custom_step = CustomTakeStep(stepsize_xy=0.05, stepsize_r=0.01, random_seed=RANDOM_SEED)

    best_circles = np.zeros((N_CIRCLES, 3), dtype=np.float64)
    try:
        # Use basinhopping to escape local minima and find a better global solution.
        res = basinhopping(
            objective,
            x0,
            minimizer_kwargs=minimizer_kwargs,
            niter=100,      # Number of hopping iterations
            T=0.001,        # Temperature for Metropolis acceptance criterion
            take_step=custom_step,
            seed=RANDOM_SEED
        )
        
        if res.lowest_optimization_result.success:
            best_circles = res.x.reshape(N_CIRCLES, 3)
        else:
            # Fallback if basinhopping fails to find a successful minimum
            fallback_params = generate_initial_guess_grid_perturbed(N_CIRCLES, seed=RANDOM_SEED)
            best_circles = fallback_params.reshape(N_CIRCLES, 3)

    except Exception:
        # General fallback for any unexpected errors during the optimization
        fallback_params = generate_initial_guess_grid_perturbed(N_CIRCLES, seed=RANDOM_SEED)
        best_circles = fallback_params.reshape(N_CIRCLES, 3)

    # Final check to ensure radii are strictly positive
    best_circles[:, 2] = np.maximum(best_circles[:, 2], MIN_RADIUS)

    return best_circles


# EVOLVE-BLOCK-END
