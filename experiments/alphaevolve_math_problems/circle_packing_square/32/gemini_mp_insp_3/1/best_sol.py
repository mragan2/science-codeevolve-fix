# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize, Bounds
# Removed differential_evolution and pdist, squareform as they are not used in the new strategy.
from joblib import Parallel, delayed # Added for parallel execution, inspired by Inspiration 3

# --- Constants (adapted from Inspiration 3 for clarity and optimal ranges) ---
N_CIRCLES = 32
UNIT_SQUARE_SIZE = 1.0
R_MIN = 1e-6  # Minimum radius to avoid degenerate circles or numerical issues (from Insp1/3)
R_MAX = 0.5   # Maximum possible radius for a circle in a unit square (half side length)
EPSILON_CONSTRAINT = 1e-8 # Small tolerance for constraints to ensure strict non-overlap and containment (from Insp3)

# --- Helper functions (module level, adapted from Inspiration 3) ---
def _extract_circles(x_flat: np.ndarray) -> np.ndarray:
    """Helper to extract circles from the flattened parameter array."""
    return x_flat.reshape(N_CIRCLES, 3)

# --- Scipy.minimize objective and constraint functions (module level, vectorized, adapted from Inspiration 3) ---
def scipy_objective(x_flat: np.ndarray) -> float:
    """Objective function for scipy.minimize: negative sum of radii (to minimize)."""
    radii = x_flat[2::3] # Radii are at indices 2, 5, 8, ...
    return -np.sum(radii)

def scipy_containment_constraints(x_flat: np.ndarray) -> np.ndarray:
    """Non-linear inequality constraints for containment (g >= 0)."""
    circles = _extract_circles(x_flat)
    cx, cy, r = circles[:, 0], circles[:, 1], circles[:, 2]
    # Constraints: x-r >= 0, 1-x-r >= 0, y-r >= 0, 1-y-r >= 0
    return np.concatenate([cx - r, UNIT_SQUARE_SIZE - cx - r, cy - r, UNIT_SQUARE_SIZE - cy - r])

# Pre-calculate upper triangle indices for efficiency (from Inspiration 3)
_UPPER_TRIANGLE_INDICES = np.triu_indices(N_CIRCLES, k=1)

def scipy_overlap_constraints(x_flat: np.ndarray) -> np.ndarray:
    """Non-linear inequality constraints for non-overlap (g >= 0)."""
    circles = _extract_circles(x_flat)
    # Extract circles for unique pairs using pre-calculated indices
    circles_i = circles[_UPPER_TRIANGLE_INDICES[0]]
    circles_j = circles[_UPPER_TRIANGLE_INDICES[1]]
    
    pos_i, r_i = circles_i[:, :2], circles_i[:, 2]
    pos_j, r_j = circles_j[:, :2], circles_j[:, 2]
    
    # Calculate squared distance and squared sum of radii for all pairs
    dist_sq = np.sum((pos_i - pos_j)**2, axis=1)
    sum_radii_sq = (r_i + r_j)**2
    
    # Constraint: d_ij^2 - (ri + rj)^2 >= 0. Add EPSILON_CONSTRAINT for strictness (from Insp3).
    return dist_sq - sum_radii_sq + EPSILON_CONSTRAINT

# --- Validation function for the final solution (adapted from Inspiration 3) ---
def validate_final_solution(circles: np.ndarray, tolerance: float = EPSILON_CONSTRAINT) -> bool:
    """
    Validates a solution for containment and non-overlap.
    Returns True if valid, False otherwise.
    """
    radii = circles[:, 2]
    x, y = circles[:, 0], circles[:, 1]

    # 1. Check radii are positive and within R_MIN, R_MAX
    if np.any(radii < R_MIN - tolerance) or np.any(radii > R_MAX + tolerance):
        return False

    # 2. Check containment (x, y coordinates must be within [r, 1-r])
    if np.any(x - radii < -tolerance) or \
       np.any(x + radii > UNIT_SQUARE_SIZE + tolerance) or \
       np.any(y - radii < -tolerance) or \
       np.any(y + radii > UNIT_SQUARE_SIZE + tolerance):
        return False

    # 3. Check non-overlap (using squared distances for efficiency)
    centers = circles[:, :2]
    # Using broadcasting to compute all pairwise differences (dx, dy)
    diffs = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
    # Sum of squares along the last axis (dx^2 + dy^2)
    sq_distances = np.sum(diffs**2, axis=-1)
    
    radii_sum_matrix = radii[:, np.newaxis] + radii[np.newaxis, :]
    
    # Check for overlaps: (ri + rj)^2 - d_ij^2 > tolerance
    overlap_check = (radii_sum_matrix**2 - sq_distances) > tolerance
    np.fill_diagonal(overlap_check, False) # Ignore self-comparison
    
    if np.any(overlap_check):
        return False
        
    return True

# --- Parallel worker function (adapted from Inspiration 3) ---
def _run_optimization_task(initial_guess: np.ndarray, objective_func, bounds, constraints_list, rng_seed: int):
    """
    Worker function for a single optimization run using the SLSQP method.
    Includes tighter options and validation of the final result before returning.
    """
    # Set a local random seed for reproducibility if randomness is used within objective/constraints
    # (though typically not for SLSQP itself, but good practice for init_guess generation if done here)
    np.random.seed(rng_seed)

    # Tighter optimization options for more precise local optima (from Inspiration 3)
    options = {'disp': False, 'maxiter': 8000, 'ftol': 1e-10}

    # Use scipy.optimize.minimize with SLSQP method
    res = minimize(objective_func, initial_guess, method='SLSQP', bounds=bounds, 
                   constraints=constraints_list, options=options)

    # Check for success or maxiter reached (status 2 for SLSQP means maximum iterations reached)
    if res.success or res.status == 2:
        final_circles = _extract_circles(res.x)
        # Re-validate with a slightly relaxed tolerance for the optimizer's output (from Insp3)
        if validate_final_solution(final_circles, tolerance=1e-7):
            return res.x, -res.fun # Return solution and sum_radii
    return None, -np.inf # Indicate failure or invalid solution

def circle_packing32() -> np.ndarray:
    """
    Places 32 non-overlapping circles in the unit square to maximize the sum of radii.
    This version uses vectorized constraints, a faster SLSQP optimizer, and parallel restarts
    from diverse initial guesses, inspired by the best-performing solution.
    """
    # Define bounds for each variable [x1, y1, r1, x2, y2, r2, ...] (adapted from Insp3)
    bounds = []
    for _ in range(N_CIRCLES):
        bounds.extend([(0.0, UNIT_SQUARE_SIZE), (0.0, UNIT_SQUARE_SIZE), (R_MIN, R_MAX)])

    # Define non-linear constraints for scipy.minimize (g(x) >= 0)
    # These functions are defined globally above and are vectorized.
    constraints = [
        {'type': 'ineq', 'fun': scipy_containment_constraints},
        {'type': 'ineq', 'fun': scipy_overlap_constraints}
    ]

    # --- Generate Diverse Initial Guesses and Parallel Optimization (from Inspiration 3) ---
    num_restarts = 150 # Increased number of restarts for better exploration
    rng = np.random.RandomState(42) # Fixed seed for reproducibility
    
    initial_guesses = []
    for i in range(num_restarts):
        x0_circles = np.zeros((N_CIRCLES, 3))
        
        # Strategy 1: Rectangular grid-based initial guess (4x8)
        if i % 3 == 0: 
            rows, cols = 4, 8 # For 32 circles
            grid_radii = min(0.5 / cols, 0.5 / rows) * 0.9 # Base radius, slightly smaller to allow packing
            grid_x, grid_y = np.meshgrid((np.arange(cols) + 0.5) / cols, (np.arange(rows) + 0.5) / rows)
            
            x0_circles[:, 0] = grid_x.flatten()
            x0_circles[:, 1] = grid_y.flatten()
            x0_circles[:, 2] = grid_radii
            
            # Add some random perturbation
            x0_circles[:, :2] += rng.uniform(-0.02, 0.02, size=(N_CIRCLES, 2))
            x0_circles[:, 2] *= rng.uniform(0.8, 1.2, size=N_CIRCLES) # Perturb radii
            
        # Strategy 2: Hexagonal grid approximation
        elif i % 3 == 1:
            # Approximate hexagonal packing for 32 circles
            hex_rows = 6
            circles_per_row = [5, 6, 5, 6, 5, 5] # Total 32 circles
            r_hex_approx = 0.045 # Approximate radius for hexagonal packing
            
            current_idx = 0
            for r_idx in range(hex_rows):
                num_circles_in_row = circles_per_row[r_idx]
                cy = r_hex_approx + r_idx * r_hex_approx * np.sqrt(3) # Y-coordinate for row
                x_offset = 0 if r_idx % 2 == 0 else r_hex_approx # Stagger rows
                
                for c_idx in range(num_circles_in_row):
                    if current_idx < N_CIRCLES: # Ensure we don't exceed N_CIRCLES
                        cx = r_hex_approx + x_offset + c_idx * 2 * r_hex_approx
                        x0_circles[current_idx] = [cx, cy, r_hex_approx]
                        current_idx += 1
            
            # Add some random perturbation
            x0_circles[:, :2] += rng.uniform(-0.015, 0.015, size=(N_CIRCLES, 2))
            x0_circles[:, 2] *= rng.uniform(0.9, 1.1, size=N_CIRCLES)
            
        # Strategy 3: Fully random initial guess
        else:
            x0_circles[:, 0] = rng.rand(N_CIRCLES) # Random x
            x0_circles[:, 1] = rng.rand(N_CIRCLES) # Random y
            x0_circles[:, 2] = rng.uniform(0.008, 0.025, N_CIRCLES) # Small random radii
        
        # Clamp initial guesses to be within reasonable bounds
        x0_circles[:, 0] = np.clip(x0_circles[:, 0], R_MIN, UNIT_SQUARE_SIZE - R_MIN)
        x0_circles[:, 1] = np.clip(x0_circles[:, 1], R_MIN, UNIT_SQUARE_SIZE - R_MIN)
        x0_circles[:, 2] = np.clip(x0_circles[:, 2], R_MIN, R_MAX)
        initial_guesses.append(x0_circles.flatten())

    # Run optimizations in parallel using joblib
    # n_jobs=-1 uses all available CPU cores
    results = Parallel(n_jobs=-1)(
        delayed(_run_optimization_task)(x0, scipy_objective, Bounds([b[0] for b in bounds], [b[1] for b in bounds]), constraints, 42 + k) # Pass unique seed
        for k, x0 in enumerate(initial_guesses)
    )

    # --- Process Results ---
    best_sum_radii = -np.inf
    best_solution_flat = None
    for x_sol, sum_radii in results:
        if x_sol is not None and sum_radii > best_sum_radii:
            best_sum_radii = sum_radii
            best_solution_flat = x_sol
    
    if best_solution_flat is None:
        # Fallback if all optimizations failed, return a trivial valid solution
        print("Warning: All optimization restarts failed or produced invalid solutions. Returning fallback solution.")
        fallback_circles = np.zeros((N_CIRCLES, 3))
        fallback_circles[:, 0] = UNIT_SQUARE_SIZE / 2
        fallback_circles[:, 1] = UNIT_SQUARE_SIZE / 2
        fallback_circles[:, 2] = R_MIN
        return fallback_circles
    
    # Return the best valid solution found
    return _extract_circles(best_solution_flat)


# EVOLVE-BLOCK-END
