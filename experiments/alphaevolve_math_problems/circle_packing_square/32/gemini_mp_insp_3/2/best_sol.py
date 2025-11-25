# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
from joblib import Parallel, delayed

# Define global constants for clarity and maintainability (inspired by Inspiration 2/3)
N_CIRCLES = 32
UNIT_SQUARE_SIZE = 1.0
R_MIN = 1e-6  # Minimum radius to avoid degenerate circles or numerical issues
R_MAX = 0.5   # Maximum possible radius for a circle in a unit square (half side length)
EPSILON_CONSTRAINT = 1e-8 # Small tolerance for constraints to ensure strict non-overlap and containment

# --- Helper functions (moved to module level for clarity and encapsulation) ---
def _extract_circles(x_flat: np.ndarray) -> np.ndarray:
    """Helper to extract circles from the flattened parameter array."""
    return x_flat.reshape(N_CIRCLES, 3)

# --- Scipy.minimize objective and constraint functions (module level) ---
def scipy_objective(x_flat: np.ndarray) -> float:
    """Objective function for scipy.minimize: negative sum of radii (to minimize)."""
    radii = x_flat[2::3]
    return -np.sum(radii)

def scipy_containment_constraints(x_flat: np.ndarray) -> np.ndarray:
    """Non-linear inequality constraints for containment (g >= 0)."""
    circles = _extract_circles(x_flat)
    cx, cy, r = circles[:, 0], circles[:, 1], circles[:, 2]
    return np.concatenate([cx - r, UNIT_SQUARE_SIZE - cx - r, cy - r, UNIT_SQUARE_SIZE - cy - r])

# Pre-calculate upper triangle indices for efficiency (inspired by Inspiration 1/3)
_UPPER_TRIANGLE_INDICES = np.triu_indices(N_CIRCLES, k=1)

def scipy_overlap_constraints(x_flat: np.ndarray) -> np.ndarray:
    """Non-linear inequality constraints for non-overlap (g >= 0)."""
    circles = _extract_circles(x_flat)
    circles_i = circles[_UPPER_TRIANGLE_INDICES[0]]
    circles_j = circles[_UPPER_TRIANGLE_INDICES[1]]
    pos_i, r_i = circles_i[:, :2], circles_i[:, 2]
    pos_j, r_j = circles_j[:, :2], circles_j[:, 2]
    dist_sq = np.sum((pos_i - pos_j)**2, axis=1)
    sum_radii_sq = (r_i + r_j)**2
    return dist_sq - sum_radii_sq + EPSILON_CONSTRAINT

# --- Validation function for the final solution ---
def validate_final_solution(circles: np.ndarray, tolerance: float = EPSILON_CONSTRAINT) -> bool:
    """
    Validates a solution for containment and non-overlap.
    Returns True if valid, False otherwise.
    """
    radii = circles[:, 2]
    x, y = circles[:, 0], circles[:, 1]
    if np.any(radii < R_MIN - tolerance) or np.any(radii > R_MAX + tolerance): return False
    if np.any(x - radii < -tolerance) or np.any(x + radii > UNIT_SQUARE_SIZE + tolerance) or \
       np.any(y - radii < -tolerance) or np.any(y + radii > UNIT_SQUARE_SIZE + tolerance): return False
    centers = circles[:, :2]
    diffs = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
    sq_distances = np.sum(diffs**2, axis=-1)
    radii_sum_matrix = radii[:, np.newaxis] + radii[np.newaxis, :]
    overlap_check = (radii_sum_matrix**2 - sq_distances) > tolerance
    np.fill_diagonal(overlap_check, False)
    if np.any(overlap_check): return False
    return True

# --- Parallel worker function (enhanced with inspiration from 1/3 and further tuning) ---
def _run_optimization_task(initial_guess, objective_func, bounds, constraints_list):
    """
    Worker function for a single optimization run using the SLSQP method.
    Includes tighter options and validation of the final result before returning.
    """
    # Tighter optimization options for more precise local optima.
    # 'maxiter' increased to 8000 and 'ftol' to 1e-10 from typical 3000, 1e-8
    # to push for higher accuracy, as seen in successful advanced implementations.
    options = {'disp': False, 'maxiter': 8000, 'ftol': 1e-10}

    res = minimize(objective_func, initial_guess, method='SLSQP', bounds=bounds, 
                   constraints=constraints_list, options=options)

    # Check for success (res.success) or maximum iterations reached (res.status == 2),
    # as both can yield potentially good solutions.
    if res.success or res.status == 2:
        final_circles = _extract_circles(res.x)
        # Re-validate with a slightly relaxed tolerance (1e-7) compared to the
        # strict constraint epsilon (1e-8) to account for floating-point inaccuracies
        # in the optimizer's solution, which is a common practice.
        if validate_final_solution(final_circles, tolerance=1e-7):
            return res.x, -res.fun # Return the solution and the positive sum of radii
    return None, -np.inf # Indicate failure or an invalid solution with a very low sum of radii

def circle_packing32() -> np.ndarray:
    """
    Places 32 non-overlapping circles in the unit square to maximize the sum of radii.
    This version uses vectorized constraints, a faster SLSQP optimizer, and parallel restarts.
    """
    # --- 1. Constraints and Bounds ---
    constraints = [
        {'type': 'ineq', 'fun': scipy_containment_constraints},
        {'type': 'ineq', 'fun': scipy_overlap_constraints}
    ]

    # Bounds for each variable (x, y, r) for all N_CIRCLES.
    # x, y are within [0, UNIT_SQUARE_SIZE]. r is within [R_MIN, R_MAX].
    bounds = [(0.0, UNIT_SQUARE_SIZE), (0.0, UNIT_SQUARE_SIZE), (R_MIN, R_MAX)] * N_CIRCLES

    # --- 2. Initial Guesses and Parallel Optimization ---
    # Increased number of restarts to 100 for better exploration of the non-convex search space.
    num_restarts = 120 # Increased number of restarts for better exploration of the non-convex search space.
    rng = np.random.RandomState(42) # Fixed random seed for reproducibility
    
    initial_guesses = []
    for i in range(num_restarts):
        x0_circles = np.zeros((N_CIRCLES, 3))
        strategy = i % 4 # Cycle through 4 different initial guess strategies

        if strategy == 0: # Strategy 1: Rectangular grid-based initial guess (4x8)
            rows, cols = 4, 8
            grid_radii = min(0.5 / cols, 0.5 / rows) * 0.9 # Base radius for grid
            grid_x, grid_y = np.meshgrid((np.arange(cols) + 0.5) / cols, (np.arange(rows) + 0.5) / rows)
            x0_circles[:, 0] = grid_x.flatten()
            x0_circles[:, 1] = grid_y.flatten()
            x0_circles[:, 2] = grid_radii
            # Add perturbation to positions and radii
            x0_circles[:, :2] += rng.uniform(-0.02, 0.02, size=(N_CIRCLES, 2))
            x0_circles[:, 2] *= rng.uniform(0.8, 1.2, size=N_CIRCLES)
        elif strategy == 1: # Strategy 2: Hexagonal grid approximation
            hex_rows, circles_per_row, r_hex_approx = 6, [5, 6, 5, 6, 5, 5], 0.045
            current_idx = 0
            for r_idx in range(hex_rows):
                num_circles_in_row = circles_per_row[r_idx]
                cy = r_hex_approx + r_idx * r_hex_approx * np.sqrt(3)
                x_offset = 0 if r_idx % 2 == 0 else r_hex_approx
                for c_idx in range(num_circles_in_row):
                    if current_idx < N_CIRCLES:
                        cx = r_hex_approx + x_offset + c_idx * 2 * r_hex_approx
                        x0_circles[current_idx] = [cx, cy, r_hex_approx]
                        current_idx += 1
            # Add perturbation to positions and radii
            x0_circles[:, :2] += rng.uniform(-0.015, 0.015, size=(N_CIRCLES, 2))
            x0_circles[:, 2] *= rng.uniform(0.9, 1.1, size=N_CIRCLES)
        elif strategy == 2: # Strategy 3: Sunflower seed packing (from Inspiration 3)
            indices = np.arange(0, N_CIRCLES, dtype=float) + 0.5
            phi = np.arccos(1 - 2*indices/N_CIRCLES)
            theta = np.pi * (1 + 5**0.5) * indices
            r_samples = np.sqrt(indices / N_CIRCLES)
            # Scale to fit unit square roughly, and center it
            x0_circles[:, 0] = (r_samples * np.cos(theta) * 0.4 + 0.5) 
            x0_circles[:, 1] = (r_samples * np.sin(theta) * 0.4 + 0.5)
            x0_circles[:, 2] = 0.04 # A reasonable starting radius for sunflower
            x0_circles[:, :2] += rng.uniform(-0.01, 0.01, size=(N_CIRCLES, 2)) # Perturb positions
            x0_circles[:, 2] *= rng.uniform(0.8, 1.1, size=N_CIRCLES) # Perturb radii
        else: # Strategy 4: Fully random initial guess
            x0_circles[:, 0] = rng.rand(N_CIRCLES)
            x0_circles[:, 1] = rng.rand(N_CIRCLES)
            x0_circles[:, 2] = rng.uniform(0.008, 0.025, N_CIRCLES) # Random radii within a reasonable range
        
        # Ensure initial guess respects the defined bounds by clipping values
        x0_circles[:, 0] = np.clip(x0_circles[:, 0], R_MIN, UNIT_SQUARE_SIZE - R_MIN)
        x0_circles[:, 1] = np.clip(x0_circles[:, 1], R_MIN, UNIT_SQUARE_SIZE - R_MIN)
        x0_circles[:, 2] = np.clip(x0_circles[:, 2], R_MIN, R_MAX)
        initial_guesses.append(x0_circles.flatten())

    # Run optimizations in parallel using all available CPU cores
    results = Parallel(n_jobs=-1)(
        delayed(_run_optimization_task)(x0, scipy_objective, bounds, constraints)
        for x0 in initial_guesses
    )

    # --- 3. Process Results ---
    best_sum_radii = -np.inf
    best_solution_flat = None
    for x_sol, sum_radii in results:
        if x_sol is not None and sum_radii > best_sum_radii:
            best_sum_radii = sum_radii
            best_solution_flat = x_sol
    
    if best_solution_flat is None:
        # Fallback if all optimizations failed: return N_CIRCLES tiny circles at the center.
        # This ensures a valid (though suboptimal) configuration is always returned.
        fallback_circles = np.zeros((N_CIRCLES, 3))
        fallback_circles[:, 0] = UNIT_SQUARE_SIZE / 2
        fallback_circles[:, 1] = UNIT_SQUARE_SIZE / 2
        fallback_circles[:, 2] = R_MIN
        return fallback_circles
    
    # The best solution obtained has already been validated within _run_optimization_task.
    return _extract_circles(best_solution_flat)


# EVOLVE-BLOCK-END
