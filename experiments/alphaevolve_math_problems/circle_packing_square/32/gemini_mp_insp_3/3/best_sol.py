# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
from joblib import Parallel, delayed # For parallel execution of multiple restarts

# Define global constants for clarity and maintainability (inspired by Inspiration 1 & 2)
N_CIRCLES = 32
PARAM_PER_CIRCLE = 3  # (x, y, r)
UNIT_SQUARE_SIZE = 1.0
R_MIN = 1e-6  # Minimum radius to avoid degenerate circles or numerical issues
R_MAX = 0.5   # Maximum possible radius for a circle in a unit square (half side length)
EPSILON_CONSTRAINT = 1e-8 # Small tolerance for constraints to ensure strict non-overlap and containment

# Pre-calculate indices for non-overlap constraints to avoid recomputing in each call (from Insp 2/3)
_upper_triangle_indices = np.triu_indices(N_CIRCLES, k=1)

# Helper function for parallel execution. Must be defined at the top level
# of the module to be pickleable by joblib.
def _run_optimization_task(initial_guess, objective_func, bounds, constraints_list):
    """
    Worker function for a single optimization run using the SLSQP method.
    Optimized options based on best-performing inspirations (IP1/IP3).
    """
    # Options tuned for high performance in IP1/IP3
    options = {'disp': False, 'maxiter': 3000, 'ftol': 1e-8} 
    res = minimize(objective_func, initial_guess, method='SLSQP', bounds=bounds, 
                   constraints=constraints_list, options=options)

    # Check for success or maxiter reached (status 2, from Insp1/2/3), or iteration limit exceeded (status 9)
    if res.success or res.status == 2 or res.status == 9:
        return res.x, -res.fun 
    else:
        return None, -np.inf # Return None for solution and -inf for sum_radii if optimization failed


def _extract_circles(params: np.ndarray) -> np.ndarray:
    """Unpacks the 1D parameter array into (N, 3) circles array (x, y, r)."""
    return params.reshape((N_CIRCLES, PARAM_PER_CIRCLE))

def objective_function(params: np.ndarray) -> float:
    """
    Objective function to minimize: negative sum of radii.
    The optimization aims to maximize the sum of radii, so we minimize its negative.
    params: flattened array [x1, y1, r1, ..., xN, yN, rN]
    """
    radii = params[2::PARAM_PER_CIRCLE] # Radii are at indices 2, 5, 8, ...
    return -np.sum(radii)

def containment_constraints(params: np.ndarray) -> np.ndarray:
    """
    Vectorized containment constraints. All values must be >= 0.
    r_i <= x_i  =>  x_i - r_i >= 0
    x_i <= 1 - r_i  =>  1 - x_i - r_i >= 0
    r_i <= y_i  =>  y_i - r_i >= 0
    y_i <= 1 - r_i  =>  1 - y_i - r_i >= 0
    (Consistent with Inspiration 1/2)
    """
    circles = _extract_circles(params)
    x, y, r = circles[:, 0], circles[:, 1], circles[:, 2]

    return np.concatenate([
        x - r,
        UNIT_SQUARE_SIZE - x - r,
        y - r,
        UNIT_SQUARE_SIZE - y - r
    ])

def calculate_pairwise_distances_sq(circles: np.ndarray):
    """Calculates squared pairwise Euclidean distances between circle centers. (From Insp 1)"""
    centers = circles[:, :2]
    # Using broadcasting to compute all pairwise differences (dx, dy)
    diffs = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
    # Sum of squares along the last axis (dx^2 + dy^2)
    sq_distances = np.sum(diffs**2, axis=-1)
    return sq_distances

def non_overlap_constraints(params: np.ndarray) -> np.ndarray:
    """
    Vectorized non-overlap constraints. All values must be >= 0.
    (xi - xj)² + (yi - yj)² - (ri + rj)² >= 0
    (Adapted from Inspiration 1/2/3 using pre-calculated indices and EPSILON_CONSTRAINT)
    """
    circles = _extract_circles(params)
    radii = circles[:, 2]
    
    sq_distances = calculate_pairwise_distances_sq(circles) # Use helper
    
    # Extract relevant squared distances for unique pairs using pre-calculated indices
    d_sq_pairs = sq_distances[_upper_triangle_indices]
    
    # Calculate sums of radii for all unique pairs
    radii_sum_matrix = radii[:, np.newaxis] + radii[np.newaxis, :]
    r_sum_pairs = radii_sum_matrix[_upper_triangle_indices]

    # Constraint: d_ij^2 - (ri + rj)^2 >= 0. Add EPSILON_CONSTRAINT for numerical stability.
    return d_sq_pairs - r_sum_pairs**2 + EPSILON_CONSTRAINT

def validate_solution(circles: np.ndarray, tolerance: float = EPSILON_CONSTRAINT) -> bool:
    """
    Strictly validates if a given set of circles meets all constraints within a tolerance.
    (Adapted from Inspiration 1/2, using the efficient distance calculation helper)
    """
    x, y, r = circles[:, 0], circles[:, 1], circles[:, 2]

    # 1. Radius validity check
    if np.any(r < R_MIN - tolerance) or np.any(r > R_MAX + tolerance):
        return False
    
    # 2. Containment check
    if np.any(x - r < -tolerance) or np.any(x + r > UNIT_SQUARE_SIZE + tolerance):
        return False
    if np.any(y - r < -tolerance) or np.any(y + r > UNIT_SQUARE_SIZE + tolerance):
        return False

    # 3. Non-overlap check
    sq_distances = calculate_pairwise_distances_sq(circles) # Use the efficient helper
    
    radii_sum_matrix = r[:, np.newaxis] + r[np.newaxis, :]
    
    # Check for overlaps: (ri + rj)^2 - d_ij^2 > tolerance
    # Use the pre-calculated upper triangle indices for efficiency
    actual_distances_sq = sq_distances[_upper_triangle_indices]
    required_distances_sq = radii_sum_matrix[_upper_triangle_indices]**2
    
    if np.any(actual_distances_sq < required_distances_sq - tolerance):
        return False
        
    return True

def circle_packing32() -> np.ndarray:
    """
    Places 32 non-overlapping circles in the unit square in order to maximize the sum of radii.
    This version uses efficient single-stage parallel optimization with diverse initial guesses
    and robust validation.
    """
    # 1. Define Bounds for Variables (x, y, r for each circle)
    bounds_list = [(R_MIN, UNIT_SQUARE_SIZE - R_MIN), (R_MIN, UNIT_SQUARE_SIZE - R_MIN), (R_MIN, R_MAX)] * N_CIRCLES
    
    # 2. Define Constraints as a list of dictionaries for SLSQP
    constraints = [
        {'type': 'ineq', 'fun': containment_constraints},
        {'type': 'ineq', 'fun': non_overlap_constraints}
    ]

    # 3. Initial Guess Generation and Parallel Optimization
    # Using diverse strategies and parallel restarts to thoroughly explore the solution space.
    num_restarts = 96 # Number of restarts, balanced for performance and exploration (from Insp1/2/3)
    rng = np.random.RandomState(42) # Use a random state generator for reproducibility
    
    initial_guesses = []
    for i in range(num_restarts):
        x0_circles = np.zeros((N_CIRCLES, PARAM_PER_CIRCLE))

        if i % 3 == 0: # Strategy 1: Rectangular grid-based initial guess (4x8)
            rows, cols = 4, 8
            grid_radii_base = min(0.5 / cols, 0.5 / rows) * 0.95 
            
            grid_x, grid_y = np.meshgrid(
                (np.arange(cols) + 0.5) / cols,
                (np.arange(rows) + 0.5) / rows
            )
            x0_circles[:, 0] = grid_x.flatten()
            x0_circles[:, 1] = grid_y.flatten()
            x0_circles[:, 2] = grid_radii_base
            
            # Add perturbation
            x0_circles[:, :2] += rng.uniform(-0.02, 0.02, size=(N_CIRCLES, 2))
            x0_circles[:, 2] *= rng.uniform(0.8, 1.2, size=N_CIRCLES)

        elif i % 3 == 1: # Strategy 2: Hexagonal grid approximation initial guess (from Insp1/2)
            hex_rows = 6
            circles_per_row = [5, 6, 5, 6, 5, 5] # Sums to 32
            
            r_hex_approx = 0.045 # Initial radius estimation for hexagonal packing.
            
            current_idx = 0
            for r_idx in range(hex_rows):
                num_circles_in_row = circles_per_row[r_idx]
                cy = r_hex_approx + r_idx * r_hex_approx * np.sqrt(3)
                x_offset = 0 if r_idx % 2 == 0 else r_hex_approx 
                
                for c_idx in range(num_circles_in_row):
                    if current_idx < N_CIRCLES:
                        cx = r_hex_approx + x_offset + c_idx * 2 * r_hex_approx
                        x0_circles[current_idx, 0] = cx
                        x0_circles[current_idx, 1] = cy
                        x0_circles[current_idx, 2] = r_hex_approx
                        current_idx += 1
            
            # Add perturbation to hex packing
            x0_circles[:, :2] += rng.uniform(-0.015, 0.015, size=(N_CIRCLES, 2))
            x0_circles[:, 2] *= rng.uniform(0.9, 1.1, size=N_CIRCLES)

        else: # Strategy 3: Fully random initial guess
            x0_circles[:, 0] = rng.rand(N_CIRCLES)
            x0_circles[:, 1] = rng.rand(N_CIRCLES)
            x0_circles[:, 2] = rng.uniform(0.008, 0.025, N_CIRCLES)

        # Ensure initial guess respects the defined bounds
        x0_circles[:, 0] = np.clip(x0_circles[:, 0], R_MIN, UNIT_SQUARE_SIZE - R_MIN)
        x0_circles[:, 1] = np.clip(x0_circles[:, 1], R_MIN, UNIT_SQUARE_SIZE - R_MIN)
        x0_circles[:, 2] = np.clip(x0_circles[:, 2], R_MIN, R_MAX)
        
        initial_guesses.append(x0_circles.flatten())

    print(f"Starting parallel optimization with {num_restarts} restarts using SLSQP.")
    results = Parallel(n_jobs=-1)(
        delayed(_run_optimization_task)(x0, objective_function, bounds_list, constraints)
        for x0 in initial_guesses
    )

    # 4. Process Results
    best_sum_radii = -np.inf
    best_solution_flat = None

    for x_sol, sum_radii in results:
        if x_sol is not None and sum_radii > best_sum_radii:
            best_sum_radii = sum_radii
            best_solution_flat = x_sol
    
    if best_solution_flat is None:
        print("Optimization failed for all initial guesses. Returning empty solution.")
        return np.zeros((N_CIRCLES, PARAM_PER_CIRCLE)) 
    
    optimal_circles = _extract_circles(best_solution_flat) # Extract circles from the best flat solution

    # Post-processing: Strictly enforce constraints on the final solution (from Insp2)
    # This corrects for any minor numerical inaccuracies from the optimizer.
    optimal_circles[:, 2] = np.clip(optimal_circles[:, 2], R_MIN, R_MAX) # Clip radii first
    optimal_circles[:, 0] = np.clip(optimal_circles[:, 0], optimal_circles[:, 2], UNIT_SQUARE_SIZE - optimal_circles[:, 2])
    optimal_circles[:, 1] = np.clip(optimal_circles[:, 1], optimal_circles[:, 2], UNIT_SQUARE_SIZE - optimal_circles[:, 2])

    # Final validation and print statements for clarity
    print(f"\nOptimization finished.")
    print(f"Final best sum of radii found: {best_sum_radii:.8f}")

    if not validate_solution(optimal_circles):
        print("ERROR: Final solution failed strict validation! This indicates a problem with the optimization or validation tolerances.")
    else:
        print("Final solution appears to satisfy constraints.")

    return optimal_circles


# EVOLVE-BLOCK-END
