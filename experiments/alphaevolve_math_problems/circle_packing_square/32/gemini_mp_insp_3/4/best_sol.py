# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
from joblib import Parallel, delayed

# Constants for the problem
N_CIRCLES = 32
# Small value for radius and bounds to ensure positivity and strict containment.
# Helps with numerical stability and prevents circles from being exactly on boundaries.
EPSILON = 1e-8 # Using 1e-8 as in Inspiration 2 for robustness

def _extract_circles(params: np.ndarray) -> np.ndarray:
    """Unpacks the 1D parameter array into (N, 3) circles array (x, y, r)."""
    return params.reshape((N_CIRCLES, 3))

def objective_function(params: np.ndarray) -> float:
    """
    Objective function to minimize: negative sum of radii.
    The optimization aims to maximize the sum of radii, so we minimize its negative.
    params: flattened array [x1, y1, r1, ..., xN, yN, rN]
    """
    # Radii are located at indices 2, 5, 8, ... in the flattened array
    radii = params[2::3]
    return -np.sum(radii)

# Pre-calculate indices for non-overlap constraints to avoid recomputing in each call
# This is done once globally to optimize the `non_overlap_constraints` function.
_upper_triangle_indices = np.triu_indices(N_CIRCLES, k=1)

def containment_constraints(params: np.ndarray) -> np.ndarray:
    """
    Vectorized containment constraints. All values must be >= 0.
    r_i <= x_i  =>  x_i - r_i >= 0
    x_i <= 1 - r_i  =>  1 - x_i - r_i >= 0
    r_i <= y_i  =>  y_i - r_i >= 0
    y_i <= 1 - r_i  =>  1 - y_i - r_i >= 0
    """
    circles = _extract_circles(params)
    x, y, r = circles[:, 0], circles[:, 1], circles[:, 2]

    return np.concatenate([
        x - r,
        1 - x - r,
        y - r,
        1 - y - r
    ])

def non_overlap_constraints(params: np.ndarray) -> np.ndarray:
    """
    Vectorized non-overlap constraints. All values must be >= 0.
    (xi - xj)^2 + (yi - yj)^2 - (ri + rj)^2 >= 0
    """
    circles = _extract_circles(params)
    
    # Efficiently select all unique pairs of circles using pre-calculated indices
    circles_i = circles[_upper_triangle_indices[0]]
    circles_j = circles[_upper_triangle_indices[1]]
    
    pos_i, r_i = circles_i[:, :2], circles_i[:, 2]
    pos_j, r_j = circles_j[:, :2], circles_j[:, 2]

    # Calculate squared distances and squared sum of radii for all pairs at once
    dist_sq = np.sum((pos_i - pos_j)**2, axis=1)
    sum_radii_sq = (r_i + r_j)**2
    return dist_sq - sum_radii_sq

# Helper function for parallel execution. Must be defined at the top level
# of the module to be pickleable by joblib.
def _run_optimization_task(initial_guess, objective_func, bounds, constraints_list):
    """
    Worker function for a single optimization run using the SLSQP method.
    """
    res = minimize(objective_func, initial_guess, method='SLSQP', bounds=bounds, 
                   constraints=constraints_list,
                   options={'disp': False, 'maxiter': 3000, 'ftol': 1e-8}) # Increased maxiter, tighter ftol

    # Check for success or maxiter reached (status 2), as both can yield good solutions
    if res.success or res.status == 2:
        return res.x, -res.fun
    else:
        return None, -np.inf # Return None for solution and -inf for sum_radii if optimization failed


def circle_packing32() -> np.ndarray:
    """
    Places 32 non-overlapping circles in the unit square in order to maximize the sum of radii.

    Returns:
        circles: np.array of shape (32,3), where the i-th row (x,y,r) stores the (x,y) coordinates of the i-th circle of radius r.
    """
    # 1. Define Bounds for Variables (x, y, r for each circle)
    # x and y coordinates must be between EPSILON and 1-EPSILON.
    # Radii (r) must be positive (EPSILON) and cannot exceed 0.5.
    bounds_list = [(EPSILON, 1.0 - EPSILON), (EPSILON, 1.0 - EPSILON), (EPSILON, 0.5)] * N_CIRCLES
    
    # 2. Define Constraints as a list of dictionaries for SLSQP
    constraints = [
        {'type': 'ineq', 'fun': containment_constraints},
        {'type': 'ineq', 'fun': non_overlap_constraints}
    ]

    # 3. Initial Guess Generation and Parallel Optimization
    # Using diverse strategies and parallel restarts to thoroughly explore the solution space.
    num_restarts = 96 # Increased number of restarts for better exploration (from Inspiration 2)
    rng = np.random.RandomState(42) # Use a random state generator for reproducibility
    
    initial_guesses = []
    for i in range(num_restarts):
        x0_circles = np.zeros((N_CIRCLES, 3))

        if i % 3 == 0: # Strategy 1: Rectangular grid-based initial guess (4x8)
            rows, cols = 4, 8
            # Slightly larger initial radii, with a small perturbation
            grid_radii_base = min(0.5 / cols, 0.5 / rows) * 0.9 
            
            grid_x, grid_y = np.meshgrid(
                (np.arange(cols) + 0.5) / cols,
                (np.arange(rows) + 0.5) / rows
            )
            x0_circles[:, 0] = grid_x.flatten()
            x0_circles[:, 1] = grid_y.flatten()
            x0_circles[:, 2] = grid_radii_base
            
            # Add perturbation to positions and radii
            x0_circles[:, :2] += rng.uniform(-0.02, 0.02, size=(N_CIRCLES, 2))
            x0_circles[:, 2] *= rng.uniform(0.8, 1.2, size=N_CIRCLES)

        elif i % 3 == 1: # Strategy 2: Hexagonal grid approximation initial guess
            hex_rows = 6
            # Distribute 32 circles among 6 rows, alternating 5 and 6 circles
            circles_per_row = [5, 6, 5, 6, 5, 5] # Sums to 32
            
            # Initial radius estimation for hexagonal packing.
            r_hex_approx = 0.045 
            
            current_idx = 0
            for r_idx in range(hex_rows):
                num_circles_in_row = circles_per_row[r_idx]
                
                # Vertical position for the row
                cy = r_hex_approx + r_idx * r_hex_approx * np.sqrt(3)

                # Horizontal offset for staggering
                x_offset = 0 if r_idx % 2 == 0 else r_hex_approx 
                
                # Calculate horizontal positions for circles in this row
                for c_idx in range(num_circles_in_row):
                    if current_idx < N_CIRCLES: # Ensure we don't exceed 32 circles
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
            x0_circles[:, 2] = rng.uniform(0.008, 0.025, N_CIRCLES) # Random radii within a reasonable small range

        # Ensure initial guess respects the defined bounds
        x0_circles[:, 0] = np.clip(x0_circles[:, 0], EPSILON, 1 - EPSILON)
        x0_circles[:, 1] = np.clip(x0_circles[:, 1], EPSILON, 1 - EPSILON)
        x0_circles[:, 2] = np.clip(x0_circles[:, 2], EPSILON, 0.5)
        
        initial_guesses.append(x0_circles.flatten())

    # Run optimizations in parallel using all available CPU cores
    results = Parallel(n_jobs=-1)(
        delayed(_run_optimization_task)(x0, objective_function, bounds_list, constraints)
        for x0 in initial_guesses
    )

    # 4. Process Results
    best_sum_radii = -np.inf
    best_solution_flat = None

    for x_sol, sum_radii in results:
        if x_sol is not None and sum_radii > best_sum_radii: # Check x_sol for None from failed runs
            best_sum_radii = sum_radii
            best_solution_flat = x_sol
    
    if best_solution_flat is None:
        # Fallback if all optimizations failed
        print("Optimization failed for all initial guesses. Returning empty solution.")
        return np.zeros((N_CIRCLES, 3)) 
    
    return _extract_circles(best_solution_flat)


# EVOLVE-BLOCK-END
