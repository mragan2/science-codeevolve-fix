# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize, differential_evolution
from numba import njit, float64, int32


# Numba-optimized function to calculate sum of squared overlap violations
@njit(float64(float64[:], int32))
def _calculate_overlap_violations_sq(params, n_circles):
    violations = 0.0
    for i in range(n_circles):
        xi, yi, ri = params[i*3:i*3+3]
        for j in range(i + 1, n_circles):
            xj, yj, rj = params[j*3:j*3+3]
            
            dx = xi - xj
            dy = yi - yj
            dist_sq = dx*dx + dy*dy
            min_dist_sq = (ri + rj)**2
            
            # Violation is positive if circles overlap
            violation = min_dist_sq - dist_sq
            if violation > 0:
                violations += violation**2 # Sum of squared positive violations
    return violations

# Objective function for scipy.optimize optimizers
def objective_function(params, n_circles, penalty_factor=1000):
    circles = params.reshape((n_circles, 3))
    radii = circles[:, 2]
    
    # Primary objective: maximize sum of radii (minimize negative sum)
    sum_radii = np.sum(radii)
    
    # Penalty for boundary violations (squared violations)
    x_coords = circles[:, 0]
    y_coords = circles[:, 1]
    
    x_min_violation = np.maximum(0, radii - x_coords)
    x_max_violation = np.maximum(0, x_coords + radii - 1)
    y_min_violation = np.maximum(0, radii - y_coords)
    y_max_violation = np.maximum(0, y_coords + radii - 1)
    
    boundary_violations_sq = np.sum(x_min_violation**2 + x_max_violation**2 + y_min_violation**2 + y_max_violation**2)
    
    # Penalty for overlap violations (using numba function for efficiency)
    overlap_violations_sq = _calculate_overlap_violations_sq(params, n_circles)
    
    total_penalty = penalty_factor * (boundary_violations_sq + overlap_violations_sq)
    
    return -sum_radii + total_penalty

# Numba-optimized function to calculate all constraint values for SLSQP
@njit(float64[:](float64[:], int32))
def _evaluate_slsqp_constraints(params, n_circles):
    num_boundary_constraints = 4 * n_circles
    num_overlap_constraints = n_circles * (n_circles - 1) // 2
    total_constraints = num_boundary_constraints + num_overlap_constraints
    
    constraint_values = np.empty(total_constraints, dtype=np.float64)
    
    offset = 0
    for i in range(n_circles):
        xi, yi, ri = params[i*3:i*3+3]
        
        constraint_values[offset] = xi - ri  # x_i - r_i >= 0
        offset += 1
        constraint_values[offset] = 1 - xi - ri # 1 - x_i - r_i >= 0
        offset += 1
        constraint_values[offset] = yi - ri  # y_i - r_i >= 0
        offset += 1
        constraint_values[offset] = 1 - yi - ri # 1 - y_i - r_i >= 0
        offset += 1
        
    for i in range(n_circles):
        xi, yi, ri = params[i*3:i*3+3]
        for j in range(i + 1, n_circles):
            xj, yj, rj = params[j*3:j*3+3]
            
            dx = xi - xj
            dy = yi - yj
            dist_sq = dx*dx + dy*dy
            min_dist_sq = (ri + rj)**2
            
            constraint_values[offset] = dist_sq - min_dist_sq # >= 0
            offset += 1
            
    return constraint_values

# Constraints for scipy.optimize.minimize (SLSQP method)
def create_slsqp_constraints(n_circles):
    # This function now returns a single constraint dictionary,
    # with 'fun' pointing to the Numba-optimized function for all constraints.
    return [{'type': 'ineq', 'fun': lambda params: _evaluate_slsqp_constraints(params, n_circles)}]

# Initial guess generation function
def generate_initial_guess(n_circles, initial_radius_factor=0.5):
    """
    Generates an initial guess for circle positions and radii.
    Arranges circles in a grid-like pattern, with a specialized pattern for N=26.
    """
    if n_circles == 26:
        # Custom initial configuration for N=26: a 5x5 grid with one additional circle.
        initial_params = []
        num_rows = 5
        num_cols = 5
        
        # Max radius for 5 circles across, if they were touching edge-to-edge
        r_grid_max = 1.0 / (2.0 * num_cols) # r_grid_max = 0.1
        
        # Use a factor to allow space for growth, starting with larger circles
        r_init_main = r_grid_max * initial_radius_factor 
        
        # Generate 25 circles in a 5x5 grid
        x_centers = np.linspace(r_init_main, 1 - r_init_main, num_cols)
        y_centers = np.linspace(r_init_main, 1 - r_init_main, num_rows)
        
        for y in y_centers:
            for x in x_centers:
                initial_params.extend([x, y, r_init_main])
        
        # Place the 26th circle in a central interstitial gap.
        # For a 5x5 grid with r=0.1, the center (0.5, 0.5) is occupied.
        # Interstitial gaps are at (0.2, 0.2), (0.2, 0.4), etc.
        # Let's place it at (0.2, 0.2) or a similar central gap.
        # Radius for this circle will be smaller.
        r_init_extra = r_grid_max * 0.5 * initial_radius_factor # Half the main radius
        initial_params.extend([0.2, 0.2, r_init_extra])
        
        return np.array(initial_params)
    
    else:
        # Generic grid for other n_circles
        side = int(np.ceil(np.sqrt(n_circles)))
        num_cols = side
        num_rows = side
        while num_cols * num_rows < n_circles:
            if num_cols <= num_rows:
                num_cols += 1
            else:
                num_rows += 1
        
        # Calculate a safe initial radius based on grid dimensions for generic case
        r_init_max = 1.0 / (2.0 * max(num_cols, num_rows)) # More precise max radius
        r_init = r_init_max * initial_radius_factor
        r_init = max(r_init, 1e-6) # Ensure radius is not too small or zero
        
        # Generate positions for the grid
        # Centers are offset by r_init to be within bounds
        x_centers = np.linspace(r_init, 1 - r_init, num_cols)
        y_centers = np.linspace(r_init, 1 - r_init, num_rows)
        
        initial_params = []
        count = 0
        for y in y_centers:
            for x in x_centers:
                if count < n_circles:
                    initial_params.extend([x, y, r_init])
                    count += 1
                else:
                    break
            if count == n_circles:
                break
                
        # Fill remaining with a central position and small radius if needed (shouldn't happen with above logic)
        while count < n_circles:
            initial_params.extend([0.5, 0.5, r_init / 2.0]) 
            count += 1

        return np.array(initial_params)

# Helper function to validate the solution against all constraints
def validate_solution(circles_array, tol=1e-6):
    """
    Validates a given array of circles for containment and non-overlap.
    Returns: (is_valid, sum_radii, total_violation)
    """
    n = circles_array.shape[0]
    sum_radii = np.sum(circles_array[:, 2])
    
    is_valid = True
    total_violation_sq = 0.0 # Sum of squared violations

    # Check containment constraints
    for i in range(n):
        x, y, r = circles_array[i]
        
        # Check if r is positive
        if r < tol:
            is_valid = False
            total_violation_sq += (tol - r)**2 # Penalize very small/negative radii
            
        # x - r >= 0
        if x - r < -tol:
            is_valid = False
            total_violation_sq += (x - r)**2
        # 1 - x - r >= 0
        if 1 - x - r < -tol:
            is_valid = False
            total_violation_sq += (1 - x - r)**2
        # y - r >= 0
        if y - r < -tol:
            is_valid = False
            total_violation_sq += (y - r)**2
        # 1 - y - r >= 0
        if 1 - y - r < -tol:
            is_valid = False
            total_violation_sq += (1 - y - r)**2
            
    # Check non-overlap constraints
    # Use numba-optimized function for efficiency
    overlap_violations_sq = _calculate_overlap_violations_sq(circles_array.flatten(), n)
    if overlap_violations_sq > tol:
        is_valid = False
        total_violation_sq += overlap_violations_sq

    return is_valid, sum_radii, total_violation_sq

def circle_packing26() -> np.ndarray:
    """
    Places 26 non-overlapping circles in the unit square in order to maximize the sum of radii.

    Returns:
        circles: np.array of shape (26,3), where the i-th row (x,y,r) stores the (x,y) coordinates of the i-th circle of radius r.
    """
    n = 26
    
    # Set a fixed random seed for reproducibility
    np.random.seed(42)

    # Bounds for x, y, r for each circle variable
    # x, y in [0, 1], r in [epsilon, 0.5]
    # These are general variable bounds; specific containment constraints handle r's effect on x,y ranges.
    bounds = [(0.0, 1.0), (0.0, 1.0), (1e-6, 0.5)] * n

    # Generate an initial guess for circle positions and radii
    # Using a factor of 0.95 for initial radius to start with larger circles and allow less room for initial growth,
    # as the initial guess is already denser.
    initial_guess_params = generate_initial_guess(n, initial_radius_factor=0.95)
    
    # Stage 1: Global optimization using Differential Evolution
    # This helps explore the search space broadly to find a good starting region.
    print("Starting Differential Evolution (global search)...")
    de_result = differential_evolution(
        func=objective_function,
        bounds=bounds,
        args=(n, 5000), # Pass n_circles and a higher penalty factor for DE
        strategy='best1bin',
        maxiter=2000, # Increased maxiter for better exploration
        popsize=20,   # Larger population for better exploration
        tol=0.001,    # Tolerance for convergence
        mutation=(0.5, 1),
        recombination=0.7,
        seed=42,
        disp=True,    # Display progress
        workers=-1    # Use all available CPU cores for parallelization
    )
    print(f"Differential Evolution finished. Best objective value: {-de_result.fun:.6f}")
    
    # Use the result from Differential Evolution as the initial guess for local optimization
    refined_initial_guess = de_result.x
    
    # Stage 2: Local optimization using SLSQP with explicit constraints
    # SLSQP is suitable for constrained problems and refines the solution locally.
    print("Starting SLSQP (local refinement)...")
    slsqp_constraints = create_slsqp_constraints(n)
    
    slsqp_result = minimize(
        fun=objective_function, # Use the same objective with a moderate penalty factor
        x0=refined_initial_guess,
        args=(n, 1000), # Moderate penalty factor as constraints are explicitly handled
        method='SLSQP',
        bounds=bounds,
        constraints=slsqp_constraints,
        options={'maxiter': 3000, 'ftol': 1e-8, 'disp': True} # Increased maxiter, tighter tolerance
    )
    print(f"SLSQP finished. Best objective value: {-slsqp_result.fun:.6f}")
    
    # Final processing of the solution
    final_params = slsqp_result.x.reshape((n, 3))
    
    # Validate the final solution to ensure all constraints are met within tolerance
    is_valid, sum_radii, total_violation = validate_solution(final_params)
    
    # If the solution is not perfectly valid, attempt a final local refinement pass
    # with a very high penalty to force constraint satisfaction.
    if not is_valid and total_violation > 1e-7: # Check against a small tolerance
        print(f"WARNING: Final solution from optimizer has minor violations (total violation: {total_violation:.2e}).")
        print("Attempting a second SLSQP pass to resolve minor violations...")
        
        slsqp_result_recheck = minimize(
            fun=objective_function,
            x0=final_params.flatten(),
            args=(n, 100000), # Significantly increase penalty to strongly enforce feasibility
            method='SLSQP',
            bounds=bounds,
            constraints=slsqp_constraints,
            options={'maxiter': 1500, 'ftol': 1e-9, 'disp': True} # Tighter tolerance
        )
        final_params = slsqp_result_recheck.x.reshape((n,3))
        is_valid, sum_radii, total_violation = validate_solution(final_params)
        
        if not is_valid:
            print(f"WARNING: Solution still has violations after re-check (total violation: {total_violation:.2e}). This might indicate a hard-to-satisfy constraint or numerical precision issues.")
        else:
            print("Second SLSQP pass successfully resolved violations.")
    
    print(f"Final sum of radii: {sum_radii:.6f}")
    print(f"Is solution valid (within {1e-6} tolerance): {is_valid}")

    return final_params


# EVOLVE-BLOCK-END
