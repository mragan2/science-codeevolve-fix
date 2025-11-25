# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize, basinhopping # Removed differential_evolution, NonlinearConstraint (no longer directly used for DE)
import numba

# Define the number of circles
N_CIRCLES = 26

# Helper to unpack parameters from the flattened array
@numba.jit(nopython=True)
def _unpack_params_numba(params_flat):
    """Unpacks a flattened parameter array into x, y coordinates and radii."""
    x = params_flat[0::3]
    y = params_flat[1::3]
    r = params_flat[2::3]
    return x, y, r

# Objective function: negative sum of radii (since scipy.optimize minimizes)
def _objective(params_flat):
    """Calculates the negative sum of radii for the given circle parameters."""
    x, y, r = _unpack_params_numba(params_flat)
    return -np.sum(r)

# Constraint function: returns an array of values that must be >= 0
# This includes boundary containment and non-overlap conditions.
@numba.jit(nopython=True)
def _constraint_values_numba(params_flat):
    """
    Calculates the values for all constraints.
    All returned values must be >= 0 for a valid packing.
    """
    n = N_CIRCLES
    x, y, r = _unpack_params_numba(params_flat)

    # Pre-allocate array for constraints for efficiency
    # 4 constraints per circle for boundary (x-r, 1-x-r, y-r, 1-y-r)
    # n*(n-1)/2 constraints for non-overlap (dist_sq - (ri+rj)_sq)
    num_boundary_constraints = 4 * n
    num_overlap_constraints = n * (n - 1) // 2
    total_constraints = num_boundary_constraints + num_overlap_constraints

    constraints = np.empty(total_constraints, dtype=params_flat.dtype)
    idx = 0

    # 1. Boundary containment constraints: ri <= xi <= 1-ri and ri <= yi <= 1-ri
    # These are expressed as:
    # xi - ri >= 0
    # 1 - xi - ri >= 0
    # yi - ri >= 0
    # 1 - yi - ri >= 0
    for i in range(n):
        constraints[idx] = x[i] - r[i]
        idx += 1
        constraints[idx] = 1 - x[i] - r[i]
        idx += 1
        constraints[idx] = y[i] - r[i]
        idx += 1
        constraints[idx] = 1 - y[i] - r[i]
        idx += 1

    # 2. Non-overlap constraints: sqrt((xi-xj)^2 + (yi-yj)^2) >= ri + rj
    # We use squared distances to avoid sqrt, which is computationally cheaper
    # (xi-xj)^2 + (yi-yj)^2 >= (ri + rj)^2
    # Expressed as: (xi-xj)^2 + (yi-yj)^2 - (ri + rj)^2 >= 0
    for i in range(n):
        for j in range(i + 1, n): # Check each pair only once
            dist_sq = (x[i] - x[j])**2 + (y[i] - y[j])**2
            min_dist_sq = (r[i] + r[j])**2
            constraints[idx] = dist_sq - min_dist_sq
            idx += 1
            
    return constraints

def _generate_initial_guess(n, seed=42):
    """
    Generates a structured initial guess based on a perturbed grid.
    This provides a much better starting point than random initialization.
    """
    np.random.seed(seed)
    
    # Use a 5x6 grid which can hold up to 30 circles.
    # For 26 circles, this is a good balance.
    rows, cols = 5, 6
    
    # Calculate spacing for centers to form a roughly uniform grid
    # Centers are placed such that there's space for radius on each side.
    x_coords = np.linspace(0.5/cols, 1 - 0.5/cols, cols)
    y_coords = np.linspace(0.5/rows, 1 - 0.5/rows, rows)
    
    xx, yy = np.meshgrid(x_coords, y_coords)
    
    # Flatten and take the first n positions
    x = xx.flatten()[:n]
    y = yy.flatten()[:n]
    
    # Add small random noise to break symmetry and allow for non-grid optima
    x += np.random.uniform(-0.015, 0.015, n) 
    y += np.random.uniform(-0.015, 0.015, n)
    
    # Initial radius: a safe, relatively large radius to start from
    # Max possible uniform radius for a 5x6 grid packing would be 1/(2*max(rows, cols)) = 1/12 = 0.08333...
    # Starting with this value directly, allowing SLSQP to resolve minor overlaps.
    initial_r_estimate = 1.0 / (2 * max(rows, cols)) 
    r = np.full(n, initial_r_estimate) # Start with the full estimated max uniform radius
    
    # Flatten into a single parameter vector
    params_flat = np.zeros(n * 3)
    params_flat[0::3] = x
    params_flat[1::3] = y
    params_flat[2::3] = r
    
    return params_flat

def circle_packing26() -> np.ndarray:
    """
    Places 26 non-overlapping circles in the unit square in order to maximize the sum of radii.
    Employs basinhopping with SLSQP as the local minimizer, starting from a high-quality initial guess.

    Returns:
        circles: np.array of shape (26,3), where the i-th row (x,y,r) stores the (x,y) coordinates of the i-th circle of radius r.
    """
    n = N_CIRCLES
    
    # --- 1. Define bounds for optimization variables ---
    # Each circle has (x, y, r) parameters.
    # x, y: [1e-6, 1-1e-6] (slightly tighter bounds to avoid numerical issues near edges)
    # r: [1e-6, 0.5] (1e-6 to avoid zero radius, 0.5 is max for a single circle)
    param_bounds = []
    for _ in range(n):
        param_bounds.append((1e-6, 1.0 - 1e-6))  # x coordinate
        param_bounds.append((1e-6, 1.0 - 1e-6))  # y coordinate
        param_bounds.append((1e-6, 0.5))         # radius

    # --- 2. Generate a high-quality initial guess ---
    x0 = _generate_initial_guess(n, seed=42)

    # --- 3. Define constraints for SLSQP (local minimizer for basinhopping) ---
    # SLSQP requires constraints in a list of dicts format.
    # The 'fun' for 'ineq' type must return an array of values that must be >= 0.
    slsqp_constraints = [{'type': 'ineq', 'fun': _constraint_values_numba}]

    # --- 4. Configure local minimizer arguments for basinhopping ---
    minimizer_kwargs = {
        "method": "SLSQP",
        "bounds": param_bounds,
        "constraints": slsqp_constraints,
        # Options for SLSQP within each basinhopping step
        "options": {'maxiter': 1500, 'ftol': 1e-9, 'disp': False} # Increased maxiter for thorough refinement
    }

    # --- 5. Global Optimization using Basinhopping ---
    # Basinhopping combines global stepping with local optimization.
    # It starts from x0 and repeatedly perturbs the solution, then runs a local minimizer.
    np.random.seed(42) # Set numpy seed for reproducibility
    
    print("Starting Basinhopping for global optimization with SLSQP local refinement...")
    bh_result = basinhopping(
        func=_objective,
        x0=x0,
        minimizer_kwargs=minimizer_kwargs,
        niter=300,       # Increased number of hopping iterations for more global exploration
        T=1.0,           # Temperature parameter for acceptance criterion (higher T allows more uphill steps)
        stepsize=0.05,   # Magnitude of random perturbations
        disp=True,       # Display optimization progress for basinhopping
        seed=42          # Seed for basinhopping's internal random number generator
    )
    print("Basinhopping finished.")
    print(f"Basinhopping Result - Objective (neg sum radii): {bh_result.fun}, Sum radii: {-bh_result.fun}")
    
    # --- 6. Format the final result ---
    final_params = bh_result.x
    x, y, r = _unpack_params_numba(final_params)
    
    circles = np.zeros((n, 3))
    circles[:, 0] = x
    circles[:, 1] = y
    circles[:, 2] = r

    # --- 6. Final validation ---
    # Check if any constraints are significantly violated in the final solution
    # A small negative tolerance is acceptable due to floating point precision
    final_constraint_violations = _constraint_values_numba(final_params)
    min_constraint_value = np.min(final_constraint_violations)
    if min_constraint_value < -1e-7: # Epsilon for floating point comparisons
        print(f"WARNING: Final solution violates constraints. Minimum constraint value: {min_constraint_value:.2e}")
        print("This may indicate an infeasible solution or numerical issues.")
    elif min_constraint_value < 0:
        print(f"NOTE: Minor constraint violations (up to {min_constraint_value:.2e}) observed, likely due to floating point precision.")

    return circles


# EVOLVE-BLOCK-END
