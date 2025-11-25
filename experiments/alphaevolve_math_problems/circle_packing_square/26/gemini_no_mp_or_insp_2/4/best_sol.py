# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import differential_evolution, minimize
from numba import njit, float64, int64
import random

# Set a fixed random seed for reproducibility
np.random.seed(42)
random.seed(42) # For any other random calls if introduced, though DE has its own seed

# Global constants for the problem
N_CIRCLES = 26
MIN_RADIUS = 1e-6 # Minimum allowed radius to prevent degenerate circles
MAX_RADIUS_SINGLE = 0.2 # Max radius for any single circle in a dense packing of 26 circles (adjusted from 0.5)
PENALTY_FACTOR = 1e4 # Large penalty factor for constraint violations in DE's objective

# Numba-jitted utility functions for performance
# These functions access parameters from the 1D array 'params'
@njit(float64(float64[:], int64), cache=True)
def _get_x(params, i):
    return params[i * 3]

@njit(float64(float64[:], int64), cache=True)
def _get_y(params, i):
    return params[i * 3 + 1]

@njit(float64(float64[:], int64), cache=True)
def _get_r(params, i):
    return params[i * 3 + 2]

@njit(float64[:](float64[:], int64), cache=True)
def _get_radii_from_params(params, n_circles):
    """Extracts radii from the 1D parameter array."""
    radii = np.empty(n_circles)
    for i in range(n_circles):
        radii[i] = _get_r(params, i)
    return radii

@njit(float64(float64[:], int64), cache=True)
def _calculate_total_penalty(params, n_circles):
    """
    Calculates total penalty for boundary and overlap violations.
    Used by differential_evolution to guide the search.
    """
    penalty = 0.0

    # Boundary violations
    for i in range(n_circles):
        x_i = _get_x(params, i)
        y_i = _get_y(params, i)
        r_i = _get_r(params, i)

        # Penalize radii smaller than MIN_RADIUS (though bounds should prevent this mostly)
        if r_i < MIN_RADIUS:
            penalty += (MIN_RADIUS - r_i) * PENALTY_FACTOR

        # Check containment within [0,1] square
        if x_i - r_i < 0:
            penalty += abs(x_i - r_i) * PENALTY_FACTOR
        if x_i + r_i > 1:
            penalty += abs(x_i + r_i - 1) * PENALTY_FACTOR
        if y_i - r_i < 0:
            penalty += abs(y_i - r_i) * PENALTY_FACTOR
        if y_i + r_i > 1:
            penalty += abs(y_i + r_i - 1) * PENALTY_FACTOR

    # Overlap violations
    for i in range(n_circles):
        for j in range(i + 1, n_circles): # Check each unique pair
            x_i, y_i, r_i = _get_x(params, i), _get_y(params, i), _get_r(params, i)
            x_j, y_j, r_j = _get_x(params, j), _get_y(params, j), _get_r(params, j)

            dist_sq = (x_i - x_j)**2 + (y_i - y_j)**2
            min_dist_sq = (r_i + r_j)**2

            if dist_sq < min_dist_sq:
                # Penalty proportional to the squared overlap distance
                penalty += (min_dist_sq - dist_sq) * PENALTY_FACTOR
                
    return penalty

# Objective function for differential_evolution
def _objective_de(params, n_circles):
    """
    Objective function for differential_evolution: minimize -sum(radii) + penalties.
    This guides the global search towards feasible solutions with larger radii.
    """
    radii = _get_radii_from_params(params, n_circles)
    sum_radii = np.sum(radii)
    
    penalty = _calculate_total_penalty(params, n_circles)
    
    return -sum_radii + penalty

# Objective function for local minimization (SLSQP) - only sum of radii
def _objective_slsqp(params, n_circles):
    """
    Objective function for local minimization: minimize -sum(radii).
    Constraints are handled separately by SLSQP's constraint mechanism.
    """
    radii = _get_radii_from_params(params, n_circles)
    return -np.sum(radii)

# Constraint functions for local minimization (SLSQP)
@njit(float64[:](float64[:], int64), cache=True)
def _boundary_constraints_slsqp(params, n_circles):
    """
    Returns an array of values for boundary inequality constraints (must be >= 0).
    For each circle (x_i, y_i, r_i):
    x_i - r_i >= 0
    1 - r_i - x_i >= 0
    y_i - r_i >= 0
    1 - r_i - y_i >= 0
    r_i - MIN_RADIUS >= 0
    
    This version is numba-jitted for performance by pre-allocating the array.
    """
    num_constraints = n_circles * 5 # 4 boundary + 1 min_radius per circle
    constraints = np.empty(num_constraints, dtype=np.float64)
    idx = 0
    for i in range(n_circles):
        x_i = _get_x(params, i)
        y_i = _get_y(params, i)
        r_i = _get_r(params, i)

        constraints[idx] = x_i - r_i
        idx += 1
        constraints[idx] = 1.0 - r_i - x_i
        idx += 1
        constraints[idx] = y_i - r_i
        idx += 1
        constraints[idx] = 1.0 - r_i - y_i
        idx += 1
        constraints[idx] = r_i - MIN_RADIUS
        idx += 1
    return constraints

@njit(float64[:](float64[:], int64), cache=True)
def _overlap_constraints_slsqp(params, n_circles):
    """
    Returns an array of values for non-overlap inequality constraints (must be >= 0).
    For each pair of circles (i, j):
    (x_i - x_j)^2 + (y_i - y_j)^2 - (r_i + r_j)^2 >= 0

    This version is numba-jitted for performance by pre-allocating the array.
    """
    num_constraints = n_circles * (n_circles - 1) // 2
    constraints = np.empty(num_constraints, dtype=np.float64)
    idx = 0
    for i in range(n_circles):
        for j in range(i + 1, n_circles): # Check each unique pair
            x_i = _get_x(params, i)
            y_i = _get_y(params, i)
            r_i = _get_r(params, i)
            x_j = _get_x(params, j)
            y_j = _get_y(params, j)
            r_j = _get_r(params, j)

            dist_sq = (x_i - x_j)**2 + (y_i - y_j)**2
            min_dist_sq = (r_i + r_j)**2
            constraints[idx] = dist_sq - min_dist_sq
            idx += 1
    return constraints

def circle_packing26() -> np.ndarray:
    """
    Places 26 non-overlapping circles in the unit square in order to maximize the sum of radii.

    Returns:
        circles: np.array of shape (26,3), where the i-th row (x,y,r) stores the (x,y) coordinates of the i-th circle of radius r.
    """
    n = N_CIRCLES
    
    # Define bounds for each (x, y, r) variable
    # x, y bounds are [0, 1]
    # r bounds are [MIN_RADIUS, MAX_RADIUS_SINGLE]
    bounds = [(0.0, 1.0), (0.0, 1.0), (MIN_RADIUS, MAX_RADIUS_SINGLE)] * n
    
    # Helper function to generate a grid-like initial population for Differential Evolution
    def _generate_grid_initial_population(n_circles, bounds, pop_size_de):
        # For 26 circles, a 5x6 grid offers 30 slots. We'll use 26 of them.
        num_cols = 5
        num_rows = 6
        
        # Calculate a reasonable initial radius based on grid dimensions
        r_val_base = 1.0 / (2.0 * max(num_cols, num_rows)) # Ensures they fit if touching
        
        min_r_bound = bounds[2][0] 
        max_r_bound = bounds[2][1]
        r_val_base = np.clip(r_val_base, min_r_bound, max_r_bound)

        initial_population_matrix = np.zeros((pop_size_de, n_circles * 3))
        for k in range(pop_size_de):
            temp_pop_list = []
            circle_count = 0
            for i in range(num_rows):
                for j in range(num_cols):
                    if circle_count < n_circles:
                        # Calculate center positions for a grid
                        x = r_val_base + j * (1.0 - 2 * r_val_base) / (num_cols - 1) if num_cols > 1 else 0.5
                        y = r_val_base + i * (1.0 - 2 * r_val_base) / (num_rows - 1) if num_rows > 1 else 0.5
                        
                        # Add small random perturbation for diversity in initial population
                        # Ensure perturbations keep values within overall bounds
                        x_p = np.clip(x + np.random.uniform(-0.01, 0.01), bounds[0][0], bounds[0][1])
                        y_p = np.clip(y + np.random.uniform(-0.01, 0.01), bounds[1][0], bounds[1][1])
                        r_p = np.clip(r_val_base + np.random.uniform(-0.001, 0.001), min_r_bound, max_r_bound)
                        
                        temp_pop_list.extend([x_p, y_p, r_p])
                        circle_count += 1
            initial_population_matrix[k, :] = np.array(temp_pop_list)
        return initial_population_matrix

    # Define DE parameters
    de_maxiter = 2000 # Reduced iterations
    de_popsize = 30   # Reduced population size
    
    # Generate initial population using the grid strategy
    initial_population_de = _generate_grid_initial_population(n, bounds, de_popsize)

    # Step 1: Global optimization using Differential Evolution
    # This step explores the search space broadly to find a good initial guess
    # for the local optimizer, less sensitive to local minima.
    de_result = differential_evolution(
        func=_objective_de,
        bounds=bounds,
        args=(n,),
        strategy='best1bin', # A common and effective strategy
        maxiter=de_maxiter,  # Use tuned iterations
        popsize=de_popsize,  # Use tuned population size
        tol=0.001,           # Tolerance for convergence
        mutation=(0.5, 1.0), # Mutation range
        recombination=0.7,   # Crossover probability
        seed=42,             # For reproducibility
        polish=False,        # We'll do local polish with SLSQP
        workers=-1,          # Use all available CPU cores for parallel evaluation
        init=initial_population_de # Provide an informed initial population
    )
    
    initial_guess = de_result.x
    
    if not de_result.success:
        print(f"Warning: Differential Evolution did not converge successfully: {de_result.message}")
        # Even if DE didn't converge, its best candidate is still a good starting point for SLSQP.

    # Step 2: Local optimization using SLSQP (Sequential Least Squares Programming)
    # SLSQP is suitable for problems with non-linear constraints and provides a precise solution.
    # It takes the result from differential_evolution as an initial guess.
    
    # Define constraints for SLSQP
    # All constraints are of type 'ineq' (inequality), meaning fun(x) >= 0
    slsqp_constraints = [
        {'type': 'ineq', 'fun': _boundary_constraints_slsqp, 'args': (n,)}, # Now Numba-jitted
        {'type': 'ineq', 'fun': _overlap_constraints_slsqp, 'args': (n,)}   # Now Numba-jitted
    ]
    
    slsqp_result = minimize(
        fun=_objective_slsqp,
        x0=initial_guess,
        args=(n,),
        method='SLSQP',
        bounds=bounds,
        constraints=slsqp_constraints,
        options={'maxiter': 2500, 'ftol': 1e-9, 'disp': False} # Increased maxiter, stricter ftol for precision for local search
    )
    
    if not slsqp_result.success:
        print(f"Warning: SLSQP optimization did not converge successfully: {slsqp_result.message}")
        # It's possible for SLSQP to fail if the problem is highly degenerate or
        # if DE didn't provide a good enough starting point.

    # Reshape the optimized parameters into the desired (n, 3) format (x, y, r)
    optimized_params = slsqp_result.x
    circles = optimized_params.reshape((n, 3))

    # Final validation and radius adjustment (optional, but good practice)
    # Ensure radii are not below MIN_RADIUS due to floating point inaccuracies
    circles[:, 2] = np.maximum(circles[:, 2], MIN_RADIUS)

    return circles


# EVOLVE-BLOCK-END
