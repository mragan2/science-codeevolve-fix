# EVOLVE-BLOCK-START
from scipy.optimize import differential_evolution, minimize, NonlinearConstraint
from numba import njit
import time
import numpy as np

# Constants for the problem
N_CIRCLES = 26
PARAM_PER_CIRCLE = 3  # x, y, r
TOTAL_PARAMS = N_CIRCLES * PARAM_PER_CIRCLE

# Set a random seed for reproducibility
np.random.seed(42)

# Numba-optimized helper functions for constraint calculations
@njit(cache=True)
def _calculate_overlap_violations(params: np.ndarray) -> np.ndarray:
    """
    Calculates squared distance violations for non-overlap constraints.
    Returns: array where each element is (dist_sq - (r_i+r_j)^2).
             For a valid packing, all elements must be >= 0.
    """
    violations = np.zeros(N_CIRCLES * (N_CIRCLES - 1) // 2, dtype=params.dtype)
    k = 0
    for i in range(N_CIRCLES):
        for j in range(i + 1, N_CIRCLES):
            xi, yi, ri = params[i*PARAM_PER_CIRCLE : i*PARAM_PER_CIRCLE + PARAM_PER_CIRCLE]
            xj, yj, rj = params[j*PARAM_PER_CIRCLE : j*PARAM_PER_CIRCLE + PARAM_PER_CIRCLE]

            dist_sq = (xi - xj)**2 + (yi - yj)**2
            min_dist_sq = (ri + rj)**2
            
            violations[k] = dist_sq - min_dist_sq
            k += 1
    return violations

@njit(cache=True)
def _calculate_boundary_violations(params: np.ndarray) -> np.ndarray:
    """
    Calculates boundary containment violations.
    Returns: array where each element is (coord - radius) or (1 - coord - radius).
             For a valid packing, all elements must be >= 0.
    """
    violations = np.zeros(N_CIRCLES * 4, dtype=params.dtype) # 4 constraints per circle
    for i in range(N_CIRCLES):
        xi, yi, ri = params[i*PARAM_PER_CIRCLE : i*PARAM_PER_CIRCLE + PARAM_PER_CIRCLE]
        violations[i*4 + 0] = xi - ri        # x_min >= 0
        violations[i*4 + 1] = 1.0 - xi - ri  # x_max <= 1
        violations[i*4 + 2] = yi - ri        # y_min >= 0
        violations[i*4 + 3] = 1.0 - yi - ri  # y_max <= 1
    return violations

# Jacobians for SLSQP
@njit(cache=True)
def _objective_jac(params: np.ndarray) -> np.ndarray:
    """
    Jacobian of the objective function (sum of radii).
    d(-sum(r_i))/d(param_k)
    """
    jac = np.zeros(TOTAL_PARAMS, dtype=params.dtype)
    # Derivative w.r.t. each radius is -1, others are 0
    jac[2::PARAM_PER_CIRCLE] = -1.0 
    return jac

@njit(cache=True)
def _combined_constraints_jac(params: np.ndarray) -> np.ndarray:
    """
    Jacobian of the combined constraint function.
    Returns a matrix (total_constraints, TOTAL_PARAMS).
    """
    n_boundary_constraints = N_CIRCLES * 4
    n_overlap_constraints = N_CIRCLES * (N_CIRCLES - 1) // 2
    total_constraints = n_boundary_constraints + n_overlap_constraints
    
    jac = np.zeros((total_constraints, TOTAL_PARAMS), dtype=params.dtype)

    # Boundary constraints Jacobian
    for i in range(N_CIRCLES):
        # x_i, y_i, r_i indices
        idx_x = i * PARAM_PER_CIRCLE
        idx_y = i * PARAM_PER_CIRCLE + 1
        idx_r = i * PARAM_PER_CIRCLE + 2

        # Constraint 4*i: xi - ri >= 0
        jac[i*4 + 0, idx_x] = 1.0
        jac[i*4 + 0, idx_r] = -1.0

        # Constraint 4*i + 1: 1 - xi - ri >= 0
        jac[i*4 + 1, idx_x] = -1.0
        jac[i*4 + 1, idx_r] = -1.0

        # Constraint 4*i + 2: yi - ri >= 0
        jac[i*4 + 2, idx_y] = 1.0
        jac[i*4 + 2, idx_r] = -1.0

        # Constraint 4*i + 3: 1 - yi - ri >= 0
        jac[i*4 + 3, idx_y] = -1.0
        jac[i*4 + 3, idx_r] = -1.0

    # Overlap constraints Jacobian
    k_offset = n_boundary_constraints
    k = 0
    for i in range(N_CIRCLES):
        for j in range(i + 1, N_CIRCLES):
            xi, yi, ri = params[i*PARAM_PER_CIRCLE : i*PARAM_PER_CIRCLE + PARAM_PER_CIRCLE]
            xj, yj, rj = params[j*PARAM_PER_CIRCLE : j*PARAM_PER_CIRCLE + PARAM_PER_CIRCLE]

            # Current constraint index in the Jacobian matrix
            current_constraint_idx = k_offset + k

            # Indices of parameters for circle i and j
            idx_xi = i * PARAM_PER_CIRCLE
            idx_yi = i * PARAM_PER_CIRCLE + 1
            idx_ri = i * PARAM_PER_CIRCLE + 2
            idx_xj = j * PARAM_PER_CIRCLE
            idx_yj = j * PARAM_PER_CIRCLE + 1
            idx_rj = j * PARAM_PER_CIRCLE + 2

            # Derivatives w.r.t. xi, xj, yi, yj
            jac[current_constraint_idx, idx_xi] = 2 * (xi - xj)
            jac[current_constraint_idx, idx_xj] = 2 * (xj - xi)
            jac[current_constraint_idx, idx_yi] = 2 * (yi - yj)
            jac[current_constraint_idx, idx_yj] = 2 * (yj - yi)

            # Derivatives w.r.t. ri, rj
            jac[current_constraint_idx, idx_ri] = -2 * (ri + rj)
            jac[current_constraint_idx, idx_rj] = -2 * (ri + rj)
            
            k += 1
    return jac

# Objective function for the actual optimization (minimize -sum(radii))
def _objective(params: np.ndarray) -> float:
    radii = params[2::PARAM_PER_CIRCLE]
    return -np.sum(radii)

# Combined constraint function for NonlinearConstraint (returns array of g(x) values)
def _combined_constraints_func(params: np.ndarray) -> np.ndarray:
    """
    Combines all boundary and overlap constraints into a single array.
    All elements in the returned array must be >= 0 for a valid solution.
    """
    boundary_violations = _calculate_boundary_violations(params)
    overlap_violations = _calculate_overlap_violations(params)
    return np.concatenate((boundary_violations, overlap_violations))

# Penalized objective function for differential_evolution
# This adds a large penalty if constraints are violated, guiding DE towards feasible regions.
def _penalized_objective_de(params: np.ndarray) -> float:
    obj = _objective(params) # Negative sum of radii

    # Calculate penalties for constraint violations
    boundary_violations = _calculate_boundary_violations(params)
    overlap_violations = _calculate_overlap_violations(params)

    # Sum of squared negative violations (only penalize if violation < 0)
    # Reduced penalty weight to allow DE to explore more freely.
    penalty_weight = 1e6 # Changed from 1e7 to 1e6 for smoother penalty landscape
    
    boundary_penalty = np.sum(np.maximum(0, -boundary_violations)**2) * penalty_weight
    overlap_penalty = np.sum(np.maximum(0, -overlap_violations)**2) * penalty_weight

    return obj + boundary_penalty + overlap_penalty

def _generate_initial_population_hexagonal(pop_size: int, bounds: list) -> np.ndarray:
    """
    Generates an initial population based on a hexagonal-like packing heuristic.
    Adds random noise to create diversity for differential_evolution.
    """
    initial_population = np.zeros((pop_size, TOTAL_PARAMS))
    
    r_base = 0.09 # A reasonable starting radius for 26 circles to form a dense pattern
    
    dx = 2 * r_base
    dy = np.sqrt(3) * r_base

    base_circles_coords = []
    
    # Iterate through rows to place circles in a hexagonal pattern
    row_idx = 0
    while True:
        y = r_base + row_idx * dy
        if y > 1.0 - r_base + 1e-9: # Add tolerance for float comparison
            break

        is_even_row = (row_idx % 2 == 0)
        x_offset = 0 if is_even_row else dx / 2.0
        
        # Iterate through columns
        col_idx = 0
        while True:
            x = r_base + x_offset + col_idx * dx
            if x > 1.0 - r_base + 1e-9: # Add tolerance for float comparison
                break
            
            base_circles_coords.append([x, y, r_base])
            col_idx += 1
        
        row_idx += 1

    # Convert to numpy array
    base_circles_coords_np = np.array(base_circles_coords)
    
    # Trim or pad to exactly N_CIRCLES
    if base_circles_coords_np.shape[0] > N_CIRCLES:
        # If more circles generated by the heuristic, take the first N_CIRCLES
        base_circles_coords_np = base_circles_coords_np[:N_CIRCLES, :]
    elif base_circles_coords_np.shape[0] < N_CIRCLES:
        # If fewer, fill remaining with random circles within bounds
        num_missing = N_CIRCLES - base_circles_coords_np.shape[0]
        random_filler = np.zeros((num_missing, PARAM_PER_CIRCLE))
        for i in range(num_missing):
            random_filler[i, 0] = np.random.uniform(bounds[0][0], bounds[0][1])
            random_filler[i, 1] = np.random.uniform(bounds[1][0], bounds[1][1])
            random_filler[i, 2] = np.random.uniform(bounds[2][0], bounds[2][1])
        base_circles_coords_np = np.vstack((base_circles_coords_np, random_filler))

    base_circles_flat = base_circles_coords_np.flatten()

    # Generate population by perturbing the base configuration
    for i in range(pop_size):
        individual = np.copy(base_circles_flat)
        
        # Add random noise to x, y, and r
        individual[0::PARAM_PER_CIRCLE] += np.random.uniform(-0.02, 0.02, N_CIRCLES) # x noise
        individual[1::PARAM_PER_CIRCLE] += np.random.uniform(-0.02, 0.02, N_CIRCLES) # y noise
        individual[2::PARAM_PER_CIRCLE] += np.random.uniform(-0.005, 0.005, N_CIRCLES) # r noise
        
        # Clip values to stay within overall bounds
        for j in range(TOTAL_PARAMS):
            individual[j] = np.clip(individual[j], bounds[j][0], bounds[j][1])
            
        initial_population[i] = individual
        
    return initial_population

# Helper function for running SLSQP with consistent options
def _run_slsqp(x0: np.ndarray, bounds: list, nl_constraint: NonlinearConstraint) -> tuple[np.ndarray, float]:
    """
    Helper to run SLSQP and return the solution and objective value.
    Ensures the initial guess x0 is clipped to bounds before optimization.
    """
    clipped_x0 = np.copy(x0)
    for j in range(TOTAL_PARAMS):
        clipped_x0[j] = np.clip(clipped_x0[j], bounds[j][0], bounds[j][1])
    
    min_result = minimize(
        fun=_objective,
        x0=clipped_x0,
        method='SLSQP',
        bounds=bounds,
        jac=_objective_jac,
        constraints=[nl_constraint],
        options={'maxiter': 2000, 'ftol': 1e-10, 'disp': False} # Increased maxiter, tighter ftol for thorough local search
    )
    return min_result.x, -min_result.fun # Return actual sum of radii


def circle_packing26() -> np.ndarray:
    """
    Places 26 non-overlapping circles in the unit square in order to maximize the sum of radii.

    Returns:
        circles: np.array of shape (26,3), where the i-th row (x,y,r) stores the (x,y) coordinates of the i-th circle of radius r.
    """
    
    # 1. Define bounds for x, y, r for each circle
    # x: [0, 1], y: [0, 1], r: [1e-4, 0.5]
    bounds = [(0.0, 1.0), (0.0, 1.0), (1e-4, 0.5)] * N_CIRCLES

    # 2. Generate initial population using a hexagonal heuristic
    pop_size_de = 100 
    initial_pop_de = _generate_initial_population_hexagonal(pop_size_de, bounds)

    # 3. Global optimization with Differential Evolution
    de_result = differential_evolution(
        func=_penalized_objective_de,
        bounds=bounds,
        init=initial_pop_de, # Use the generated initial population
        strategy='best2bin',
        maxiter=500, # Max iterations for DE
        popsize=pop_size_de,
        tol=0.005,   # Tolerance for DE convergence
        mutation=(0.5, 1.0),
        recombination=0.7,
        seed=42,
        disp=False,
        workers=-1
    )
    
    # 4. Local optimization with SLSQP using the best result from DE, with multi-start refinement.
    
    # Define the NonlinearConstraint for SLSQP
    n_boundary_constraints = N_CIRCLES * 4
    n_overlap_constraints = N_CIRCLES * (N_CIRCLES - 1) // 2
    total_constraints = n_boundary_constraints + n_overlap_constraints
    
    lower_bounds_constraints = np.zeros(total_constraints)
    upper_bounds_constraints = np.full(total_constraints, np.inf)
    
    nl_constraint = NonlinearConstraint(
        _combined_constraints_func,
        lower_bounds_constraints,
        upper_bounds_constraints,
        jac=_combined_constraints_jac, # Use analytical Jacobian
        hess=None
    )

    # Initialize best solution with the result from Differential Evolution
    best_x = np.copy(de_result.x)
    best_sum_radii = -_objective(best_x) # Objective is -sum(radii), so negate for actual sum

    # Multi-start local optimization
    num_slsqp_restarts = 5 # Number of restarts, including the initial one
    perturbation_scale_xy = 5e-4 # Small perturbation for x, y coordinates
    perturbation_scale_r = 1e-5  # Even smaller perturbation for radii

    for i in range(num_slsqp_restarts):
        # Start subsequent restarts from the current best solution, perturbed
        current_x0 = np.copy(best_x) 

        if i > 0: # Add perturbation for restarts after the first one
            noise = np.random.normal(0, 1, TOTAL_PARAMS)
            for j in range(N_CIRCLES):
                current_x0[j*PARAM_PER_CIRCLE] += noise[j*PARAM_PER_CIRCLE] * perturbation_scale_xy # x
                current_x0[j*PARAM_PER_CIRCLE+1] += noise[j*PARAM_PER_CIRCLE+1] * perturbation_scale_xy # y
                current_x0[j*PARAM_PER_CIRCLE+2] += noise[j*PARAM_PER_CIRCLE+2] * perturbation_scale_r # r
        
        # Run SLSQP from the (potentially perturbed) starting point
        current_x, current_sum_radii = _run_slsqp(current_x0, bounds, nl_constraint)

        # Update best solution if current run yielded a better sum of radii
        if current_sum_radii > best_sum_radii:
            best_sum_radii = current_sum_radii
            best_x = np.copy(current_x)
            # Optional: print(f"Restart {i+1}: New best sum_radii: {best_sum_radii:.6f}")

    # Reshape the best optimized parameters into the desired (N, 3) format
    optimized_circles = best_x.reshape((N_CIRCLES, PARAM_PER_CIRCLE))

    # Final validation: Ensure all radii are positive.
    optimized_circles[:, 2] = np.maximum(optimized_circles[:, 2], 1e-5) # Ensure strictly positive radii

    # One final check: if the solution is not fully feasible, log a warning.
    final_violations = _combined_constraints_func(best_x) # Check the best_x found across all restarts
    if np.any(final_violations < -1e-6): # Allow for small numerical tolerance
        print(f"Warning: Final solution from SLSQP is not fully feasible. Max violation: {np.min(final_violations):.2e}")

    return optimized_circles


# EVOLVE-BLOCK-END
