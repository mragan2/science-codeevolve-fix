# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import differential_evolution, minimize
from numba import njit # For performance critical parts

# Constants
N_CIRCLES = 32
EPS = 1e-7 # Small epsilon for radii and distances, to avoid numerical issues with zero or near-zero values

@njit(cache=True)
def _calculate_violations(positions_radii: np.ndarray) -> tuple[float, float, int]:
    """
    Calculates total boundary and overlap violations for a given configuration.
    positions_radii is a flattened array [x0, y0, r0, x1, y1, r1, ...]
    Returns: total_boundary_violation, total_overlap_violation, num_overlaps
    """
    n = len(positions_radii) // 3
    total_boundary_violation = 0.0
    total_overlap_violation = 0.0
    num_overlaps = 0

    # Ensure radii are positive, if not, treat as severe violation
    for i in range(n):
        r_i = positions_radii[i*3 + 2]
        if r_i < EPS:
            total_boundary_violation += (EPS - r_i) * 1000 # Heavy penalty for non-positive radii

    for i in range(n):
        x_i = positions_radii[i*3]
        y_i = positions_radii[i*3 + 1]
        r_i = positions_radii[i*3 + 2]

        # Boundary containment violations: r <= x <= 1-r and r <= y <= 1-r
        total_boundary_violation += max(0.0, r_i - x_i)
        total_boundary_violation += max(0.0, x_i + r_i - 1.0)
        total_boundary_violation += max(0.0, r_i - y_i)
        total_boundary_violation += max(0.0, y_i + r_i - 1.0)

        # Overlap violations: distance >= r_i + r_j
        for j in range(i + 1, n):
            x_j = positions_radii[j*3]
            y_j = positions_radii[j*3 + 1]
            r_j = positions_radii[j*3 + 2]

            dist_sq = (x_i - x_j)**2 + (y_i - y_j)**2
            min_dist_sq = (r_i + r_j)**2

            # Check for overlap, allowing a small tolerance (EPS) for floating point precision
            if dist_sq < min_dist_sq - EPS:
                overlap_dist = np.sqrt(dist_sq)
                violation = (r_i + r_j) - overlap_dist
                total_overlap_violation += violation
                num_overlaps += 1
    return total_boundary_violation, total_overlap_violation, num_overlaps

def _objective_de(positions_radii: np.ndarray, penalty_factor_boundary: float, penalty_factor_overlap: float) -> float:
    """
    Objective function for differential_evolution. Minimizes -sum(radii) + penalties for violations.
    """
    radii = positions_radii[2::3]
    sum_radii = np.sum(radii)

    boundary_violation, overlap_violation, _ = _calculate_violations(positions_radii)

    # Minimize negative sum of radii, heavily penalize any violations
    return -sum_radii + penalty_factor_boundary * boundary_violation + penalty_factor_overlap * overlap_violation

# Objective function for SLSQP. Purely minimizes -sum(radii).
# Constraints will handle feasibility.
def _objective_slsqp(positions_radii: np.ndarray) -> float:
    radii = positions_radii[2::3]
    # Add a small penalty if any radius is non-positive, though bounds should prevent this.
    if np.any(radii < EPS):
        return 1e10 # Very large value to push away from invalid radii
    return -np.sum(radii)

# Jacobian for the objective function for SLSQP
@njit(cache=True)
def _objective_slsqp_jac(positions_radii: np.ndarray) -> np.ndarray:
    n = len(positions_radii) // 3
    jacobian = np.zeros(n * 3, dtype=np.float64)
    # For each radius r_i, the derivative of -sum(radii) is -1
    for i in range(n):
        jacobian[i*3 + 2] = -1.0
    return jacobian

# --- New/Modified Constraint Functions and their Jacobians for SLSQP ---
@njit(cache=True)
def _slsqp_boundary_constraints(positions_radii: np.ndarray) -> np.ndarray:
    """
    Returns an array of boundary constraint values (val >= 0).
    For each circle i: x_i - r_i, 1 - x_i - r_i, y_i - r_i, 1 - y_i - r_i
    """
    n = len(positions_radii) // 3
    constraints = np.empty(n * 4, dtype=np.float64)
    for i in range(n):
        x_i = positions_radii[i*3]
        y_i = positions_radii[i*3 + 1]
        r_i = positions_radii[i*3 + 2]
        constraints[i*4] = x_i - r_i
        constraints[i*4 + 1] = 1.0 - x_i - r_i
        constraints[i*4 + 2] = y_i - r_i
        constraints[i*4 + 3] = 1.0 - y_i - r_i
    return constraints

@njit(cache=True)
def _slsqp_boundary_jacobian(positions_radii: np.ndarray) -> np.ndarray:
    """
    Returns the Jacobian matrix for the boundary constraints.
    Shape: (num_boundary_constraints, num_variables)
    """
    n = len(positions_radii) // 3
    num_vars = n * 3
    num_boundary_constraints = n * 4
    jacobian = np.zeros((num_boundary_constraints, num_vars), dtype=np.float64)

    for i in range(n):
        # Constraint: x_i - r_i >= 0
        jacobian[i*4, i*3] = 1.0      # d(x_i - r_i)/dx_i
        jacobian[i*4, i*3 + 2] = -1.0 # d(x_i - r_i)/dr_i

        # Constraint: 1.0 - x_i - r_i >= 0
        jacobian[i*4 + 1, i*3] = -1.0 # d(1 - x_i - r_i)/dx_i
        jacobian[i*4 + 1, i*3 + 2] = -1.0 # d(1 - x_i - r_i)/dr_i

        # Constraint: y_i - r_i >= 0
        jacobian[i*4 + 2, i*3 + 1] = 1.0 # d(y_i - r_i)/dy_i
        jacobian[i*4 + 2, i*3 + 2] = -1.0 # d(y_i - r_i)/dr_i

        # Constraint: 1.0 - y_i - r_i >= 0
        jacobian[i*4 + 3, i*3 + 1] = -1.0 # d(1 - y_i - r_i)/dy_i
        jacobian[i*4 + 3, i*3 + 2] = -1.0 # d(1 - y_i - r_i)/dr_i
    return jacobian

@njit(cache=True)
def _slsqp_overlap_constraints(positions_radii: np.ndarray) -> np.ndarray:
    """
    Returns an array of overlap constraint values (val >= 0) using squared distances.
    For each pair (i, j): (x_i - x_j)^2 + (y_i - y_j)^2 - (r_i + r_j)^2
    """
    n = len(positions_radii) // 3
    num_pairs = n * (n - 1) // 2
    constraints = np.empty(num_pairs, dtype=np.float64)
    k = 0
    for i in range(n):
        x_i = positions_radii[i*3]
        y_i = positions_radii[i*3 + 1]
        r_i = positions_radii[i*3 + 2]
        for j in range(i + 1, n):
            x_j = positions_radii[j*3]
            y_j = positions_radii[j*3 + 1]
            r_j = positions_radii[j*3 + 2]

            dist_sq = (x_i - x_j)**2 + (y_i - y_j)**2
            min_dist_sq = (r_i + r_j)**2
            constraints[k] = dist_sq - min_dist_sq
            k += 1
    return constraints

@njit(cache=True)
def _slsqp_overlap_jacobian(positions_radii: np.ndarray) -> np.ndarray:
    """
    Returns the Jacobian matrix for the overlap constraints.
    Shape: (num_overlap_constraints, num_variables)
    """
    n = len(positions_radii) // 3
    num_vars = n * 3
    num_overlap_constraints = n * (n - 1) // 2
    jacobian = np.zeros((num_overlap_constraints, num_vars), dtype=np.float64)
    
    k = 0 # Constraint index
    for i in range(n):
        x_i = positions_radii[i*3]
        y_i = positions_radii[i*3 + 1]
        r_i = positions_radii[i*3 + 2]
        for j in range(i + 1, n):
            x_j = positions_radii[j*3]
            y_j = positions_radii[j*3 + 1]
            r_j = positions_radii[j*3 + 2]

            # Derivatives for circle i
            jacobian[k, i*3] = 2 * (x_i - x_j)       # dC_ij / dx_i
            jacobian[k, i*3 + 1] = 2 * (y_i - y_j)   # dC_ij / dy_i
            jacobian[k, i*3 + 2] = -2 * (r_i + r_j)  # dC_ij / dr_i

            # Derivatives for circle j
            jacobian[k, j*3] = -2 * (x_i - x_j)      # dC_ij / dx_j
            jacobian[k, j*3 + 1] = -2 * (y_i - y_j)  # dC_ij / dy_j
            jacobian[k, j*3 + 2] = -2 * (r_i + r_j)  # dC_ij / dr_j
            k += 1
    return jacobian

def _get_slsqp_constraint_definitions(n_circles: int):
    """
    Generates a list of constraint dictionaries for scipy.optimize.minimize (SLSQP).
    Constraints are of type 'ineq', meaning fun(x) >= 0.
    Uses vectorized, Numba-compiled constraint functions and their Jacobians for efficiency.
    """
    constraints = []
    constraints.append({'type': 'ineq', 'fun': _slsqp_boundary_constraints, 'jac': _slsqp_boundary_jacobian})
    constraints.append({'type': 'ineq', 'fun': _slsqp_overlap_constraints, 'jac': _slsqp_overlap_jacobian})
    return constraints

# --- Initial Population Generation for DE ---
def _generate_initial_population(n_circles: int, bounds: list[tuple[float, float]], seed: int, de_popsize_multiplier: int, num_structured: int = 5):
    """
    Generates an initial population for Differential Evolution, including structured patterns.
    num_structured: number of structured initial configurations to generate.
    """
    # Use numpy.random.Generator for modern, reproducible randomness
    rng = np.random.default_rng(seed=seed) 

    # Total number of parameters
    num_params = n_circles * 3
    
    # Calculate actual popsize for DE
    actual_de_popsize = de_popsize_multiplier * num_params

    # Initialize with Latin Hypercube sampling for good initial diversity
    initial_pop = np.zeros((actual_de_popsize, num_params), dtype=np.float64)
    for i in range(num_params):
        low, high = bounds[i]
        initial_pop[:, i] = rng.uniform(low, high, actual_de_popsize)

    structured_configs = []

    # 1. 6x6 grid, remove 4 circles (for N_CIRCLES = 32)
    if n_circles == 32:
        grid_dim = 6
        r_grid = 1.0 / (2 * grid_dim)
        grid_centers = []
        for i in range(grid_dim):
            for j in range(grid_dim):
                x = r_grid + j * 2 * r_grid # Ensure x/y are consistent
                y = r_grid + i * 2 * r_grid
                grid_centers.append((x, y, r_grid))
        
        chosen_indices = rng.choice(len(grid_centers), n_circles, replace=False)
        grid_config = np.array([grid_centers[idx] for idx in chosen_indices]).flatten()
        structured_configs.append(grid_config)

    # 2. 4x8 grid, with slight perturbation and radius variation
    if n_circles == 32:
        grid_dim_x = 4
        grid_dim_y = 8
        r_base = 1.0 / (2 * grid_dim_y) # Use smallest dimension for base radius (0.0625)
        
        config_circles = []
        for i in range(grid_dim_y):
            for j in range(grid_dim_x):
                x = r_base + j * 2 * r_base
                y = r_base + i * 2 * r_base
                config_circles.append((x, y, r_base))
        
        config = np.array(config_circles).flatten()
        
        # Add a slight random perturbation to break perfect symmetry
        perturbation_scale = 0.05 # 5% of radius as max perturbation
        for k in range(n_circles):
            r_k = config[k*3+2]
            config[k*3] += rng.uniform(-r_k * perturbation_scale, r_k * perturbation_scale) # Perturb x
            config[k*3+1] += rng.uniform(-r_k * perturbation_scale, r_k * perturbation_scale) # Perturb y
            config[k*3+2] += rng.uniform(-r_k * perturbation_scale * 0.5, r_k * perturbation_scale * 0.5) # Perturb r slightly
        
        # Ensure it stays within bounds after perturbation
        for k in range(num_params):
            config[k] = np.clip(config[k], bounds[k][0], bounds[k][1])
        structured_configs.append(config)
    
    # 3. Hexagonal-like packing for 32 circles
    if n_circles == 32:
        num_rows_hex = 5
        row_counts_hex = [7, 6, 7, 6, 6] # Sums to 32
        
        # Calculate base radius assuming vertical spacing for hexagonal packing
        # Total height = r + (num_rows-1)*sqrt(3)*r + r = (2 + (num_rows-1)*sqrt(3))*r = 1.0
        r_base_vertical = 1.0 / (2 + (num_rows_hex - 1) * np.sqrt(3))
        
        config_hex_packing = []
        current_y_hex = r_base_vertical
        
        for i, count in enumerate(row_counts_hex):
            # Calculate radius needed to fit 'count' circles horizontally in 1.0 unit
            r_horiz = 1.0 / (2 * count)
            
            # Take the minimum of the two radii to ensure both horizontal and vertical fit
            r_row = min(r_base_vertical, r_horiz)
            
            x_offset_hex = r_row if i % 2 == 1 else 0.0 # Staggering for odd rows
            
            for j in range(count):
                x = r_row + j * 2 * r_row + x_offset_hex
                # Ensure x is within bounds [r_row, 1.0 - r_row]
                x = np.clip(x, r_row, 1.0 - r_row)
                config_hex_packing.append((x, current_y_hex, r_row))
            
            current_y_hex += r_row * np.sqrt(3) # Standard vertical step for hex packing
            
        # Add perturbation to break symmetry and allow optimization
        perturbation_scale_hex = 0.02 # Smaller perturbation for hex packing
        for k in range(n_circles):
            r_k = config_hex_packing[k][2]
            config_hex_packing[k] = (
                config_hex_packing[k][0] + rng.uniform(-r_k * perturbation_scale_hex, r_k * perturbation_scale_hex),
                config_hex_packing[k][1] + rng.uniform(-r_k * perturbation_scale_hex, r_k * perturbation_scale_hex),
                config_hex_packing[k][2] + rng.uniform(-r_k * perturbation_scale_hex * 0.5, r_k * perturbation_scale_hex * 0.5)
            )
        
        # Clip positions and radii to bounds after perturbation
        for k in range(n_circles):
            x_k, y_k, r_k = config_hex_packing[k]
            config_hex_packing[k] = (
                np.clip(x_k, bounds[k*3][0], bounds[k*3][1]),
                np.clip(y_k, bounds[k*3+1][0], bounds[k*3+1][1]),
                np.clip(r_k, bounds[k*3+2][0], bounds[k*3+2][1])
            )
        
        structured_configs.append(np.array(config_hex_packing).flatten())
        num_structured += 1 # Increment count of structured configs

    # Add more random configurations if num_structured is higher than available specific patterns
    while len(structured_configs) < num_structured:
        random_config = np.array([rng.uniform(b[0], b[1]) for b in bounds])
        structured_configs.append(random_config)

    # Place structured configs at the beginning of the initial population
    for i, config in enumerate(structured_configs):
        if i < actual_de_popsize: # Ensure we don't exceed actual_de_popsize
            initial_pop[i] = config
    
    return initial_pop

def circle_packing32() -> np.ndarray:
    """
    Places 32 non-overlapping circles in the unit square in order to maximize the sum of radii.
    Employs a two-phase optimization: Differential Evolution for global search,
    followed by SLSQP for local refinement.

    Returns:
        circles: np.array of shape (32,3), where the i-th row (x,y,r) stores the (x,y) coordinates of the i-th circle of radius r.
    """
    n = N_CIRCLES
    
    # --- Optimization Parameters ---
    seed = 42 # For reproducibility
    np.random.seed(seed) # For older numpy functions that might still be used
    
    # Bounds for each variable (x, y, r)
    # x, y are in [0, 1]. r is in [EPS, 0.5] (max possible radius for a single circle)
    bounds = [(0.0, 1.0), (0.0, 1.0), (EPS, 0.5)] * n
    
    # Differential Evolution parameters
    de_popsize = 15 # Population size multiplier (default is 15 * num_variables)
    de_maxiter = 800 # Increased iterations for DE for better global search
    de_workers = -1 # Use all available CPU cores
    de_penalty_boundary = 10000.0 # High penalty for boundary violations
    de_penalty_overlap = 10000.0  # High penalty for overlap violations
    de_num_structured_init = 5 # Number of structured initial configurations for DE (will be adjusted in function)

    # SLSQP parameters
    slsqp_maxiter = 500 # Max iterations for SLSQP

    start_total_time = time.time()

    # Phase 1: Global optimization with Differential Evolution
    print(f"Starting Differential Evolution for {n} circles (seed={seed})...")
    de_start_time = time.time()
    
    # Generate initial population including structured patterns
    # The _generate_initial_population function will internally adjust num_structured
    # if it adds new patterns.
    initial_pop_de = _generate_initial_population(n, bounds, seed, de_popsize, de_num_structured_init)

    de_result = differential_evolution(
        func=_objective_de,
        bounds=bounds,
        args=(de_penalty_boundary, de_penalty_overlap),
        strategy='best2bin', # Good balance of exploration and exploitation
        maxiter=de_maxiter,
        popsize=de_popsize,
        mutation=(0.5, 1.0), # Mutation range
        recombination=0.7,   # Crossover probability
        seed=seed,
        workers=de_workers,
        disp=True,           # Display progress
        polish=False,         # Don't polish yet, SLSQP will do final refinement
        init=initial_pop_de # Provide custom initial population
    )
    de_end_time = time.time()
    print(f"Differential Evolution finished in {de_end_time - de_start_time:.2f} seconds.")
    print(f"DE result objective (sum_radii with penalties): {de_result.fun}")
    
    best_de_x = de_result.x
    
    # Evaluate the best DE result
    boundary_v, overlap_v, num_o = _calculate_violations(best_de_x)
    current_sum_radii = np.sum(best_de_x[2::3])
    print(f"DE best solution sum_radii: {current_sum_radii:.6f}")
    print(f"DE violations: boundary={boundary_v:.6f}, overlap={overlap_v:.6f}, num_overlaps={num_o}")

    # Phase 2: Local refinement with SLSQP
    print(f"\nStarting SLSQP for local refinement from DE result...")
    slsqp_start_time = time.time()
    slsqp_constraints = _get_slsqp_constraint_definitions(n) # Use new function name
    
    slsqp_result = minimize(
        fun=_objective_slsqp,
        x0=best_de_x, # Start from the best solution found by DE
        method='SLSQP',
        jac=_objective_slsqp_jac, # Provide analytical Jacobian for objective
        bounds=bounds, # Simple variable bounds, actual containment is via constraints
        constraints=slsqp_constraints,
        options={'disp': True, 'maxiter': slsqp_maxiter, 'ftol': 1e-8} # Tighten ftol for more precision
    )
    slsqp_end_time = time.time()
    print(f"SLSQP finished in {slsqp_end_time - slsqp_start_time:.2f} seconds.")
    print(f"SLSQP result objective (-sum_radii): {slsqp_result.fun}")

    final_circles_flat = slsqp_result.x
    
    # Final validation and potential fallback
    boundary_v_final, overlap_v_final, num_o_final = _calculate_violations(final_circles_flat)
    final_sum_radii = np.sum(final_circles_flat[2::3])

    print(f"\n--- Final Solution ---")
    print(f"Total optimization time: {time.time() - start_total_time:.2f} seconds.")
    print(f"Final sum_radii: {final_sum_radii:.6f}")
    print(f"Final violations: boundary={boundary_v_final:.8f}, overlap={overlap_v_final:.8f}, num_overlaps={num_o_final}")
    print(f"SLSQP success: {slsqp_result.success}, Message: {slsqp_result.message}")

    # Check for feasibility after SLSQP. If not fully feasible, it implies SLSQP struggled.
    # We might need to fall back to the DE result if it was better in terms of feasibility,
    # or just accept the SLSQP result knowing it's the best it could do under constraints.
    # For this problem, strict feasibility is required, so small violations are not acceptable.
    if not slsqp_result.success or boundary_v_final > EPS or overlap_v_final > EPS:
        print("WARNING: SLSQP did not converge to a fully feasible solution or has residual violations.")
        print("Comparing with the best DE result for feasibility.")
        
        # Recalculate DE result's feasibility metrics precisely
        de_boundary_v, de_overlap_v, de_num_o = _calculate_violations(best_de_x)
        
        # Fallback logic: If SLSQP is infeasible (or failed), but DE was feasible, use DE's result.
        # If both are infeasible, stick with SLSQP's result as it's the result of a constrained optimizer.
        if (boundary_v_final > EPS or overlap_v_final > EPS) and \
           (de_boundary_v <= EPS and de_overlap_v <= EPS): # If DE was feasible and SLSQP is not
            print("Falling back to DE result due to SLSQP infeasibility.")
            final_circles_flat = best_de_x
            final_sum_radii = np.sum(final_circles_flat[2::3])
            print(f"Fallback DE sum_radii: {final_sum_radii:.6f}")
            boundary_v_final, overlap_v_final, num_o_final = _calculate_violations(final_circles_flat)
            print(f"Fallback DE violations: boundary={boundary_v_final:.8f}, overlap={overlap_v_final:.8f}, num_overlaps={num_o_final}")
        elif (boundary_v_final > EPS or overlap_v_final > EPS) and \
             (de_boundary_v > EPS or de_overlap_v > EPS): # Both are infeasible
             print("Both DE and SLSQP results show violations. Returning SLSQP result anyway as it's a constrained optimization attempt.")
        # Else, SLSQP was successful and feasible, so use it (this is the desired path).

    # Reshape the flattened array into (N_CIRCLES, 3)
    circles = final_circles_flat.reshape(n, 3)

    return circles


# EVOLVE-BLOCK-END
