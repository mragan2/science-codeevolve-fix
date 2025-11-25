# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize, basinhopping, differential_evolution, NonlinearConstraint
from numba import njit # JIT compiler for performance critical sections

# --- Constants and Global Settings ---
N_CIRCLES = 32
EPSILON = 1e-7 # Smallest allowed radius, for numerical stability

# --- Objective Function and its Jacobian ---
def _objective(params):
    """Calculates the negative sum of radii to be minimized."""
    radii = params[2::3]
    return -np.sum(radii)

def _objective_jacobian(params):
    """Calculates the Jacobian of the objective function."""
    grad = np.zeros_like(params)
    grad[2::3] = -1.0 # Gradient with respect to each radius is -1
    return grad

# --- High-Quality Initial Guess Generators (from Inspirations 1 & 3) ---
def _generate_hexagonal_initial_guess(n_circles: int) -> np.ndarray:
    """
    Generates an initial configuration of circles arranged in an approximate
    hexagonal packing pattern within the unit square, specifically tailored for 32 circles.
    This provides a strong starting point for the optimizer.
    """
    circles_list = []
    
    # Define row structure for 32 circles to achieve a dense hexagonal packing
    # This pattern sums to 32 circles: 6 + 5 + 6 + 5 + 6 + 4 = 32
    row_counts = [6, 5, 6, 5, 6, 4]
    
    # Fallback if n_circles is not 32 (though fixed for this problem)
    if n_circles != sum(row_counts):
        return np.array([[0.5, 0.5, EPSILON]] * n_circles).flatten()

    num_rows = len(row_counts)
    max_cols = max(row_counts)

    # Calculate an initial radius (r_init) that allows the pattern to fit
    r_h_limit = 1.0 / (2.0 * max_cols)
    r_v_limit = 1.0 / (2.0 + (num_rows - 1) * np.sqrt(3))
    r_init = min(r_h_limit, r_v_limit) * 0.98 # Scale down slightly for initial space
    
    if r_init < EPSILON:
        r_init = 0.05 # Fallback to a small but significant radius

    # Calculate global offsets to center the pattern
    pattern_actual_width = max_cols * 2 * r_init
    pattern_actual_height = 2 * r_init + (num_rows - 1) * np.sqrt(3) * r_init
    global_x_offset = (1.0 - pattern_actual_width) / 2.0
    global_y_offset = (1.0 - pattern_actual_height) / 2.0
    
    current_y = r_init + global_y_offset

    circle_idx = 0
    for row_idx, num_in_row in enumerate(row_counts):
        row_x_start_offset = 0.0
        if row_idx % 2 != 0: # Odd-indexed rows are shifted
            row_x_start_offset = r_init
            
        current_x = r_init + global_x_offset + row_x_start_offset

        for _ in range(num_in_row):
            if circle_idx < n_circles:
                circles_list.append([current_x, current_y, r_init])
                circle_idx += 1
            current_x += 2 * r_init
        
        current_y += np.sqrt(3) * r_init

    return np.array(circles_list).flatten()

def _generate_grid_initial_guess(n_circles: int, num_rows: int, num_cols: int, r_base_factor: float = 0.9, rng: np.random.Generator = np.random.default_rng()) -> np.ndarray:
    """
    Generates a grid-based initial placement for N circles, with random perturbations.
    Accepts a random number generator (rng) for reproducibility. (From Inspirations 1 & 3)
    """
    circles = np.zeros((n_circles, 3))
    
    r_fit_x = 0.5 / num_cols if num_cols > 0 else 0.25
    r_fit_y = 0.5 / num_rows if num_rows > 0 else 0.25
    r_base = min(r_fit_x, r_fit_y) * r_base_factor
    if r_base < EPSILON: r_base = EPSILON

    count = 0
    for row in range(num_rows):
        for col in range(num_cols):
            if count < n_circles:
                r = r_base * (1 + (rng.uniform(-0.5, 0.5)) * 0.2)
                
                x_step = 1.0 / num_cols
                y_step = 1.0 / num_rows
                
                x = x_step * (col + 0.5)
                y = y_step * (row + 0.5)
                
                x += (rng.uniform(-0.5, 0.5)) * x_step * 0.1
                y += (rng.uniform(-0.5, 0.5)) * y_step * 0.1

                circles[count] = [x, y, r]
                count += 1
            else:
                break
        if count == n_circles:
            break
            
    while count < n_circles: 
        circles[count] = [0.5, 0.5, EPSILON]
        count += 1
        
    circles[:, 2] = np.clip(circles[:, 2], EPSILON, 0.5)
    for i in range(n_circles):
        r = circles[i, 2]
        circles[i, 0] = np.clip(circles[i, 0], r, 1 - r)
        circles[i, 1] = np.clip(circles[i, 1], r, 1 - r)

    return circles.flatten()

# --- Numba-Optimized Constraint Functions & Jacobians ---
# These functions remain as they are, they are highly efficient and provide Jacobians.
@njit(cache=True)
def _boundary_constraints_func_numba(params):
    """Numba-optimized function for boundary constraints (must be >= 0)."""
    n = len(params) // 3
    constraints = np.empty(n * 4, dtype=params.dtype)
    for i in range(n):
        x, y, r = params[i*3], params[i*3+1], params[i*3+2]
        constraints[i*4]     = x - r
        constraints[i*4 + 1] = 1 - x - r
        constraints[i*4 + 2] = y - r
        constraints[i*4 + 3] = 1 - y - r
    return constraints

@njit(cache=True)
def _boundary_constraints_jacobian_numba(params):
    """Numba-optimized Jacobian for boundary constraints."""
    n = len(params) // 3
    jac = np.zeros((n * 4, n * 3), dtype=params.dtype)
    for i in range(n):
        jac[i*4, i*3] = 1.0; jac[i*4, i*3+2] = -1.0
        jac[i*4+1, i*3] = -1.0; jac[i*4+1, i*3+2] = -1.0
        jac[i*4+2, i*3+1] = 1.0; jac[i*4+2, i*3+2] = -1.0
        jac[i*4+3, i*3+1] = -1.0; jac[i*4+3, i*3+2] = -1.0
    return jac

@njit(cache=True)
def _overlap_constraints_func_numba(params):
    """Numba-optimized function for overlap constraints (must be >= 0)."""
    n = len(params) // 3
    num_pairs = n * (n - 1) // 2
    constraints = np.empty(num_pairs, dtype=params.dtype)
    k = 0
    for i in range(n):
        for j in range(i + 1, n):
            xi, yi, ri = params[i*3], params[i*3+1], params[i*3+2]
            xj, yj, rj = params[j*3], params[j*3+1], params[j*3+2]
            dist_sq = (xi - xj)**2 + (yi - yj)**2
            radii_sum_sq = (ri + rj)**2
            constraints[k] = dist_sq - radii_sum_sq
            k += 1
    return constraints

@njit(cache=True)
def _overlap_constraints_jacobian_numba(params):
    """Numba-optimized Jacobian for overlap constraints."""
    n = len(params) // 3
    num_pairs = n * (n - 1) // 2
    jac = np.zeros((num_pairs, n * 3), dtype=params.dtype)
    k = 0
    for i in range(n):
        for j in range(i + 1, n):
            xi, yi, ri = params[i*3], params[i*3+1], params[i*3+2]
            xj, yj, rj = params[j*3], params[j*3+1], params[j*3+2]
            jac[k, i*3] = 2 * (xi - xj); jac[k, i*3+1] = 2 * (yi - yj); jac[k, i*3+2] = -2 * (ri + rj)
            jac[k, j*3] = -2 * (xi - xj); jac[k, j*3+1] = -2 * (yi - yj); jac[k, j*3+2] = -2 * (ri + rj)
            k += 1
    return jac

# --- Combined Constraint Functions for NonlinearConstraint (inspired by Insp 1/3 structure) ---
# These wrappers combine the Numba-optimized separate constraint functions.
def _combined_constraints_func(params):
    """Combines boundary and overlap constraint functions into a single array."""
    boundary_c = _boundary_constraints_func_numba(params)
    overlap_c = _overlap_constraints_func_numba(params)
    return np.concatenate((boundary_c, overlap_c))

def _combined_constraints_jacobian(params):
    """Combines boundary and overlap constraint Jacobians into a single matrix."""
    boundary_jac = _boundary_constraints_jacobian_numba(params)
    overlap_jac = _overlap_constraints_jacobian_numba(params)
    return np.vstack((boundary_jac, overlap_jac))

# --- Helper wrappers for basinhopping's older constraint format (from Insp 1/3) ---
# These are necessary because basinhopping's minimizer_kwargs expects dict-style constraints,
# but the functions it calls must align with the specific Numba functions we have.
def _bh_constraint_boundary_wrapper(params):
    """Returns only the boundary/containment constraint values from Numba function."""
    return _boundary_constraints_func_numba(params)

def _bh_constraint_overlap_wrapper(params):
    """Returns only the non-overlap constraint values from Numba function."""
    return _overlap_constraints_func_numba(params)


def circle_packing32() -> np.ndarray:
    """
    Generates an optimal arrangement of 32 circles using a robust multi-stage optimization strategy:
    1. Differential Evolution for broad global search.
    2. SLSQP for local refinement of DE's best result.
    3. Basin Hopping to escape local minima, starting from the SLSQP result.
    4. Final SLSQP for high-precision refinement of the best found solution.
    This approach combines the strengths of all inspiration programs, aiming for higher sum_radii
    and robust convergence.
    """
    n = N_CIRCLES
    
    # --- Define Bounds and Constraints ---
    bounds = []
    for _ in range(n):
        bounds.extend([(0.0, 1.0), (0.0, 1.0), (EPSILON, 0.5)])
    
    # For NonlinearConstraint, we need a single function returning all constraint values
    # and its Jacobian.
    num_boundary_constraints = n * 4
    num_overlap_constraints = n * (n - 1) // 2
    num_total_constraints = num_boundary_constraints + num_overlap_constraints
    
    lb_constraints = np.zeros(num_total_constraints)
    ub_constraints = np.full(num_total_constraints, np.inf)
    
    # NonlinearConstraint object for DE and primary SLSQP stages
    nl_constraints = NonlinearConstraint(
        _combined_constraints_func,
        lb_constraints,
        ub_constraints,
        jac=_combined_constraints_jacobian
    )

    # Dictionary-style constraints for basinhopping's minimizer_kwargs
    # This format is required for `minimize` when called by `basinhopping`.
    bh_constraints = [
        {'type': 'ineq', 'fun': _bh_constraint_boundary_wrapper, 'jac': _boundary_constraints_jacobian_numba},
        {'type': 'ineq', 'fun': _bh_constraint_overlap_wrapper, 'jac': _overlap_constraints_jacobian_numba}
    ]

    # Use a fixed random number generator for reproducibility
    rng = np.random.default_rng(42)

    # --- 1. Create a strong and diverse initial population for Differential Evolution ---
    initial_population_size = 80 # From Insp 1/3
    initial_population = np.zeros((initial_population_size, n * 3))
    
    # Seed with high-quality deterministic guesses
    initial_population[0] = _generate_hexagonal_initial_guess(n) # Tailored hex guess (Insp 1/3)
    initial_population[1] = _generate_grid_initial_guess(n, 4, 8, r_base_factor=0.95, rng=rng)
    initial_population[2] = _generate_grid_initial_guess(n, 8, 4, r_base_factor=0.95, rng=rng)
    initial_population[3] = _generate_grid_initial_guess(n, 5, 7, r_base_factor=0.9, rng=rng)
    initial_population[4] = _generate_grid_initial_guess(n, 6, 6, r_base_factor=0.9, rng=rng)
    
    # Fill the rest with random guesses (from Insp 1/3, with slightly adjusted radius range)
    for i in range(5, initial_population_size):
        for j, (lower, upper) in enumerate(bounds):
            if j % 3 == 2: # Radius
                initial_population[i, j] = rng.uniform(0.02, 0.15) # From Insp 3
            else: # x or y
                initial_population[i, j] = rng.uniform(lower, upper)

    # --- 2. Stage 1: Global Search with Differential Evolution (from Insp 1/3) ---
    print("Starting Stage 1: Differential Evolution for global search...")
    de_result = differential_evolution(
        func=_objective,
        bounds=bounds,
        constraints=nl_constraints, # Use NonlinearConstraint
        maxiter=3000,              # From Insp 1/3
        popsize=initial_population_size,
        seed=42,
        workers=-1,                # Use all available CPU cores
        disp=False,
        polish=False,              # Polish will be handled by subsequent SLSQP
        init=initial_population    # Pass the diverse initial population
    )
    print(f"DE finished. Best objective: {-de_result.fun:.6f}")

    # --- 3. Stage 2: Local Refinement with SLSQP (after DE, from Insp 1/3) ---
    print("Starting Stage 2: SLSQP refinement after DE...")
    slsqp_de_result = minimize(
        fun=_objective,
        x0=de_result.x,
        method='SLSQP',
        jac=_objective_jacobian, # Provide analytical objective Jacobian
        bounds=bounds,
        constraints=nl_constraints, # Use NonlinearConstraint
        options={'maxiter': 5000, 'ftol': 1e-12, 'disp': False} # From Insp 1/3
    )
    print(f"SLSQP after DE finished. Best objective: {-slsqp_de_result.fun:.6f}")

    # --- 4. Stage 3: Basin Hopping for Further Improvement (from Insp 1/2/3) ---
    # This stage tries to escape the local minimum found by SLSQP.
    print("Starting Stage 3: Basin Hopping for global exploration...")
    bh_minimizer_kwargs = {
        'method': 'SLSQP',
        'jac': _objective_jacobian,
        'bounds': bounds,
        'constraints': bh_constraints, # Use dictionary-style constraints with Numba Jacobians
        'options': {'maxiter': 1000, 'ftol': 1e-10} # Reduced maxiter from Insp 1/3's 2000 to balance speed/accuracy
    }
    bh_result = basinhopping(
        func=_objective,
        x0=slsqp_de_result.x, # Start from the best result of DE+SLSQP
        minimizer_kwargs=bh_minimizer_kwargs,
        niter=100, # Increased from Insp 1/3's 80, closer to Insp 2's 150.
        T=0.6,    # Adjusted "Temperature" (between Insp 1/3's 0.5 and Insp 2's 0.75)
        stepsize=0.04, # Slightly reduced stepsize for finer exploration
        seed=rng,
        disp=False
    )
    print(f"Basin Hopping finished. Best objective: {-bh_result.fun:.6f}")

    # --- 5. Stage 4: Final High-Precision Refinement with SLSQP (from Insp 1/2/3) ---
    # This is a final polish on the best solution found by Basin Hopping.
    print("Starting Stage 4: Final high-precision refinement with SLSQP...")
    final_slsqp_result = minimize(
        fun=_objective,
        x0=bh_result.x,
        method='SLSQP',
        jac=_objective_jacobian, # Provide analytical gradient
        bounds=bounds,
        constraints=nl_constraints, # Use NonlinearConstraint
        options={'maxiter': 6000, 'ftol': 1e-12, 'disp': False} # Tighter tolerance
    )
    
    # --- 6. Solution Selection and Validation ---
    def _get_score(params: np.ndarray, tolerance: float = 1e-7):
        """Returns sum_radii if valid, else -inf."""
        # Check all combined constraints
        is_valid = np.all(_combined_constraints_func(params) >= -tolerance)
        return np.sum(params[2::3]) if is_valid else -np.inf

    results = {
        "DE": de_result.x,
        "SLSQP_after_DE": slsqp_de_result.x,
        "BasinHopping": bh_result.x,
        "Final_SLSQP": final_slsqp_result.x
    }
    
    scores = {name: _get_score(params) for name, params in results.items()}
    
    best_name = max(scores, key=scores.get)
    
    if scores[best_name] > -np.inf:
        optimized_params = results[best_name]
        print(f"Best valid solution from '{best_name}' stage. Sum of radii: {scores[best_name]:.15f}")
    else:
        # Fallback: if no solution is strictly valid, choose the one with the highest objective
        # value from the most thorough search (Basin Hopping or Final SLSQP).
        print("Warning: No strictly valid solution found. Selecting best objective from final stage.")
        optimized_params = final_slsqp_result.x if final_slsqp_result.success else bh_result.x

    # --- 7. Post-processing and Formatting Output ---
    circles = optimized_params.reshape((N_CIRCLES, 3))
    # Clip coordinates and enforce minimum radius to guarantee a valid final state
    final_radii = np.maximum(circles[:, 2], EPSILON)
    final_x = np.clip(circles[:, 0], final_radii, 1.0 - final_radii)
    final_y = np.clip(circles[:, 1], final_radii, 1.0 - final_radii)
    
    return np.vstack((final_x, final_y, final_radii)).T


# EVOLVE-BLOCK-END
