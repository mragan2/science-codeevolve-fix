# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
from numba import njit
import time # Included for potential debugging/timing, not strictly used in final output.

# Set a fixed random seed for reproducibility to ensure deterministic results across runs.
np.random.seed(42)

@njit(cache=True)
def _check_overlaps_numba(coords_radii_flat: np.ndarray, n: int) -> np.ndarray:
    """
    Numba-optimized function to calculate overlap constraint violations.
    Returns: an array of (xi - xj)^2 + (yi - yj)^2 - (ri + rj)^2 for all pairs.
    These values must be >= 0 for no overlap, as required by scipy.optimize.minimize's
    inequality constraints (g(x) >= 0). A negative value indicates an overlap.
    """
    num_pairs = n * (n - 1) // 2
    # Pre-allocate array for efficiency to avoid Python list append overhead
    violations = np.empty(num_pairs, dtype=coords_radii_flat.dtype)
    k = 0 # Index for the violations array
    
    for i in range(n):
        # Extract x, y, r for the i-th circle
        xi = coords_radii_flat[i * 3]
        yi = coords_radii_flat[i * 3 + 1]
        ri = coords_radii_flat[i * 3 + 2]
        
        # Iterate over unique pairs (j > i) to avoid redundant checks and self-comparison
        for j in range(i + 1, n):
            # Extract x, y, r for the j-th circle
            xj = coords_radii_flat[j * 3]
            yj = coords_radii_flat[j * 3 + 1]
            rj = coords_radii_flat[j * 3 + 2]

            # Calculate the squared minimum distance required for no overlap
            min_dist_sq = (ri + rj) ** 2
            # Calculate the actual squared distance between circle centers
            actual_dist_sq = (xi - xj) ** 2 + (yi - yj) ** 2
            
            # The constraint is actual_dist_sq - min_dist_sq >= 0.
            # If this value is negative, it means the circles overlap.
            violations[k] = actual_dist_sq - min_dist_sq
            k += 1
    return violations

def objective_function(x: np.ndarray, n: int) -> float:
    """
    Objective function to maximize the sum of radii.
    scipy.optimize.minimize performs minimization, so we return the negative sum of radii.
    x is a flat array: [x1, y1, r1, x2, y2, r2, ..., xN, yN, rN]
    """
    # Radii are located at indices 2, 5, 8, ... (i.e., every 3rd element starting from index 2)
    radii = x[2::3]
    return -np.sum(radii)

def constraints_function(x: np.ndarray, n: int) -> np.ndarray:
    """
    Combines all inequality constraints g(x) >= 0 for the optimization problem.
    1. Boundary constraints: Ensure each circle is fully contained within the unit square.
       - ri <= xi <= 1-ri  =>  xi - ri >= 0  AND  1 - xi - ri >= 0
       - ri <= yi <= 1-ri  =>  yi - ri >= 0  AND  1 - yi - ri >= 0
    2. Non-overlap constraints: Ensure no two circles intersect.
       - (xi - xj)^2 + (yi - yj)^2 >= (ri + rj)^2  =>  (xi - xj)^2 + (yi - yj)^2 - (ri + rj)^2 >= 0
    """
    # Calculate total number of constraints
    num_boundary_constraints = 4 * n # 4 constraints per circle for boundaries
    num_overlap_constraints = n * (n - 1) // 2 # Number of unique pairs for overlap checks
    total_constraints = num_boundary_constraints + num_overlap_constraints
    
    # Pre-allocate array for all constraint values for efficiency
    c = np.empty(total_constraints, dtype=x.dtype)

    # 1. Boundary constraints
    k = 0 # Current index in the constraints array 'c'
    for i in range(n):
        # Extract x, y, r for the current circle
        xi = x[i * 3]
        yi = x[i * 3 + 1]
        ri = x[i * 3 + 2]
        
        # Left boundary: center x minus radius must be >= 0
        c[k] = xi - ri          
        k += 1
        # Right boundary: center x plus radius must be <= 1 (so 1 - (xi + ri) >= 0)
        c[k] = 1 - xi - ri      
        k += 1
        # Bottom boundary: center y minus radius must be >= 0
        c[k] = yi - ri          
        k += 1
        # Top boundary: center y plus radius must be <= 1 (so 1 - (yi + ri) >= 0)
        c[k] = 1 - yi - ri      
        k += 1

    # 2. Non-overlap constraints
    # Fill the remaining part of the array with values from the numba-optimized overlap checker
    c[k:] = _check_overlaps_numba(x, n)
    
    return c

# Helper function to generate a hexagonal initial guess based on row distribution.
def _generate_hex_guess(rows_circles: list[int], n: int, initial_r: float = 0.08) -> np.ndarray:
    initial_circles = []
    y_current = initial_r
    x_step = 2 * initial_r
    y_step = np.sqrt(3) * initial_r

    for row_idx, num_circles_in_row in enumerate(rows_circles):
        x_start = initial_r if row_idx % 2 == 0 else initial_r + initial_r
        for col_idx in range(num_circles_in_row):
            if len(initial_circles) >= n: break
            x = x_start + col_idx * x_step
            y = y_current
            # Ensure circles start within bounds, considering their initial radius
            x = np.clip(x, initial_r, 1 - initial_r)
            y = np.clip(y, initial_r, 1 - initial_r)
            initial_circles.append([x, y, initial_r])
        y_current += y_step
    return np.array(initial_circles).flatten()

# Helper function to generate a square grid initial guess.
def _generate_square_guess(rows_circles: list[int], n: int, initial_r: float = 0.09) -> np.ndarray:
    initial_circles = []
    y_current = initial_r
    x_step = 2 * initial_r
    y_step = 2 * initial_r # Square grid has 2r vertical spacing

    for row_idx, num_circles_in_row in enumerate(rows_circles):
        # Calculate total width occupied by centers and radii for centering
        total_width_occupied = (num_circles_in_row - 1) * x_step + 2 * initial_r
        x_offset_for_centering = (1.0 - total_width_occupied) / 2.0
        x_start_centered = x_offset_for_centering + initial_r if x_offset_for_centering > 1e-9 else initial_r

        for col_idx in range(num_circles_in_row):
            if len(initial_circles) >= n: break
            x = x_start_centered + col_idx * x_step
            y = y_current
            # Ensure circles start within bounds, considering their initial radius
            x = np.clip(x, initial_r, 1 - initial_r)
            y = np.clip(y, initial_r, 1 - initial_r)
            initial_circles.append([x, y, initial_r])
        y_current += y_step
    return np.array(initial_circles).flatten()

# Helper function to generate a random initial guess with small radii.
def _generate_random_guess(n: int, initial_r: float = 0.01, rng=None) -> np.ndarray:
    if rng is None:
        rng = np.random.default_rng()

    initial_circles = []
    for _ in range(n):
        # Ensure circles start within bounds, considering their initial small radius
        x = rng.uniform(initial_r, 1 - initial_r)
        y = rng.uniform(initial_r, 1 - initial_r)
        initial_circles.append([x, y, initial_r])
    return np.array(initial_circles).flatten()

# Helper function for a corner/edge focused initial guess
def _generate_corner_edge_guess(n: int, rng=None) -> np.ndarray:
    if rng is None:
        rng = np.random.default_rng()

    initial_circles_list = [] # Use a list to build, then convert to numpy array
    
    # Place 4 large circles in corners
    corner_r = 0.12 # Moderately large radius
    initial_circles_list.append([corner_r, corner_r, corner_r])
    initial_circles_list.append([1 - corner_r, corner_r, corner_r])
    initial_circles_list.append([corner_r, 1 - corner_r, corner_r])
    initial_circles_list.append([1 - corner_r, 1 - corner_r, corner_r])
    
    # Place 4 medium circles along edges
    edge_r = 0.09
    initial_circles_list.append([0.5, edge_r, edge_r])
    initial_circles_list.append([edge_r, 0.5, edge_r])
    initial_circles_list.append([0.5, 1 - edge_r, edge_r])
    initial_circles_list.append([1 - edge_r, 0.5, edge_r])

    # Place 1 large circle in the center
    center_r = 0.15
    initial_circles_list.append([0.5, 0.5, center_r])
    
    # Fill remaining circles with smaller radii, randomly
    while len(initial_circles_list) < n:
        r_fill = rng.uniform(0.01, 0.03)
        x_fill = rng.uniform(r_fill, 1 - r_fill) # These are already clipped based on r_fill
        y_fill = rng.uniform(r_fill, 1 - r_fill)
        initial_circles_list.append([x_fill, y_fill, r_fill])

    initial_circles_array = np.array(initial_circles_list)

    # Apply clipping for all circles based on their initial radius and overall bounds
    initial_radii = initial_circles_array[:, 2]
    initial_circles_array[:, 0] = np.clip(initial_circles_array[:, 0], initial_radii, 1 - initial_radii)
    initial_circles_array[:, 1] = np.clip(initial_circles_array[:, 1], initial_radii, 1 - initial_radii)
    initial_circles_array[:, 2] = np.clip(initial_circles_array[:, 2], 1e-7, 0.5) # Also clip radii to overall bounds

    return initial_circles_array.flatten()

# Helper function for a concentric/layered initial guess
def _generate_concentric_guess(n: int, rng=None) -> np.ndarray:
    if rng is None:
        rng = np.random.default_rng()

    initial_circles_list = []
    
    # Layer 1: Center circle
    if len(initial_circles_list) < n:
        r1 = 0.15
        initial_circles_list.append([0.5, 0.5, r1])

    # Layer 2: Ring of circles around the center
    num_ring1 = min(6, n - len(initial_circles_list))
    r2 = 0.08
    # Distance from center should be r1 + r2 + a small gap
    dist_from_center1 = r1 + r2 + 0.005 
    for i in range(num_ring1):
        if len(initial_circles_list) >= n: break
        angle = 2 * np.pi * i / num_ring1
        x = 0.5 + dist_from_center1 * np.cos(angle)
        y = 0.5 + dist_from_center1 * np.sin(angle)
        initial_circles_list.append([x, y, r2])

    # Layer 3: Second ring of circles
    num_ring2 = min(12, n - len(initial_circles_list))
    r3 = 0.05
    # Distance from center for second ring
    dist_from_center2 = dist_from_center1 + r2 + r3 + 0.005 
    for i in range(num_ring2):
        if len(initial_circles_list) >= n: break
        angle = 2 * np.pi * i / num_ring2
        x = 0.5 + dist_from_center2 * np.cos(angle)
        y = 0.5 + dist_from_center2 * np.sin(angle)
        initial_circles_list.append([x, y, r3])

    # Fill remaining circles with smaller radii, randomly
    while len(initial_circles_list) < n:
        r_fill = rng.uniform(0.01, 0.03)
        x_fill = rng.uniform(r_fill, 1 - r_fill)
        y_fill = rng.uniform(r_fill, 1 - r_fill)
        initial_circles_list.append([x_fill, y_fill, r_fill])

    initial_circles_array = np.array(initial_circles_list)
    initial_radii = initial_circles_array[:, 2]
    initial_circles_array[:, 0] = np.clip(initial_circles_array[:, 0], initial_radii, 1 - initial_radii)
    initial_circles_array[:, 1] = np.clip(initial_circles_array[:, 1], initial_radii, 1 - initial_radii)
    initial_circles_array[:, 2] = np.clip(initial_circles_array[:, 2], 1e-7, 0.5)

    return initial_circles_array.flatten()


def circle_packing26() -> np.ndarray:
    """
    Places exactly 26 non-overlapping circles within a unit square [0,1] × [0,1],
    maximizing the sum of their radii using a hybrid global-local optimization strategy.

    Returns:
        circles: np.array of shape (26,3), where the i-th row (x,y,r) stores the 
                 (x,y) coordinates of the i-th circle of radius r.
    """
    from scipy.optimize import differential_evolution
    
    n = 26 # Number of circles to pack

    # --- Define Variable Bounds ---
    bounds = []
    for _ in range(n):
        bounds.append((0.0, 1.0))   # x-coordinate bound
        bounds.append((0.0, 1.0))   # y-coordinate bound
        bounds.append((1e-7, 0.5))  # radius bound (positive and max 0.5)

    # --- Define Constraints for Scipy Optimizer ---
    cons = ({'type': 'ineq', 'fun': constraints_function, 'args': (n,)})

    # --- Phase 1: Global Optimization with Differential Evolution ---
    global_solution = None
    global_score = -np.inf
    
    try:
        # Optimized DE parameters for balance between exploration and efficiency
        global_result = differential_evolution(
            objective_function,
            bounds,
            args=(n,),
            constraints=cons,
            maxiter=2000,      # Moderate iterations for reasonable computation time
            popsize=20,        # Balanced population size for exploration vs efficiency
            tol=1e-5,          # Convergence tolerance for DE
            seed=42,           # For reproducibility
            disp=False,
            polish=False       # We'll do our own local polishing with SLSQP
        )
        
        if global_result.success:
            global_solution = global_result.x
            global_score = -global_result.fun
        else:
            global_solution = None
            global_score = -np.inf
    except Exception as e:
        global_solution = None
        global_score = -np.inf

    # --- Phase 2: Local Refinement with SLSQP ---
    best_result = None
    best_score = -np.inf

    # If global optimization succeeded, refine that solution first
    if global_solution is not None:
        try:
            local_result = minimize(
                objective_function,
                global_solution,
                args=(n,),
                method='SLSQP',
                bounds=bounds,
                constraints=cons,
                options={'maxiter': 10000, 'ftol': 1e-10, 'disp': False, 'eps': 1e-8}
            )
            
            local_score = -local_result.fun if local_result.success else -np.inf
            if local_score > best_score:
                best_score = local_score
                best_result = local_result
        except Exception as e:
            pass

    # --- Phase 3: Conditional Multi-Start Local Optimization ---
    # Only run extensive multi-start if we haven't found a sufficiently good solution
    if best_score < 2.63:  # Threshold below benchmark (~2.6358)
        # Use dedicated, seeded RNGs for reproducible perturbations and random guesses.
        rng_for_perturbation = np.random.default_rng(44) # New seed for fallback
        rng_for_random_guess = np.random.default_rng(45) # New seed for fallback
        rng_for_corner_edge_guess = np.random.default_rng(46) # For random parts of corner/edge guess
        rng_for_concentric_guess = np.random.default_rng(47) # For random parts of concentric guess

        # Generate focused set of high-quality initial guesses
        fallback_initial_guesses = []

        # Core hexagonal patterns (most promising)
        fallback_initial_guesses.append(_generate_hex_guess([5, 5, 5, 5, 6], n, initial_r=0.08))
        fallback_initial_guesses.append(_generate_hex_guess([4, 5, 6, 5, 6], n, initial_r=0.075))
        fallback_initial_guesses.append(_generate_hex_guess([5, 6, 5, 6, 4], n, initial_r=0.078))
        
        # Perturbed hexagonal with radius variations
        x0_perturbed_hex_fallback = fallback_initial_guesses[0].copy()
        perturbation_scale_pos = 0.02
        perturbation_scale_r = 0.005
        pos_perturbations_fallback = rng_for_perturbation.uniform(-perturbation_scale_pos, perturbation_scale_pos, 2 * n)
        r_perturbations_fallback = rng_for_perturbation.uniform(-perturbation_scale_r, perturbation_scale_r, n)
        x0_perturbed_hex_fallback[0::3] += pos_perturbations_fallback[0:n]
        x0_perturbed_hex_fallback[1::3] += pos_perturbations_fallback[n:2*n]
        x0_perturbed_hex_fallback[2::3] += r_perturbations_fallback
        fallback_initial_guesses.append(x0_perturbed_hex_fallback)

        # Square grid patterns
        fallback_initial_guesses.append(_generate_square_guess([5, 5, 5, 5, 3, 3], n, initial_r=0.09))
        fallback_initial_guesses.append(_generate_square_guess([6, 5, 5, 5, 5], n, initial_r=0.085))
        
        # Strategic patterns
        fallback_initial_guesses.append(_generate_corner_edge_guess(n, rng=rng_for_corner_edge_guess))
        fallback_initial_guesses.append(_generate_concentric_guess(n, rng=rng_for_concentric_guess))
        
        # Random exploration
        fallback_initial_guesses.append(_generate_random_guess(n, initial_r=0.01, rng=rng_for_random_guess))

        # Run multi-start optimization on fallback guesses
        for i, x0 in enumerate(fallback_initial_guesses):
            if len(x0) != 3 * n:
                continue
            
            # Ensure bounds compliance
            x0_clipped = x0.copy()
            x0_clipped[2::3] = np.clip(x0_clipped[2::3], bounds[2][0], bounds[2][1])
            radii_x0 = x0_clipped[2::3]
            x0_clipped[0::3] = np.clip(x0_clipped[0::3], radii_x0, 1 - radii_x0)
            x0_clipped[1::3] = np.clip(x0_clipped[1::3], radii_x0, 1 - radii_x0)

            try:
                result = minimize(
                    objective_function,
                    x0_clipped,
                    args=(n,),
                    method='SLSQP',
                    bounds=bounds,
                    constraints=cons,
                    options={'maxiter': 15000, 'ftol': 1e-10, 'disp': False, 'eps': 1e-8}
                )
                
                current_score = -result.fun if result.success else -np.inf
                if current_score > best_score:
                    best_score = current_score
                    best_result = result
            except Exception as e:
                continue

    # --- Final Fallback ---
    if best_result is None:
        # Last resort: simple hexagonal layout optimization
        x0_fallback = _generate_hex_guess([5, 5, 5, 5, 6], n, initial_r=0.08)
        x0_fallback[2::3] = np.clip(x0_fallback[2::3], bounds[2][0], bounds[2][1])
        radii_fallback = x0_fallback[2::3]
        x0_fallback[0::3] = np.clip(x0_fallback[0::3], radii_fallback, 1 - radii_fallback)
        x0_fallback[1::3] = np.clip(x0_fallback[1::3], radii_fallback, 1 - radii_fallback)
        
        best_result = minimize(
            objective_function, x0_fallback, args=(n,), method='SLSQP', bounds=bounds,
            constraints=cons, options={'maxiter': 10000, 'ftol': 1e-9, 'disp': False, 'eps': 1e-7}
        )

    if not best_result.success:
        print(f"Optimization warning: {best_result.message}")

    # Reshape and validate result
    optimized_circles = best_result.x.reshape((n, 3))
    optimized_circles[:, 2] = np.maximum(optimized_circles[:, 2], bounds[2][0])

    return optimized_circles


# EVOLVE-BLOCK-END
