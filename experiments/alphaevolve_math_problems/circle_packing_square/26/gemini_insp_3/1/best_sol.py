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
            x = np.clip(x, initial_r, 1 - initial_r)
            y = np.clip(y, initial_r, 1 - initial_r)
            initial_circles.append([x, y, initial_r])
        y_current += y_step
    return np.array(initial_circles).flatten()

# Helper function to generate a random initial guess with small radii.
def _generate_random_guess(n: int, initial_r: float = 0.01, rng=None) -> np.ndarray:
    if rng is None:
        rng = np.random.default_rng() # Use a default RNG if not provided

    initial_circles = []
    for _ in range(n):
        # Ensure circles start within bounds, considering their initial small radius
        x = rng.uniform(initial_r, 1 - initial_r)
        y = rng.uniform(initial_r, 1 - initial_r)
        initial_circles.append([x, y, initial_r])
    return np.array(initial_circles).flatten()


def circle_packing26() -> np.ndarray:
    """
    Places exactly 26 non-overlapping circles within a unit square [0,1] × [0,1],
    maximizing the sum of their radii using a multi-start optimization strategy.

    Returns:
        circles: np.array of shape (26,3), where the i-th row (x,y,r) stores the 
                 (x,y) coordinates of the i-th circle of radius r.
    """
    n = 26 # Number of circles to pack

    # --- Generate Multiple Initial Guesses ---
    initial_guesses = []
    # Use dedicated, seeded RNGs for reproducible perturbations and random guesses.
    rng_for_perturbation = np.random.default_rng(42)
    rng_for_random_guess = np.random.default_rng(43) # Different seed for truly random start

    # Guess 1: Hexagonal layout (5,5,5,5,6) with standard initial radius
    x0_hex1 = _generate_hex_guess([5, 5, 5, 5, 6], n, initial_r=0.08)
    initial_guesses.append(x0_hex1)

    # Guess 2: Hexagonal layout (4,5,6,5,6) with a slightly smaller initial radius
    x0_hex2 = _generate_hex_guess([4, 5, 6, 5, 6], n, initial_r=0.075)
    initial_guesses.append(x0_hex2)

    # Guess 3: Perturbed version of Hexagonal 1
    x0_perturbed_hex1 = x0_hex1.copy()
    perturbation_scale1 = 0.02
    pos_perturbations1 = rng_for_perturbation.uniform(-perturbation_scale1, perturbation_scale1, 2 * n)
    x0_perturbed_hex1[0::3] += pos_perturbations1[0:n] # Perturb x-coords
    x0_perturbed_hex1[1::3] += pos_perturbations1[n:2*n] # Perturb y-coords
    initial_guesses.append(np.clip(x0_perturbed_hex1, 0, 1)) # Clip to [0,1] range

    # Guess 4: Square grid layout (4 rows of 5, 2 rows of 3)
    x0_square_grid1 = _generate_square_guess([5, 5, 5, 5, 3, 3], n, initial_r=0.09)
    initial_guesses.append(x0_square_grid1)

    # Guess 5: Another hexagonal layout (6 rows: 5,4,5,4,5,3) with a medium initial radius
    x0_hex3 = _generate_hex_guess([5, 4, 5, 4, 5, 3], n, initial_r=0.078)
    initial_guesses.append(x0_hex3)

    # Guess 6: Another square grid (5 rows: 6,5,5,5,5) with standard initial radius
    x0_square_grid2 = _generate_square_guess([6, 5, 5, 5, 5], n, initial_r=0.085)
    initial_guesses.append(x0_square_grid2)

    # Guess 7: Perturbed version of Hexagonal 2 with slightly larger perturbation
    x0_perturbed_hex2 = x0_hex2.copy()
    perturbation_scale2 = 0.03
    pos_perturbations2 = rng_for_perturbation.uniform(-perturbation_scale2, perturbation_scale2, 2 * n)
    x0_perturbed_hex2[0::3] += pos_perturbations2[0:n]
    x0_perturbed_hex2[1::3] += pos_perturbations2[n:2*n]
    initial_guesses.append(np.clip(x0_perturbed_hex2, 0, 1))

    # Guess 8: Random placement with very small radii
    x0_random_small_r = _generate_random_guess(n, initial_r=0.01, rng=rng_for_random_guess)
    initial_guesses.append(x0_random_small_r)

    # --- Define Variable Bounds ---
    bounds = []
    for _ in range(n):
        bounds.append((0.0, 1.0))   # x-coordinate bound
        bounds.append((0.0, 1.0))   # y-coordinate bound
        bounds.append((1e-7, 0.5))  # radius bound (positive and max 0.5), tightened min_r slightly

    # --- Define Constraints for Scipy Optimizer ---
    cons = ({'type': 'ineq', 'fun': constraints_function, 'args': (n,)})

    # --- Run Multi-Start Optimization ---
    best_result = None
    best_score = -np.inf

    for i, x0 in enumerate(initial_guesses):
        if len(x0) != 3 * n:
            print(f"Warning: Initial guess {i} has wrong dimensions ({len(x0)} != {3*n}). Skipping.")
            continue
        
        # Ensure initial radii are valid for bounds
        x0_radii_bound_min = bounds[2][0]
        x0_radii_bound_max = bounds[2][1]
        x0[2::3] = np.clip(x0[2::3], x0_radii_bound_min, x0_radii_bound_max)
        # Ensure initial x, y are valid for bounds
        x0_xy_bound_min = bounds[0][0] # x and y have same min bound
        x0_xy_bound_max = bounds[0][1] # x and y have same max bound
        x0[0::3] = np.clip(x0[0::3], x0_xy_bound_min, x0_xy_bound_max)
        x0[1::3] = np.clip(x0[1::3], x0_xy_bound_min, x0_xy_bound_max)


        result = minimize(
            objective_function,
            x0,
            args=(n,),
            method='SLSQP',
            bounds=bounds,
            constraints=cons,
            # Increased maxiter for deeper search, tightened ftol, and refined eps for gradient approximation.
            options={'maxiter': 25000, 'ftol': 1e-10, 'disp': False, 'eps': 1e-8}
        )
        
        current_score = -result.fun if result.success else -np.inf

        if current_score > best_score:
            best_score = current_score
            best_result = result
            # Optional: print(f"New best score found: {best_score:.6f} from guess {i}")

    if best_result is None:
        # Fallback: if all optimizations fail, try again with the first guess and log error.
        print("Error: All optimization runs failed to converge or produced invalid results. Retrying with a robust fallback guess.")
        # Fallback to a well-performing hexagonal guess if all others fail
        fallback_x0 = _generate_hex_guess([5, 5, 5, 5, 6], n, initial_r=0.08)
        fallback_x0[2::3] = np.clip(fallback_x0[2::3], bounds[2][0], bounds[2][1]) # Clip radii
        fallback_x0[0::3] = np.clip(fallback_x0[0::3], bounds[0][0], bounds[0][1]) # Clip x
        fallback_x0[1::3] = np.clip(fallback_x0[1::3], bounds[0][0], bounds[0][1]) # Clip y

        best_result = minimize(
            objective_function, fallback_x0, args=(n,), method='SLSQP', bounds=bounds,
            constraints=cons, options={'maxiter': 25000, 'ftol': 1e-10, 'disp': False, 'eps': 1e-8}
        )
        if not best_result.success:
            print(f"Critical Error: Fallback optimization failed: {best_result.message}")
            # If even fallback fails, we might return a suboptimal or invalid result, but must return something.

    if not best_result.success:
        print(f"Optimization warning for best found solution: {best_result.message}")

    # Reshape the flattened result array back into the (n, 3) format (x, y, r)
    optimized_circles = best_result.x.reshape((n, 3))
    
    # Final validation: Ensure all radii are strictly positive,
    # clipping to the minimum allowed radius from bounds.
    optimized_circles[:, 2] = np.maximum(optimized_circles[:, 2], bounds[2][0])

    return optimized_circles


# EVOLVE-BLOCK-END
