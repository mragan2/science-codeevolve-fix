# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import differential_evolution, minimize, NonlinearConstraint
from scipy.spatial.distance import pdist, squareform

# Constants for the problem
N_CIRCLES = 26
PENALTY_FACTOR_DE = 1e4 # Penalty for Differential Evolution's objective function
EPSILON = 1e-7        # Small value for floating point comparisons and validation tolerance
SEED = 42             # For reproducibility

def _calculate_distances_sq(params_reshaped: np.ndarray) -> np.ndarray:
    """
    Calculates squared Euclidean distances between all circle centers.
    Uses scipy.spatial.distance.pdist for efficiency.
    """
    centers = params_reshaped[:, :2]  # Extract (x, y) coordinates
    # pdist computes a condensed distance matrix, squareform converts it to a full matrix
    return squareform(pdist(centers, metric='sqeuclidean'))

def _objective_func_de(params: np.ndarray) -> float:
    """
    Objective function for differential_evolution.
    Minimizes -sum(radii) + penalties for constraint violations.
    """
    circles = params.reshape((N_CIRCLES, 3))
    x, y, r = circles[:, 0], circles[:, 1], circles[:, 2]

    # Initialize objective with negative sum of radii (since we are minimizing)
    sum_radii = np.sum(r)
    objective = -sum_radii

    penalty = 0.0

    # 1. Containment constraints: r <= x <= 1-r and r <= y <= 1-r
    # Penalize if r is negative (should be caught by bounds, but good to have a soft penalty)
    penalty += np.sum(np.maximum(0, -r)) # Should be handled by bounds, but safety
    
    # Penalize if any part of the circle is outside the [0,1] square
    penalty += np.sum(np.maximum(0, r - x))          # Left boundary (x < r)
    penalty += np.sum(np.maximum(0, x + r - 1))      # Right boundary (x + r > 1)
    penalty += np.sum(np.maximum(0, r - y))          # Bottom boundary (y < r)
    penalty += np.sum(np.maximum(0, y + r - 1))      # Top boundary (y + r > 1)

    # 2. Non-overlap constraints: (x_i - x_j)^2 + (y_i - y_j)^2 >= (r_i + r_j)^2
    dist_sq = _calculate_distances_sq(circles)
    
    # Calculate required squared distance for non-overlap
    # r[i] + r[j] for all pairs
    radii_sum_matrix = r[:, np.newaxis] + r[np.newaxis, :]
    required_dist_sq = radii_sum_matrix**2

    # Identify overlap violations (only consider upper triangle to avoid self-comparison and duplicates)
    upper_triangle_indices = np.triu_indices(N_CIRCLES, k=1)
    
    actual_dist_sq_pairs = dist_sq[upper_triangle_indices]
    required_dist_sq_pairs = required_dist_sq[upper_triangle_indices]
    
    # Add penalty for any overlap
    overlap_violations = np.maximum(0, required_dist_sq_pairs - actual_dist_sq_pairs)
    penalty += np.sum(overlap_violations)

    return objective + PENALTY_FACTOR_DE * penalty

# New objective function for local refinement (SLSQP)
def _objective_func_local(params: np.ndarray) -> float:
    """
    Objective function for local optimization (SLSQP).
    Minimizes -sum(radii). Constraints are handled separately.
    """
    circles = params.reshape((N_CIRCLES, 3))
    r = circles[:, 2]
    return -np.sum(r)

# New constraint function for local refinement (SLSQP)
def _constraints_local(params: np.ndarray) -> np.ndarray:
    """
    Returns an array of constraint values for local optimization.
    All constraints are of the form g_k(params) >= 0.
    """
    circles = params.reshape((N_CIRCLES, 3))
    x, y, r = circles[:, 0], circles[:, 1], circles[:, 2]

    # 1. Containment constraints: r <= x <= 1-r and r <= y <= 1-r
    # These are equivalent to:
    # x - r >= 0
    # 1 - x - r >= 0
    # y - r >= 0
    # 1 - y - r >= 0
    containment_constraints = np.concatenate([
        x - r,
        1 - x - r,
        y - r,
        1 - y - r
    ])

    # 2. Non-overlap constraints: (x_i - x_j)^2 + (y_i - y_j)^2 >= (r_i + r_j)^2
    # Equivalent to: (x_i - x_j)^2 + (y_i - y_j)^2 - (r_i + r_j)^2 >= 0
    dist_sq = _calculate_distances_sq(circles)
    
    radii_sum_matrix = r[:, np.newaxis] + r[np.newaxis, :]
    required_dist_sq = radii_sum_matrix**2

    upper_triangle_indices = np.triu_indices(N_CIRCLES, k=1)
    
    non_overlap_constraints = dist_sq[upper_triangle_indices] - required_dist_sq[upper_triangle_indices]
    
    return np.concatenate([containment_constraints, non_overlap_constraints])


def _validate_packing(circles: np.ndarray, n_circles: int, tolerance: float = EPSILON) -> tuple[bool, str]:
    """
    Validates a given circle packing configuration against positive radii, containment, and non-overlap.
    Returns (True, "Valid") if valid, else (False, "Error message").
    """
    if circles.shape != (n_circles, 3):
        return False, f"Invalid circles shape: {circles.shape}, expected ({n_circles}, 3)"

    x, y, r = circles[:, 0], circles[:, 1], circles[:, 2]

    # 1. Positive Radii
    # Allow for radii very close to zero if they were optimized away
    if np.any(r < -tolerance): # Should not happen with bounds, but safety check
        return False, f"Negative radius detected. Smallest: {np.min(r):.6e}"

    # 2. Containment: r <= x <= 1-r and r <= y <= 1-r
    # Check x-coordinates
    if np.any(x - r < -tolerance) or np.any(x + r > 1 + tolerance):
        return False, f"X-containment violation detected."
    # Check y-coordinates
    if np.any(y - r < -tolerance) or np.any(y + r > 1 + tolerance):
        return False, f"Y-containment violation detected."

    # 3. Non-overlap: sqrt((x_i - x_j)^2 + (y_i - y_j)^2) >= r_i + r_j
    dist_sq = _calculate_distances_sq(circles)
    
    radii_sum_matrix = r[:, np.newaxis] + r[np.newaxis, :]
    required_dist_sq = radii_sum_matrix**2

    upper_triangle_indices = np.triu_indices(n_circles, k=1)
    
    actual_dist_sq_pairs = dist_sq[upper_triangle_indices]
    required_dist_sq_pairs = required_dist_sq[upper_triangle_indices]
    
    # Check if any actual squared distance is less than the required squared distance (minus tolerance)
    if np.any(actual_dist_sq_pairs < required_dist_sq_pairs - tolerance):
        violating_indices = np.where(actual_dist_sq_pairs < required_dist_sq_pairs - tolerance)[0]
        i_violating, j_violating = upper_triangle_indices[0][violating_indices], upper_triangle_indices[1][violating_indices]
        
        first_i, first_j = i_violating[0], j_violating[0]
        actual_d = np.sqrt(actual_dist_sq_pairs[violating_indices[0]])
        required_d = np.sqrt(required_dist_sq_pairs[violating_indices[0]])
        
        return False, f"Overlap violation between circles {first_i} and {first_j}: " \
                       f"Distance {actual_d:.6f} < Required {required_d:.6f} (r{first_i}={r[first_i]:.6f}, r{first_j}={r[first_j]:.6f})"

    return True, "Valid packing"


def circle_packing26() -> np.ndarray:
    """
    Places 26 non-overlapping circles in the unit square in order to maximize the sum of radii.
    Uses a hybrid optimization approach: Differential Evolution for global search,
    followed by SLSQP for local refinement.

    Returns:
        circles: np.array of shape (26,3), where the i-th row (x,y,r) stores the (x,y) coordinates of the i-th circle of radius r.
    """
    n = N_CIRCLES
    
    # Define parameter bounds for each circle: (x, y, r)
    # These are fixed bounds for the parameters, not the dependent containment constraints.
    param_bounds = []
    for _ in range(n):
        param_bounds.append((0.0, 1.0))       # x_i
        param_bounds.append((0.0, 1.0))       # y_i
        param_bounds.append((EPSILON, 0.5))   # r_i (radius must be positive, max 0.5)

    # Stage 1: Global Search with Differential Evolution
    # Parameters tuned for a balance of exploration and runtime.
    de_strategy = 'best2bin'
    de_maxiter = 1500
    de_popsize = 10
    de_tol = 1e-3
    
    np.random.seed(SEED) # Ensure DE's internal random state is also seeded
    
    de_result = differential_evolution(
        _objective_func_de, # Use the penalty-based objective
        param_bounds,
        strategy=de_strategy,
        maxiter=de_maxiter,
        popsize=de_popsize,
        tol=de_tol,
        seed=SEED, # Pass seed to DE for reproducibility
        disp=False,
        workers=-1 # Use all available CPU cores for parallelization
    )
    
    # Use the best result from DE as the starting point for local optimization
    initial_guess_local = de_result.x
    best_de_circles = initial_guess_local.reshape((n, 3))
    is_de_valid, de_validation_msg = _validate_packing(best_de_circles, n, tolerance=EPSILON)
    de_sum_radii = -_objective_func_local(initial_guess_local) # Calculate actual sum_radii

    # Stage 2: Local Refinement with SLSQP
    # Use explicit constraints for better precision and constraint satisfaction.
    
    # Create NonlinearConstraint for all inequality constraints (g_k(params) >= 0 for all k)
    nlc = NonlinearConstraint(_constraints_local, lb=0, ub=np.inf)

    slsqp_result = minimize(
        _objective_func_local, # Use the pure sum_radii objective
        initial_guess_local,
        method='SLSQP',
        bounds=param_bounds, # Use the same parameter bounds
        constraints=[nlc], # Pass explicit nonlinear constraints
        options={'ftol': 1e-8, 'maxiter': 1000, 'disp': False} # Stricter tolerance for local refinement
    )

    optimal_params = slsqp_result.x
    optimal_circles = optimal_params.reshape((n, 3))
    
    # Validate the final solution from SLSQP
    is_slsqp_valid, slsqp_validation_msg = _validate_packing(optimal_circles, n, tolerance=EPSILON)
    slsqp_sum_radii = -slsqp_result.fun if slsqp_result.success else 0.0 # Get sum_radii if successful

    # Final Solution Selection Logic:
    # Prioritize validity. If both are valid, choose the one with higher sum_radii.
    # If SLSQP is invalid, but DE was valid, prefer DE's result.
    final_circles = optimal_circles
    final_sum_radii = slsqp_sum_radii

    if not is_slsqp_valid:
        # If SLSQP result is invalid, check if DE's result was valid and use it as fallback.
        if is_de_valid:
            final_circles = best_de_circles
            final_sum_radii = de_sum_radii
        # Else (both invalid or DE was also invalid), proceed with SLSQP's best attempt,
        # but the validation framework will report an error.
    elif is_de_valid and de_sum_radii > slsqp_sum_radii + EPSILON:
        # If SLSQP is valid but DE found a better valid solution, prefer DE's.
        final_circles = best_de_circles
        final_sum_radii = de_sum_radii

    # Ensure radii are not negative (due to floating point noise, they might be slightly negative)
    final_circles[:, 2] = np.maximum(0, final_circles[:, 2])
    
    # Ensure very small radii are zeroed out if they effectively disappeared
    final_circles[final_circles[:, 2] < EPSILON, 2] = 0.0

    return final_circles


# EVOLVE-BLOCK-END
