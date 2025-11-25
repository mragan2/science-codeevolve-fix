# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import differential_evolution, minimize
from scipy.spatial.distance import pdist, squareform # For efficient pairwise distance calculation
import time # To measure eval_time

# Set a fixed random seed for reproducibility
np.random.seed(42)

def _params_to_circles(params: np.ndarray, n: int) -> np.ndarray:
    """Reshapes a flat parameter array into an (n, 3) circles array."""
    return params.reshape((n, 3))

def _circles_to_params(circles: np.ndarray) -> np.ndarray:
    """Flattens an (n, 3) circles array into a parameter array."""
    return circles.flatten()

def is_valid_packing(circles: np.ndarray, square_size: float = 1.0, tolerance: float = 1e-6) -> tuple[bool, str]:
    """
    Checks if a given set of circles forms a valid packing within the unit square.
    Returns True and "Valid" if all constraints are met, False and a message otherwise.
    """
    n = circles.shape[0]
    x, y, r = circles[:, 0], circles[:, 1], circles[:, 2]

    # 1. Radius positivity
    if np.any(r < tolerance):
        return False, f"Violation: Negative or too small radius found (min r: {np.min(r):.2e})."

    # 2. Containment within unit square
    # Check x-r >= 0, x+r <= square_size, y-r >= 0, y+r <= square_size
    if np.any(x - r < -tolerance) or np.any(x + r > square_size + tolerance) or \
       np.any(y - r < -tolerance) or np.any(y + r > square_size + tolerance):
        return False, "Violation: Circle outside square."

    # 3. Non-overlap constraints (vectorized)
    if n > 1:
        # Calculate pairwise squared distances between circle centers
        centers = circles[:, :2] # (N, 2) array of (x,y) coordinates
        # Use 'sqeuclidean' metric to directly get squared distances
        dist_sq_matrix = squareform(pdist(centers, 'sqeuclidean'))

        # Calculate pairwise sum of radii
        radii_sum_matrix = r[:, np.newaxis] + r[np.newaxis, :]
        min_dist_sq_matrix = radii_sum_matrix**2

        # Check for overlaps in the upper triangle (excluding diagonal)
        # We want to check if (min_dist_sq_matrix - dist_sq_matrix) > tolerance
        overlap_check = np.triu(min_dist_sq_matrix - dist_sq_matrix, k=1)
        if np.any(overlap_check > tolerance):
            # Find an example overlap for the message
            idx = np.argwhere(overlap_check > tolerance)[0]
            i, j = idx[0], idx[1]
            overlap_amount = overlap_check[i, j]
            return False, f"Violation: Overlap between circle {i} and {j}. Overlap amount: {overlap_amount:.4e}"
    
    return True, "Valid"

def _calculate_violations(circles: np.ndarray, square_size: float = 1.0, tolerance: float = 1e-7) -> float:
    """
    Calculates a total violation score for a given set of circles.
    Used for the penalty function in differential_evolution.
    """
    n = circles.shape[0]
    x, y, r = circles[:, 0], circles[:, 1], circles[:, 2]

    total_violation = 0.0
    penalty_factor = 1000.0 # Multiplier for each violation type

    # 1. Radius positivity (should be handled by bounds, but for safety in penalty)
    total_violation += np.sum(np.maximum(0, -r + tolerance)) * penalty_factor

    # 2. Containment within unit square (vectorized)
    total_violation += np.sum(np.maximum(0, r - x + tolerance)) * penalty_factor
    total_violation += np.sum(np.maximum(0, x + r - square_size + tolerance)) * penalty_factor
    total_violation += np.sum(np.maximum(0, r - y + tolerance)) * penalty_factor
    total_violation += np.sum(np.maximum(0, y + r - square_size + tolerance)) * penalty_factor

    # 3. Non-overlap constraints (vectorized)
    if n > 1:
        centers = circles[:, :2]
        dist_sq_matrix = squareform(pdist(centers, 'sqeuclidean'))
        radii_sum_matrix = r[:, np.newaxis] + r[np.newaxis, :]
        min_dist_sq_matrix = radii_sum_matrix**2

        # Calculate violations for the upper triangle (excluding diagonal)
        # A violation occurs if min_dist_sq > dist_sq
        overlap_violations = np.triu(np.maximum(0, min_dist_sq_matrix - dist_sq_matrix), k=1)
        total_violation += np.sum(overlap_violations) * penalty_factor

    return total_violation

def _objective(params: np.ndarray, n: int) -> float:
    """
    Objective function for differential_evolution.
    Minimizes -sum_radii + penalty_for_violations.
    """
    circles = _params_to_circles(params, n)
    sum_radii = np.sum(circles[:, 2])
    penalty = _calculate_violations(circles)
    return -sum_radii + penalty

def _slsqp_objective(params: np.ndarray) -> float:
    """
    Objective function for SLSQP.
    Minimizes -sum_radii (constraints handled explicitly).
    """
    circles = _params_to_circles(params, 32) # n is fixed for this problem
    return -np.sum(circles[:, 2])

def _slsqp_constraints_vectorized(params: np.ndarray, n: int, square_size: float) -> np.ndarray:
    """
    Calculates all inequality constraints for SLSQP in a vectorized manner.
    All constraints are of the form g(x) >= 0.
    """
    circles = _params_to_circles(params, n)
    x, y, r = circles[:, 0], circles[:, 1], circles[:, 2]

    # 1. Containment constraints (4*n constraints)
    # x_i - r_i >= 0, 1 - x_i - r_i >= 0, etc.
    containment_constraints = np.concatenate([
        x - r,
        square_size - x - r,
        y - r,
        square_size - y - r,
    ])

    # 2. Non-overlap constraints (n*(n-1)/2 constraints)
    # (x_i-x_j)^2 + (y_i-y_j)^2 - (r_i+r_j)^2 >= 0
    if n > 1:
        centers = circles[:, :2]
        dist_sq_matrix = squareform(pdist(centers, 'sqeuclidean'))

        radii_sum = r[:, np.newaxis] + r[np.newaxis, :]
        radii_sum_sq = radii_sum**2

        # We only need the upper triangle of the matrix to avoid redundant calculations
        iu = np.triu_indices(n, k=1)
        overlap_constraints = dist_sq_matrix[iu] - radii_sum_sq[iu]
        
        return np.concatenate([containment_constraints, overlap_constraints])
    
    return containment_constraints

def _create_slsqp_constraints(n: int, square_size: float = 1.0) -> list:
    """
    Creates a single constraint dictionary for scipy.optimize.minimize (SLSQP method)
    using a vectorized constraint function for much better performance.
    """
    return [{
        'type': 'ineq',
        'fun': _slsqp_constraints_vectorized,
        'args': (n, square_size)
    }]


def circle_packing32() -> np.ndarray:
    """
    Places 32 non-overlapping circles in the unit square in order to maximize the sum of radii.
    Uses a hybrid optimization approach: Differential Evolution for global search,
    followed by SLSQP for local refinement.

    Returns:
        circles: np.array of shape (32,3), where the i-th row (x,y,r) stores the (x,y)
                 coordinates of the i-th circle of radius r.
    """
    n = 32
    
    # Define bounds for all 3*n variables (x, y, r for each circle)
    # x, y coordinates can range from 0 to 1.
    # Radii can range from a small positive value (e.g., 1e-7) to 0.5 (max possible for a single circle).
    min_radius = 1e-7
    max_radius = 0.5
    
    bounds_optimizer = []
    for _ in range(n):
        bounds_optimizer.extend([(0, 1), (0, 1), (min_radius, max_radius)])

    # --- Step 1: Global optimization with Differential Evolution ---
    # DE explores the search space broadly, guided by the objective function
    # which includes a penalty for constraint violations.
    print(f"Starting Differential Evolution for {n} circles...")
    
    # Adjust DE parameters for potentially faster execution, given vectorization
    # Reduced maxiter slightly to balance speed and accuracy.
    # The 'polish=True' step already runs a local optimizer (L-BFGS-B or similar)
    # at the end of DE, which is often very effective.
    res_de = differential_evolution(
        _objective,
        bounds_optimizer,
        args=(n,),
        strategy='best1bin',  # Default and generally robust strategy
        maxiter=400,          # Drastically reduced to avoid timeout, DE's goal is a good starting point
        popsize=25,           # Reduced population size slightly
        tol=0.01,             # Relative tolerance for convergence
        polish=True,          # Perform L-BFGS-B local optimization at the end of DE
        workers=-1,           # Use all available CPU cores for parallel evaluation
        seed=42               # Fixed seed for reproducibility
    )
    print(f"Differential Evolution finished. Best sum of radii found (with penalty): {-res_de.fun:.6f}")
    
    # Check if DE result is valid before passing to SLSQP
    de_circles = _params_to_circles(res_de.x, n)
    is_de_valid, de_msg = is_valid_packing(de_circles)
    if not is_de_valid:
        print(f"Warning: DE's best result is not strictly valid: {de_msg}. Attempting SLSQP refinement anyway.")
    else:
        print("DE's best result is valid.")

    initial_guess_slsqp = res_de.x
    
    # --- Step 2: Local optimization with SLSQP for refinement ---
    # SLSQP fine-tunes the solution found by DE using explicit constraints,
    # which can lead to a more precise and valid packing.
    print(f"Starting SLSQP refinement for {n} circles...")
    slsqp_constraints = _create_slsqp_constraints(n)
    
    res_slsqp = minimize(
        _slsqp_objective,
        initial_guess_slsqp,
        method='SLSQP',
        bounds=bounds_optimizer, # Same bounds as DE
        constraints=slsqp_constraints,
        options={'ftol': 1e-9, 'eps': 1e-8, 'disp': False, 'maxiter': 1500} # Increased maxiter for SLSQP and stricter ftol
    )
    print(f"SLSQP finished. Best sum of radii found: {-res_slsqp.fun:.6f}")

    # Final result
    optimal_params = res_slsqp.x
    circles = _params_to_circles(optimal_params, n)

    # Validate the final packing
    is_final_valid, final_msg = is_valid_packing(circles)
    if not is_final_valid:
        print(f"CRITICAL WARNING: Final packing after SLSQP is not strictly valid: {final_msg}.")
        if not res_slsqp.success:
             print("SLSQP did not converge successfully. Consider increasing maxiter or adjusting options.")
        # If SLSQP failed or produced an invalid result, fall back to DE's best valid result if possible
        if is_de_valid:
            print("Falling back to DE's best valid result.")
            circles = de_circles
        else:
            print("Both DE and SLSQP produced invalid results. Returning SLSQP's best effort.")
    else:
        print("Final packing is valid.")

    return circles


# EVOLVE-BLOCK-END
