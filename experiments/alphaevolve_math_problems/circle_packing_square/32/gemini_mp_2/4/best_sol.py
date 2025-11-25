# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize, basinhopping # Added import

# Helper function for validation (moved outside for general utility)
def validate_circles(circles: np.ndarray) -> dict:
    n = circles.shape[0]
    x = circles[:, 0]
    y = circles[:, 1]
    r = circles[:, 2]

    results = {
        'all_positive_radii': bool(np.all(r > 0)),
        'all_contained': True,
        'no_overlaps': True,
        'violated_containment_count': 0,
        'violated_overlap_count': 0,
        'sum_radii': np.sum(r)
    }

    # Check containment
    # Using a small tolerance for floating point comparisons to avoid spurious violations
    containment_violations_x_min = np.sum(x - r < -1e-9)
    containment_violations_x_max = np.sum(1 - x - r < -1e-9)
    containment_violations_y_min = np.sum(y - r < -1e-9)
    containment_violations_y_max = np.sum(1 - y - r < -1e-9)

    results['violated_containment_count'] = (
        containment_violations_x_min + containment_violations_x_max +
        containment_violations_y_min + containment_violations_y_max
    )
    if results['violated_containment_count'] > 0:
        results['all_contained'] = False

    # Check non-overlap (vectorized for performance)
    if n > 1:
        # Create distance matrices using broadcasting
        dx = x[:, np.newaxis] - x
        dy = y[:, np.newaxis] - y
        dist_sq = dx**2 + dy**2

        # Create sum of radii matrix
        r_sum = r[:, np.newaxis] + r
        min_dist_sq = r_sum**2

        # Get upper triangle indices (k=1 to exclude diagonal)
        i, j = np.triu_indices(n, k=1)

        # Check for violations in the unique pairs
        violations_mask = dist_sq[i, j] < min_dist_sq[i, j] - 1e-9
        overlap_violations = np.sum(violations_mask)
    else:
        overlap_violations = 0
        
    results['violated_overlap_count'] = int(overlap_violations)
    if overlap_violations > 0:
        results['no_overlaps'] = False
        
    return results


def circle_packing32() -> np.ndarray:
    """
    Places 32 non-overlapping circles in the unit square in order to maximize the sum of radii.

    Returns:
        circles: np.array of shape (32,3), where the i-th row (x,y,r) stores the (x,y) coordinates of the i-th circle of radius r.
    """
    n = 32
    BENCHMARK_SUM_RADII = 2.937944526205518

    # Helper to unpack parameters from the 1D array [x1, y1, r1, ..., xN, yN, rN]
    def unpack_params(params):
        x = params[0::3]
        y = params[1::3]
        r = params[2::3]
        return x, y, r

    # Objective function: minimize negative sum of radii to maximize sum_radii
    def objective(params):
        _, _, r = unpack_params(params)
        return -np.sum(r)

    # Define constraints for scipy.optimize.minimize
    constraints = []

    # 1. Containment constraints: r <= x <= 1-r and r <= y <= 1-r
    # These are vectorized for efficiency, each returning an array of constraint values (fun >= 0)
    # x - r >= 0
    constraints.append({
        'type': 'ineq',
        'fun': lambda params: params[0::3] - params[2::3]
    })
    # 1 - x - r >= 0
    constraints.append({
        'type': 'ineq',
        'fun': lambda params: 1 - params[0::3] - params[2::3]
    })
    # y - r >= 0
    constraints.append({
        'type': 'ineq',
        'fun': lambda params: params[1::3] - params[2::3]
    })
    # 1 - y - r >= 0
    constraints.append({
        'type': 'ineq',
        'fun': lambda params: 1 - params[1::3] - params[2::3]
    })

    # 2. Non-overlap constraints: (xi - xj)^2 + (yi - yj)^2 - (ri + rj)^2 >= 0
    # This function is vectorized for performance, returning an array of values for all unique pairs.
    def non_overlap_constraint(params):
        x, y, r = unpack_params(params)
        
        # Create distance matrices using broadcasting
        dx = x[:, np.newaxis] - x
        dy = y[:, np.newaxis] - y
        dist_sq = dx**2 + dy**2
        
        # Create sum of radii matrix
        r_sum = r[:, np.newaxis] + r
        min_dist_sq = r_sum**2
        
        # The constraint is dist_sq - min_dist_sq >= 0. We evaluate this for all unique pairs.
        # np.triu_indices gives the indices of the upper triangle (i < j)
        i, j = np.triu_indices(n, k=1)
        
        return (dist_sq - min_dist_sq)[i, j]

    constraints.append({
        'type': 'ineq',
        'fun': non_overlap_constraint
    })

    # Define bounds for x, y, r parameters
    # x and y coordinates are in [0, 1], radii are in [1e-6, 0.5]
    bounds = []
    for _ in range(n):
        bounds.extend([(0, 1), (0, 1), (1e-6, 0.5)]) # (x_min, x_max), (y_min, y_max), (r_min, r_max)

    # Function to generate a smarter initial guess for circle parameters
    def generate_initial_guess():
        # Attempt to place circles without initial overlap to give the optimizer a better start.
        max_attempts_per_circle = 100
        circles = np.zeros((n, 3))
        
        for i in range(n):
            r_init = np.random.uniform(0.01, 0.05)
            placed = False
            for _ in range(max_attempts_per_circle):
                x_init = np.random.uniform(r_init, 1 - r_init)
                y_init = np.random.uniform(r_init, 1 - r_init)
                
                # Check for overlap with previously placed circles
                is_overlapping = False
                if i > 0:
                    prev_circles = circles[:i, :]
                    dist_sq = (prev_circles[:, 0] - x_init)**2 + (prev_circles[:, 1] - y_init)**2
                    min_dist_sq = (prev_circles[:, 2] + r_init)**2
                    if np.any(dist_sq < min_dist_sq):
                        is_overlapping = True
                
                if not is_overlapping:
                    circles[i] = [x_init, y_init, r_init]
                    placed = True
                    break
            
            if not placed:
                # Fallback to simple random placement if non-overlapping placement fails
                circles[i] = [
                    np.random.uniform(r_init, 1 - r_init),
                    np.random.uniform(r_init, 1 - r_init),
                    r_init
                ]

        return circles.flatten()

    # Set a fixed random seed for reproducibility of initial guesses and basinhopping
    np.random.seed(42) 

    initial_guess = generate_initial_guess()

    # Define minimizer_kwargs for the local optimization step within basinhopping
    minimizer_kwargs = {
        "method": "SLSQP", # Sequential Least Squares Programming, suitable for many constraints
        "bounds": bounds,
        "constraints": constraints,
        # Increased maxiter and tightened tolerance for a more thorough search
        "options": {'disp': False, 'maxiter': 2000, 'ftol': 1e-9} 
    }

    # Perform global optimization using basinhopping to escape local minima
    # niter: number of basin hopping iterations (each involves a local minimization)
    # T: temperature for the Metropolis criterion (higher T means more exploration)
    # stepsize: maximum step size for random perturbations
    res_bh = basinhopping(
        objective,
        initial_guess,
        minimizer_kwargs=minimizer_kwargs,
        niter=100, # Number of basin hopping iterations
        T=1.0, 
        stepsize=0.05, 
        seed=42 # Seed for basinhopping's internal random number generator
    )

    # Extract the best parameters found by basinhopping
    best_params = res_bh.x
    best_sum_radii = -res_bh.fun # basinhopping minimizes, so objective is negative sum

    # Validate the final solution from basinhopping
    temp_x, temp_y, temp_r = unpack_params(best_params)
    temp_circles = np.vstack((temp_x, temp_y, temp_r)).T
    validation_check = validate_circles(temp_circles)

    # If the final solution is not valid, it indicates a problem with the optimization.
    # This scenario is less likely with basinhopping as it prioritizes valid local minima.
    if not (validation_check['all_contained'] and validation_check['no_overlaps']):
        print("Warning: Basinhopping converged but produced an invalid final solution. This is unexpected.")
        # Fallback to zero circles or handle as an error
        return np.zeros((n, 3))
    
    # Unpack the best parameters found into the (N, 3) circle array format
    final_x, final_y, final_r = unpack_params(best_params)
    final_circles = np.vstack((final_x, final_y, final_r)).T

    # Internal reporting of performance metrics and validation, as per prompt requirements
    final_sum_radii = np.sum(final_r)
    benchmark_ratio = final_sum_radii / BENCHMARK_SUM_RADII
    final_validation = validate_circles(final_circles)

    print(f"--- Optimization Results for {n} Circles ---")
    print(f"Sum of Radii: {final_sum_radii:.15f}")
    print(f"Benchmark Ratio: {benchmark_ratio:.15f}")
    print(f"Validation:")
    for key, value in final_validation.items():
        print(f"  {key}: {value}")
    print(f"-------------------------------------------")

    return final_circles


# EVOLVE-BLOCK-END
