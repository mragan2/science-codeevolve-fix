# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize, basinhopping, Bounds, NonlinearConstraint
import time # Import time for eval_time metric

# --- Constants for the problem ---
N_CIRCLES = 26
RANDOM_SEED = 42
MIN_RADIUS_EPS = 1e-6 # Minimum allowed radius to prevent numerical issues or degenerate circles


def circle_packing26() -> np.ndarray:
    """
    Generates an optimal arrangement of exactly 26 non-overlapping circles within a unit square
    [0,1] x [0,1], maximizing the sum of their radii.

    Uses scipy.optimize.basinhopping for global optimization, leveraging SLSQP as the local minimizer,
    to solve the constrained optimization problem. Multiple basinhopping runs are performed to enhance
    global exploration and robustness.

    Returns:
        circles: np.array of shape (26,3), where the i-th row (x,y,r) stores the (x,y) coordinates
                 of the i-th circle and its radius r.
    """
    n = N_CIRCLES
    
    # 1. Define Objective Function and its Jacobian: Minimize the negative sum of radii.
    def objective(params):
        # params is a flattened array: [x1, y1, r1, x2, y2, r2, ..., xn, yn, rn]
        radii = params[2::3] # Extract all radii (every 3rd element starting from index 2)
        return -np.sum(radii)

    def jac_objective(params):
        # The gradient of -sum(r_i) with respect to (x_i, y_i, r_i) is (0, 0, -1).
        # For the flattened array, it's [0, 0, -1, 0, 0, -1, ..., 0, 0, -1].
        grad = np.zeros_like(params)
        grad[2::3] = -1.0 # Set gradient for radii components to -1
        return grad

    # 2. Define Constraints and their Jacobian: Containment and Non-overlap.
    def constraints(params):
        circles_flat = params.reshape(n, 3)
        x_coords, y_coords, r_coords = circles_flat[:, 0], circles_flat[:, 1], circles_flat[:, 2]

        num_constraints = 4 * n + n * (n - 1) // 2
        cons_array = np.empty(num_constraints) # Pre-allocate array for efficiency
        
        k = 0 # Index for filling cons_array

        # Containment constraints:
        # r_i <= x_i <= 1 - r_i  =>  x_i - r_i >= 0  AND  1 - x_i - r_i >= 0
        # r_i <= y_i <= 1 - r_i  =>  y_i - r_i >= 0  AND  1 - y_i - r_i >= 0
        for i in range(n):
            cons_array[k] = x_coords[i] - r_coords[i]          # Left boundary
            k += 1
            cons_array[k] = 1 - x_coords[i] - r_coords[i]      # Right boundary
            k += 1
            cons_array[k] = y_coords[i] - r_coords[i]          # Bottom boundary
            k += 1
            cons_array[k] = 1 - y_coords[i] - r_coords[i]      # Top boundary
            k += 1

        # Non-overlap constraints:
        # (x_i - x_j)^2 + (y_i - y_j)^2 >= (r_i + r_j)^2
        # This translates to: (x_i - x_j)^2 + (y_i - y_j)^2 - (r_i + r_j)^2 >= 0
        for i in range(n):
            for j in range(i + 1, n): # Iterate only upper triangle to avoid duplicates and self-comparison
                dist_sq = (x_coords[i] - x_coords[j])**2 + (y_coords[i] - y_coords[j])**2
                min_dist_sq = (r_coords[i] + r_coords[j])**2
                cons_array[k] = dist_sq - min_dist_sq
                k += 1
        
        return cons_array

    def jac_constraints(params):
        circles_flat = params.reshape(n, 3)
        x_coords, y_coords, r_coords = circles_flat[:, 0], circles_flat[:, 1], circles_flat[:, 2]

        num_constraints = 4 * n + n * (n - 1) // 2
        num_params = 3 * n
        
        jac_matrix = np.zeros((num_constraints, num_params))
        
        k = 0 # Index for rows in jac_matrix (corresponding to constraints)

        # Jacobian for Containment constraints
        for i in range(n):
            # Constraint: x_i - r_i >= 0
            jac_matrix[k, 3*i] = 1.0    # d(x_i - r_i)/dx_i
            jac_matrix[k, 3*i+2] = -1.0 # d(x_i - r_i)/dr_i
            k += 1

            # Constraint: 1 - x_i - r_i >= 0
            jac_matrix[k, 3*i] = -1.0   # d(1 - x_i - r_i)/dx_i
            jac_matrix[k, 3*i+2] = -1.0 # d(1 - x_i - r_i)/dr_i
            k += 1

            # Constraint: y_i - r_i >= 0
            jac_matrix[k, 3*i+1] = 1.0  # d(y_i - r_i)/dy_i
            jac_matrix[k, 3*i+2] = -1.0 # d(y_i - r_i)/dr_i
            k += 1

            # Constraint: 1 - y_i - r_i >= 0
            jac_matrix[k, 3*i+1] = -1.0 # d(1 - y_i - r_i)/dy_i
            jac_matrix[k, 3*i+2] = -1.0 # d(1 - y_i - r_i)/dr_i
            k += 1

        # Jacobian for Non-overlap constraints
        for i in range(n):
            for j in range(i + 1, n):
                dx = x_coords[i] - x_coords[j]
                dy = y_coords[i] - y_coords[j]
                dr = r_coords[i] + r_coords[j]

                # Derivatives w.r.t. circle i (x_i, y_i, r_i)
                jac_matrix[k, 3*i] = 2 * dx      # d(c_ij)/dx_i
                jac_matrix[k, 3*i+1] = 2 * dy    # d(c_ij)/dy_i
                jac_matrix[k, 3*i+2] = -2 * dr   # d(c_ij)/dr_i

                # Derivatives w.r.t. circle j (x_j, y_j, r_j)
                jac_matrix[k, 3*j] = -2 * dx     # d(c_ij)/dx_j
                jac_matrix[k, 3*j+1] = -2 * dy   # d(c_ij)/dy_j
                jac_matrix[k, 3*j+2] = -2 * dr   # d(c_ij)/dr_j
                
                k += 1
                
        return jac_matrix
    
    # 3. Define Bounds for the parameters: Loose general bounds.
    # x, y coordinates must be between 0 and 1.
    # Radii must be positive (e.g., > MIN_RADIUS_EPS) and cannot exceed 0.5 (half the square side).
    lb = np.array([0.0, 0.0, MIN_RADIUS_EPS] * n) # Lower bounds for x, y, r
    ub = np.array([1.0, 1.0, 0.5] * n)  # Upper bounds for x, y, r
    bounds = Bounds(lb, ub)

    # 4. Define the NonlinearConstraint object: All constraints are c(x) >= 0.
    # So, the lower bound for the constraint function output is 0, upper is infinity.
    # Provide the analytical Jacobian for constraints to the NonlinearConstraint object.
    nlc = NonlinearConstraint(constraints, 0, np.inf, jac=jac_constraints)

    # 5. Local Minimizer Options for SLSQP
    local_minimizer_options = {'maxiter': 3000, 'ftol': 1e-8, 'disp': False, 'eps': 1e-6} 

    # minimizer_kwargs will be passed to the local minimizer (SLSQP).
    minimizer_kwargs = {
        "method": "SLSQP",
        "bounds": bounds,
        "constraints": [nlc],
        "options": local_minimizer_options,
        "jac": jac_objective # Pass the analytical Jacobian for the objective function
    }

    # 6. Global Optimization with `basinhopping`: Multiple runs for robustness
    start_time = time.time()
    
    best_sum_radii = -np.inf
    best_circles = np.zeros((n, 3))

    num_basinhopping_runs = 3 # Number of independent basinhopping runs
    bh_niter_per_run = 100    # Number of basin hopping steps for each run (total 300 steps)
    bh_T = 1.5                # "Temperature" parameter, higher allows larger jumps (from Inspiration 3)
    bh_stepsize = 0.05        # Maximum step size for random perturbation

    for i in range(num_basinhopping_runs):
        # Generate a new random initial configuration for each basinhopping run.
        # Starting with very small radii helps avoid immediate massive overlap penalties.
        rng = np.random.default_rng(RANDOM_SEED + i) # Use default_rng for better random number generation
        initial_coords = np.zeros((n, 3))
        initial_coords[:, 0] = rng.uniform(0.1, 0.9, n) # x_coords between 0.1 and 0.9
        initial_coords[:, 1] = rng.uniform(0.1, 0.9, n) # y_coords between 0.1 and 0.9
        initial_coords[:, 2] = rng.uniform(MIN_RADIUS_EPS, 0.02, n) # Small initial radii
        initial_params = initial_coords.flatten()

        print(f"Starting basinhopping run {i+1}/{num_basinhopping_runs}...")
        current_result = basinhopping(
            objective,
            initial_params,
            minimizer_kwargs=minimizer_kwargs,
            niter=bh_niter_per_run,
            T=bh_T,
            stepsize=bh_stepsize,
            seed=RANDOM_SEED + i, # Vary seed for different runs
            disp=False # Suppress basinhopping messages per run
        )
        
        # Reshape the optimized parameters from the current run
        current_optimized_circles = current_result.x.reshape(n, 3)
        current_sum_radii = np.sum(current_optimized_circles[:, 2])

        if current_sum_radii > best_sum_radii:
            best_sum_radii = current_sum_radii
            best_circles = current_optimized_circles
            print(f"Run {i+1}: New best sum_radii found: {best_sum_radii:.10f}")
        else:
            print(f"Run {i+1}: Sum_radii: {current_sum_radii:.10f} (no improvement)")

    end_time = time.time()

    # Post-processing: Ensure radii are strictly positive to avoid numerical issues
    # or degenerate circles that might have converged to r=0.
    best_circles[:, 2] = np.maximum(best_circles[:, 2], MIN_RADIUS_EPS)

    return best_circles


# EVOLVE-BLOCK-END
