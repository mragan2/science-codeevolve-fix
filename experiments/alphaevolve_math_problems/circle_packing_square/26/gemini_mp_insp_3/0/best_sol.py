# EVOLVE-BLOCK-START
import numpy as np
import time # Import time for eval_time metric
from scipy.optimize import minimize, basinhopping, Bounds, NonlinearConstraint


def circle_packing26() -> np.ndarray:
    """
    Generates an optimal arrangement of exactly 26 non-overlapping circles within a unit square
    [0,1] x [0,1], maximizing the sum of their radii.

    Uses scipy.optimize.basinhopping for global optimization, leveraging SLSQP as the local minimizer,
    to solve the constrained optimization problem.

    Returns:
        circles: np.array of shape (26,3), where the i-th row (x,y,r) stores the (x,y) coordinates
                 of the i-th circle and its radius r.
    """
    n = 26
    
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

        # Number of constraints: 4 per circle (containment) + N*(N-1)/2 for non-overlap
        num_constraints = 4 * n + n * (n - 1) // 2
        cons_array = np.empty(num_constraints) # Pre-allocate array for efficiency
        
        k = 0 # Index for filling cons_array

        # Containment constraints:
        # x_i - r_i >= 0  AND  1 - x_i - r_i >= 0
        # y_i - r_i >= 0  AND  1 - y_i - r_i >= 0
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
        # (x_i - x_j)^2 + (y_i - y_j)^2 - (r_i + r_j)^2 >= 0
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
    
    # 3. Initial Guess: A more randomized approach suitable for global optimization.
    np.random.seed(42) # Ensure reproducibility

    initial_coords = np.zeros((n, 3))
    
    # Start with very small radii, allowing them to grow during optimization.
    # This helps avoid immediate overlap constraints and gives the optimizer more room to explore.
    initial_r = 0.01 

    # Place centers randomly across the entire unit square.
    # The bounds and nonlinear constraints will ensure containment and non-overlap.
    initial_coords[:, 0] = np.random.rand(n) # x_coords between 0 and 1
    initial_coords[:, 1] = np.random.rand(n) # y_coords between 0 and 1
    initial_coords[:, 2] = initial_r         # radii

    initial_params = initial_coords.flatten()

    # 4. Define Bounds for the parameters: Loose general bounds.
    # The `NonlinearConstraint` handles the tighter, `r`-dependent containment.
    # x, y coordinates must be between 0 and 1.
    # Radii must be positive (e.g., > 1e-6) and cannot exceed 0.5 (half the square side).
    lb = np.array([0.0, 0.0, 1e-6] * n) # Lower bounds for x, y, r
    ub = np.array([1.0, 1.0, 0.5] * n)  # Upper bounds for x, y, r
    bounds = Bounds(lb, ub)

    # 5. Define the NonlinearConstraint object: All constraints are c(x) >= 0.
    # So, the lower bound for the constraint function output is 0, upper is infinity.
    # Provide the analytical Jacobian for constraints to the NonlinearConstraint object.
    nlc = NonlinearConstraint(constraints, 0, np.inf, jac=jac_constraints)

    # 6. Call the optimizer: `basinhopping` for global optimization.
    # `options`: Parameters for the local minimizer (SLSQP in this case).
    # `maxiter`: Max iterations for the local minimizer.
    # `ftol`: Function tolerance for local minimizer termination.
    # `disp`: Set to False to suppress optimization messages.
    # `eps`: Step size for finite difference approximations of gradients (for constraints if no jacobian provided).
    local_minimizer_options = {'maxiter': 3000, 'ftol': 1e-8, 'disp': False, 'eps': 1e-6} 

    # minimizer_kwargs will be passed to the local minimizer (SLSQP).
    minimizer_kwargs = {
        "method": "SLSQP",
        "bounds": bounds,
        "constraints": [nlc],
        "options": local_minimizer_options,
        "jac": jac_objective # Pass the analytical Jacobian for the objective function
    }

    # Basinhopping parameters:
    # niter: Number of hopping iterations. More iterations mean more exploration.
    # T: Temperature parameter. Higher T allows for accepting worse solutions, promoting wider exploration.
    # stepsize: The maximum step size for the random perturbation.
    bh_options = {
        'niter': 200,    # Number of basin-hopping iterations (increased for better exploration)
        'T': 1.0,        # Temperature parameter for the Metropolis acceptance criterion
        'stepsize': 0.05, # Maximum step size for random perturbations
        'disp': False    # Suppress basinhopping messages
    }

    start_time = time.time()
    result = basinhopping(
        objective,
        initial_params,
        minimizer_kwargs=minimizer_kwargs,
        **bh_options
    )
    end_time = time.time()

    if not result.success:
        # If optimization fails to converge, print a warning.
        # The function will still return the best state found by the optimizer.
        print(f"Optimization failed: {result.message}")

    # Reshape the optimized parameters back into (n, 3) format (x, y, r)
    optimized_circles = result.x.reshape(n, 3)
    
    # Post-processing: Ensure radii are strictly positive to avoid numerical issues
    # or degenerate circles that might have converged to r=0.
    optimized_circles[:, 2] = np.maximum(optimized_circles[:, 2], 1e-6)

    return optimized_circles


# EVOLVE-BLOCK-END
