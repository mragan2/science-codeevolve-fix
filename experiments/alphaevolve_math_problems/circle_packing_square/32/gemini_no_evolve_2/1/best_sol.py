# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
from scipy.spatial.distance import pdist, squareform


def circle_packing32() -> np.ndarray:
    """
    Places 32 non-overlapping circles in the unit square in order to maximize the sum of radii.
    This is achieved by formulating it as a constrained optimization problem and solving it
    using the Sequential Least Squares Programming (SLSQP) method.

    The variables are the centers (x_i, y_i) and radii (r_i) of the circles.
    The objective is to maximize the sum of r_i.
    The constraints are:
    1. Each circle is fully inside the unit square.
    2. No two circles overlap.

    Returns:
        circles: np.array of shape (32,3), where the i-th row (x,y,r) stores the
                 (x,y) coordinates of the i-th circle of radius r.
    """
    n = 32

    def objective(params, n_circles):
        """The objective function to be minimized (negative sum of radii)."""
        radii = params[2::3]
        return -np.sum(radii)

    def constraints_func(params, n_circles):
        """
        Defines the inequality constraints for the optimizer.
        All constraints are of the form C(x) >= 0.
        """
        circles = params.reshape((n_circles, 3))
        x, y, r = circles[:, 0], circles[:, 1], circles[:, 2]

        # 1. Boundary constraints: 4 constraints per circle.
        #    ri <= xi <= 1-ri  =>  xi - ri >= 0  AND  1 - xi - ri >= 0
        #    ri <= yi <= 1-ri  =>  yi - ri >= 0  AND  1 - yi - ri >= 0
        boundary_constraints = np.concatenate([
            x - r,
            1 - x - r,
            y - r,
            1 - y - r
        ])

        # 2. Non-overlap constraints: n*(n-1)/2 constraints.
        #    sqrt((xi-xj)^2 + (yi-yj)^2) >= ri + rj
        #    Using squared form for numerical stability: (xi-xj)^2 + (yi-yj)^2 - (ri+rj)^2 >= 0
        pos = circles[:, :2]
        
        # Efficiently calculate squared distances between all pairs of centers
        sq_dist_matrix = squareform(pdist(pos, 'sqeuclidean'))
        
        # Calculate squared sum of radii for all pairs
        radii_sum = r[:, np.newaxis] + r[np.newaxis, :]
        sq_radii_sum_matrix = radii_sum**2

        # We only need the upper triangle of the matrix to get unique pairs (i > j)
        iu = np.triu_indices(n_circles, k=1)
        overlap_constraints = sq_dist_matrix[iu] - sq_radii_sum_matrix[iu]
        
        return np.concatenate([boundary_constraints, overlap_constraints])

    # --- Bounds for Variables ---
    # 0 <= x_i <= 1,  0 <= y_i <= 1,  0 <= r_i <= 0.5
    bounds = []
    for i in range(n):
        bounds.append((0, 1))    # x_i
        bounds.append((0, 1))    # y_i
        # Adjusted upper bound for radii: 0.15 is a more realistic max for 32 circles,
        # helping the optimizer search in a more relevant range. A minimum radius of 0.001
        # is set to avoid degenerate zero-radius circles and potential numerical issues.
        bounds.append((0.001, 0.15))  # r_i

    # --- Optimization ---
    cons = [{'type': 'ineq', 'fun': constraints_func, 'args': (n,)}]

    best_sum_radii = -np.inf
    best_circles = None
    last_res_x = None # Stores the result from the last run as a fallback

    # Implement multi-start optimization to escape local minima
    n_restarts = 50 # Increased restarts for better exploration of the solution space, crucial for complex landscapes
    base_seed = 42 # Base seed for reproducibility of the multi-start process

    for i in range(n_restarts):
        current_seed = base_seed + i
        np.random.seed(current_seed) # Set seed for initial guess generation

        # Generate initial guess for the current restart using varied strategies
        if i % 3 == 0: # Strategy 1: Perturbed grid layout
            # Dynamically adjust grid size to be slightly larger than sqrt(n) for flexibility
            grid_size = int(np.ceil(np.sqrt(n)))
            
            # Create grid centers
            x_coords, y_coords = np.meshgrid(
                np.linspace(1 / (2 * grid_size), 1 - 1 / (2 * grid_size), grid_size),
                np.linspace(1 / (2 * grid_size), 1 - 1 / (2 * grid_size), grid_size)
            )
            initial_positions = np.stack([x_coords.ravel(), y_coords.ravel()], axis=1)[:n]
            
            # Add a small random perturbation to break symmetry
            perturbation = (np.random.rand(n, 2) - 0.5) / (grid_size * 10) # Smaller perturbation
            initial_positions += perturbation
            
            # Start with small, equal radii
            initial_radii = np.full(n, 0.01)
            
        elif i % 3 == 1: # Strategy 2: Random uniform placement with small, varying radii
            initial_positions = np.random.rand(n, 2) # Random positions within unit square
            initial_radii = np.random.uniform(0.005, 0.02, n) # Varying small radii
            
        else: # Strategy 3: Anchor circles (corners/center) with slightly larger radii, rest random
            initial_positions = np.random.rand(n, 2)
            initial_radii = np.random.uniform(0.005, 0.02, n)
            
            # Place a few "anchor" circles at strategic positions with larger initial radii
            num_anchors = min(n, 4) # Use up to 4 corners
            anchor_positions = np.array([[0.1, 0.1], [0.9, 0.1], [0.1, 0.9], [0.9, 0.9]])
            anchor_radii = np.full(num_anchors, 0.05) # Larger radii for anchors

            initial_positions[:num_anchors] = anchor_positions[:num_anchors]
            initial_radii[:num_anchors] = anchor_radii[:num_anchors]

            # Optionally, add a central anchor if enough circles are available
            if n > 4:
                initial_positions[num_anchors] = np.array([0.5, 0.5])
                initial_radii[num_anchors] = 0.05
            
        # Flatten into the 1D array required by the optimizer
        x0 = np.hstack([initial_positions, initial_radii.reshape(-1, 1)]).ravel()

        # Run the optimizer for the current initial guess
        # Increased maxiter and tighter ftol to allow for deeper local search and more precise convergence
        res = minimize(objective, x0, args=(n,), method='SLSQP', bounds=bounds, constraints=cons,
                       options={'maxiter': 5000, 'ftol': 1e-9, 'disp': False})

        current_sum_radii = -res.fun # Objective function returns negative sum_radii

        # Update best result if current run found a better sum of radii
        if current_sum_radii > best_sum_radii:
            best_sum_radii = current_sum_radii
            best_circles = res.x.reshape((n, 3))
        
        last_res_x = res.x # Keep the result from the last attempt as a basic fallback

        if not res.success:
            # print(f"Warning: Optimizer restart {i+1} did not converge. Reason: {res.message}")
            pass # Suppress warning for each non-converged run, only care about the best result

    if best_circles is None:
        # Fallback if no valid circle arrangement was found after all restarts (highly unlikely with proper setup)
        print("Warning: No valid circle arrangement found after multiple restarts. Returning result from last attempt.")
        best_circles = last_res_x.reshape((n, 3))

    return best_circles


# EVOLVE-BLOCK-END
