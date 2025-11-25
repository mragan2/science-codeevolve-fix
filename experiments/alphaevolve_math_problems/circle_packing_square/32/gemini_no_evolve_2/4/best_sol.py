# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
from scipy.spatial.distance import pdist, squareform


def circle_packing32() -> np.ndarray:
    """
    Generates an arrangement of 32 non-overlapping circles in a unit square,
    aiming to maximize the sum of their radii using numerical optimization (SLSQP).
    Employs a multi-start strategy with refined initial guesses and increased
    iterations for the local optimizer to escape local minima and improve results.

    Returns:
        circles: np.array of shape (32,3), where the i-th row (x,y,r) stores the 
                 (x,y) coordinates of the i-th circle of radius r.
    """
    n_circles = 32
    # Use a fixed random seed for the overall multi-start process for reproducibility.
    np.random.seed(42)

    best_sum_radii = -np.inf
    best_circles = None

    n_restarts = 15 # Increased number of independent optimization runs to explore more of the search space
    maxiter_per_run = 4000 # Increased maxiter for each SLSQP run for better convergence

    # 2. Objective Function to Minimize (defined once, used in loop)
    # The goal is to MAXIMIZE sum of radii, so we MINIMIZE the negative sum of radii.
    def objective(params):
        radii = params[2::3]
        return -np.sum(radii)

    # 3. Constraint Function (defined once, used in loop)
    # All constraints are formulated as g(x) >= 0.
    def constraints(params):
        p = params.reshape((n_circles, 3))
        coords, radii = p[:, :2], p[:, 2]
        
        # a) Containment constraints: circles must be fully inside the [0,1]x[0,1] square.
        # r <= x <= 1-r  and  r <= y <= 1-r
        containment = np.concatenate([
            coords[:, 0] - radii,          # x - r >= 0
            1.0 - coords[:, 0] - radii,    # 1 - x - r >= 0
            coords[:, 1] - radii,          # y - r >= 0
            1.0 - coords[:, 1] - radii     # 1 - y - r >= 0
        ])
        
        # b) Non-overlap constraints: distance between centers must be >= sum of radii.
        # sqrt((xi-xj)^2 + (yi-yj)^2) >= ri + rj
        # To avoid sqrt, we use the squared form: (xi-xj)^2 + (yi-yj)^2 - (ri+rj)^2 >= 0
        
        dist_sq_matrix = squareform(pdist(coords, 'sqeuclidean'))
        radii_sum = radii[:, np.newaxis] + radii
        radii_sum_sq_matrix = radii_sum**2
        
        iu = np.triu_indices(n_circles, k=1)
        non_overlap = dist_sq_matrix[iu] - radii_sum_sq_matrix[iu]
        
        return np.concatenate([containment, non_overlap])

    # 4. Define Bounds for each variable (defined once, used in loop)
    # 0 <= x, y <= 1 and 0 <= r <= 0.5 (a single circle cannot exceed this radius).
    bounds = []
    for _ in range(n_circles):
        bounds.extend([(0, 1), (0, 1), (0, 0.5)])

    # Multi-start loop
    for i in range(n_restarts):
        # Generate a new initial guess for each restart
        # Use a different seed for each restart's jitter to ensure variety
        rng = np.random.default_rng(42 + i) # Separate random generator for each run's jitter

        # 1. Initial Guess Generation
        # Use a 4x8 grid, which perfectly fits 32 circles, for better initial spacing.
        grid_x, grid_y = 4, 8
        x_coords, y_coords = np.meshgrid(
            # Use centered linspace for initial grid points for a more balanced start
            np.linspace(0.5 / grid_x, 1 - 0.5 / grid_x, grid_x),
            np.linspace(0.5 / grid_y, 1 - 0.5 / grid_y, grid_y)
        )
        initial_centers = np.vstack([x_coords.ravel(), y_coords.ravel()]).T
        
        # Add a random perturbation to break symmetry and avoid local minima.
        # Slightly increased jitter for more exploration.
        jitter = 0.01 
        initial_centers += rng.uniform(-jitter, jitter, initial_centers.shape)
        
        # Initialize radii with a base value, varying based on distance from center, plus random jitter.
        # Circles closer to the boundary can often be larger.
        dist_from_center = np.linalg.norm(initial_centers - 0.5, axis=1)
        if dist_from_center.max() > dist_from_center.min():
            norm_dist = (dist_from_center - dist_from_center.min()) / (dist_from_center.max() - dist_from_center.min())
        else:
            norm_dist = np.zeros_like(dist_from_center)
        
        # Target radii range for initial guess. Max radius for a 4x8 grid is approx 0.055.
        target_min_r, target_max_r = 0.03, 0.055
        initial_radii = target_min_r + (target_max_r - target_min_r) * norm_dist

        # Add slight random variation to radii for more diversity in initial guesses
        radii_jitter_amount = 0.005
        initial_radii += rng.uniform(-radii_jitter_amount, radii_jitter_amount, n_circles)
        
        # Ensure initial radii are positive and within reasonable bounds for feasibility
        initial_radii = np.clip(initial_radii, 0.01, 0.055) 

        # The optimizer works with a flat 1D array of parameters: [x0, y0, r0, x1, y1, r1, ...].
        x0 = np.hstack([initial_centers, initial_radii.reshape(-1, 1)]).ravel()

        # 5. Execute the Optimization
        cons = [{'type': 'ineq', 'fun': constraints}]
        
        result = minimize(
            objective,
            x0,
            method='SLSQP',
            bounds=bounds,
            constraints=cons,
            options={'maxiter': maxiter_per_run, 'disp': False, 'ftol': 1e-10}
        )

        # Update best solution if current run is successful and better
        if result.success:
            current_sum_radii = -result.fun
            if current_sum_radii > best_sum_radii:
                best_sum_radii = current_sum_radii
                best_circles = result.x.reshape((n_circles, 3))
                # print(f"Restart {i+1}: New best sum_radii = {best_sum_radii:.6f}") # For debugging
        # else:
            # print(f"Restart {i+1}: Optimization did not converge. Reason: {result.message}") # For debugging

    # Graceful handling if no run successfully converged
    if best_circles is None:
        print("Warning: No successful optimization run found. Returning a default initial guess.")
        # Fallback to a simple, safe initial grid with small uniform radii
        grid_x, grid_y = 4, 8
        x_coords, y_coords = np.meshgrid(
            np.linspace(0.5 / grid_x, 1 - 0.5 / grid_x, grid_x),
            np.linspace(0.5 / grid_y, 1 - 0.5 / grid_y, grid_y)
        )
        fallback_centers = np.vstack([x_coords.ravel(), y_coords.ravel()]).T
        fallback_radii = np.full(n_circles, 0.04) # A safe small uniform radius
        best_circles = np.hstack([fallback_centers, fallback_radii.reshape(-1, 1)]).reshape((n_circles, 3))

    return best_circles


# EVOLVE-BLOCK-END
