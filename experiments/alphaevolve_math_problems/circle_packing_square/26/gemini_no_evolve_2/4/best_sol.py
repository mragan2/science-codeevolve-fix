# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
from scipy.spatial.distance import pdist, squareform


def circle_packing26() -> np.ndarray:
    """
    Places 26 non-overlapping circles in the unit square in order to maximize the sum of radii.
    This is achieved by formulating it as a constrained optimization problem and solving
    it with Sequential Least Squares Programming (SLSQP).

    Returns:
        circles: np.array of shape (26,3), where the i-th row (x,y,r) stores the
                 (x,y) coordinates of the i-th circle of radius r.
    """
    N_CIRCLES = 26
    # Use a fixed seed for reproducibility of the initial guess
    np.random.seed(42)

    def objective(params: np.ndarray) -> float:
        """The objective function to minimize is the negative sum of radii."""
        radii = params[2::3]
        return -np.sum(radii)

    def constraints(params: np.ndarray) -> np.ndarray:
        """
        Constraint function. All values must be non-negative.
        Combines containment and non-overlap constraints.
        """
        circles = params.reshape((N_CIRCLES, 3))
        x, y, r = circles[:, 0], circles[:, 1], circles[:, 2]

        # 1. Containment constraints: ri <= xi, xi <= 1-ri, ri <= yi, yi <= 1-ri
        containment_c = np.concatenate([
            x - r,      # x_i - r_i >= 0
            1 - x - r,  # 1 - x_i - r_i >= 0
            y - r,      # y_i - r_i >= 0
            1 - y - r   # 1 - y_i - r_i >= 0
        ])

        # 2. Non-overlap constraints: (xi-xj)^2 + (yi-yj)^2 >= (ri+rj)^2
        coords = circles[:, :2]
        # Pairwise squared Euclidean distances between circle centers
        dist_sq = squareform(pdist(coords, 'sqeuclidean'))
        
        # Pairwise sum of radii, squared
        radii_sum = r[:, np.newaxis] + r
        radii_sum_sq = radii_sum**2

        # Get upper triangle indices to avoid duplicate constraints and self-comparison
        iu = np.triu_indices(N_CIRCLES, k=1)
        non_overlap_c = dist_sq[iu] - radii_sum_sq[iu]

        return np.concatenate([containment_c, non_overlap_c])

    # --- Initial Guess ---
    # Start with a jittered 5x6 grid layout.
    nx, ny = 5, 6
    x_centers = np.linspace(0.1, 0.9, nx)
    y_centers = np.linspace(0.1, 0.9, ny)
    grid_x, grid_y = np.meshgrid(x_centers, y_centers)
    
    initial_params = np.zeros((N_CIRCLES, 3))
    coords_flat = np.vstack([grid_x.ravel(), grid_y.ravel()]).T
    initial_params[:, :2] = coords_flat[:N_CIRCLES]
    
    # Add small random jitter to break symmetry and aid optimization
    initial_params[:, :2] += np.random.uniform(-0.01, 0.01, (N_CIRCLES, 2))
    
    # A reasonable initial radius based on grid spacing to avoid initial overlap
    min_dist = min(x_centers[1] - x_centers[0], y_centers[1] - y_centers[0])
    initial_params[:, 2] = min_dist / 2.1
    
    x0 = initial_params.flatten()

    # --- Bounds ---
    # (x_min, x_max), (y_min, y_max), (r_min, r_max) for each circle
    bounds = []
    for _ in range(N_CIRCLES):
        bounds.append((0.0, 1.0))  # x_i
        bounds.append((0.0, 1.0))  # y_i
        bounds.append((1e-6, 0.5)) # r_i (must be positive)
        
    # --- Optimization ---
    cons = ({'type': 'ineq', 'fun': constraints})
    
    # SLSQP is suitable for this type of constrained non-linear problem.
    # We increase maxiter and set a tighter tolerance for a higher quality solution.
    result = minimize(objective, x0, method='SLSQP', bounds=bounds, constraints=cons,
                      options={'maxiter': 2000, 'disp': False, 'ftol': 1e-9})

    if result.success:
        final_circles = result.x.reshape((N_CIRCLES, 3))
    else:
        # If optimization fails, return the last attempted solution.
        # This is better than returning zeros, as it might be partially optimized.
        final_circles = result.x.reshape((N_CIRCLES, 3))
        
    return final_circles


# EVOLVE-BLOCK-END
