# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
from scipy.spatial.distance import pdist, squareform


def circle_packing32() -> np.ndarray:
    """
    Generates an optimal arrangement of 32 non-overlapping circles in a unit square
    to maximize the sum of their radii, using SLSQP optimization.

    Returns:
        np.ndarray: An array of shape (32, 3) where each row is [x, y, r].
    """
    n = 32
    # Set a seed for reproducibility of the initial guess perturbation
    np.random.seed(42)

    # 1. Define the objective function (to be minimized)
    def objective(params: np.ndarray) -> float:
        """
        The objective is to maximize the sum of radii. Scipy's minimize function
        minimizes a scalar function, so we minimize the negative sum of radii.
        `params` is a flattened 1D array of [x0, y0, r0, x1, y1, r1, ...].
        """
        radii = params[2::3]
        return -np.sum(radii)

    # 2. Define the constraint functions
    def constraints(params: np.ndarray) -> np.ndarray:
        """
        Returns a 1D array where each element must be non-negative (>= 0)
        for the solution to be feasible.
        """
        circles = params.reshape((n, 3))
        x, y, r = circles[:, 0], circles[:, 1], circles[:, 2]

        # Constraint 1: Boundary constraints (4*n constraints)
        # Circles must be fully inside the [0,1] x [0,1] square.
        # ri <= xi <= 1-ri  =>  xi - ri >= 0  AND  1 - xi - ri >= 0
        # ri <= yi <= 1-ri  =>  yi - ri >= 0  AND  1 - yi - ri >= 0
        c_boundary = np.concatenate([
            x - r,
            1 - x - r,
            y - r,
            1 - y - r
        ])

        # Constraint 2: Non-overlap constraints (n*(n-1)/2 constraints)
        # The distance between centers must be >= sum of radii.
        # sqrt((xi-xj)^2 + (yi-yj)^2) >= ri + rj
        # (xi-xj)^2 + (yi-yj)^2 - (ri+rj)^2 >= 0
        pos = circles[:, :2]
        # Use 'sqeuclidean' for performance as it avoids the sqrt
        dist_sq_matrix = squareform(pdist(pos, 'sqeuclidean'))

        radii_sum = np.add.outer(r, r)
        radii_sum_sq_matrix = radii_sum**2

        # We only need the upper triangle of the matrix to get unique pairs
        indices = np.triu_indices(n, k=1)
        c_overlap = dist_sq_matrix[indices] - radii_sum_sq_matrix[indices]

        return np.concatenate([c_boundary, c_overlap])

    # 3. Create an initial guess (x0)
    # Start with a 6x6 grid, removing the 4 corners, giving 32 positions.
    # This provides a good, symmetric starting distribution.
    coords = []
    grid_size = 6
    spacing = 1.0 / grid_size
    for i in range(grid_size):
        for j in range(grid_size):
            # Exclude the four corner positions to get 32 points
            if not ((i in [0, grid_size - 1]) and (j in [0, grid_size - 1])):
                # Add a small random perturbation to break perfect symmetry
                px = spacing * (i + 0.5) + np.random.uniform(-0.01, 0.01)
                py = spacing * (j + 0.5) + np.random.uniform(-0.01, 0.01)
                coords.append([px, py])

    initial_coords = np.array(coords)
    # Start with small, uniform radii to ensure the initial guess is feasible
    initial_radii = np.full((n, 1), 0.01)
    x0 = np.hstack([initial_coords, initial_radii]).flatten()

    # 4. Define bounds for each variable (x, y, r)
    # 0 <= x, y <= 1
    # 0 < r <= 0.5 (max possible radius for one circle)
    bounds = []
    for _ in range(n):
        bounds.extend([(0.0, 1.0), (0.0, 1.0), (1e-6, 0.5)])

    # 5. Set up the constraints for the optimizer
    cons = {'type': 'ineq', 'fun': constraints}

    # 6. Run the optimization
    # SLSQP is suitable for constrained optimization.
    # We increase maxiter for better convergence on this complex problem.
    options = {'maxiter': 1000, 'ftol': 1e-9, 'disp': False}
    result = minimize(objective, x0, method='SLSQP', bounds=bounds, constraints=cons, options=options)

    if result.success:
        final_circles = result.x.reshape((n, 3))
    else:
        # If optimization fails, return the initial guess reshaped.
        # This is better than returning zeros as it's a valid (if suboptimal) packing.
        print(f"Optimization failed: {result.message}")
        final_circles = x0.reshape((n, 3))

    return final_circles


# EVOLVE-BLOCK-END
