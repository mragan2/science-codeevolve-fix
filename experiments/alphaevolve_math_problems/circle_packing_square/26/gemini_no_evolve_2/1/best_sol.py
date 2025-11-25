# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize, Bounds


def circle_packing26() -> np.ndarray:
    """
    Generates an optimal arrangement of 26 non-overlapping circles within a unit square,
    maximizing the sum of their radii, using SLSQP for optimization.

    This approach starts with a slightly perturbed 5x5 grid configuration (with one
    extra circle) and uses a gradient-based local optimizer (SLSQP) to find a
    high-quality solution.

    Returns:
        circles: np.array of shape (26,3), where the i-th row (x,y,r) stores the
                 (x,y) coordinates of the i-th circle of radius r.
    """
    n = 26
    # For reproducibility of the initial guess
    np.random.seed(42)

    # --- 1. Initial Guess Generation ---
    # A good initial guess is critical. We start with a slightly perturbed grid.
    # A 5x5 grid fits 25 circles. We'll add a 26th and jiggle them.
    x, y = np.meshgrid(np.linspace(0.1, 0.9, 5), np.linspace(0.1, 0.9, 5))
    initial_positions = np.vstack([x.ravel(), y.ravel()]).T
    
    # Add the 26th circle's position near the center
    initial_positions = np.vstack([initial_positions, [0.5, 0.5]])

    # Add small random perturbations to break symmetry
    initial_positions += np.random.normal(scale=0.01, size=initial_positions.shape)
    
    # Initial radii: small and uniform, allowing the optimizer to grow them
    initial_radii = np.full(n, 0.05)

    # Flatten into the 1D array required by the optimizer
    initial_guess = np.hstack([initial_positions, initial_radii[:, np.newaxis]]).flatten()

    # --- 2. Objective Function ---
    # We want to maximize the sum of radii, so we minimize its negative.
    def objective(params):
        radii = params[2::3]
        return -np.sum(radii)

    # --- 3. Constraints ---
    # Constraints are defined as functions that must evaluate to >= 0.
    # Using vectorized functions is much more efficient than individual lambdas.
    
    # Pre-calculate indices for x, y, r of each circle
    indices = np.arange(n)
    x_idx, y_idx, r_idx = indices * 3, indices * 3 + 1, indices * 3 + 2
    
    # Get all unique pairs of indices (i, j) where i < j for non-overlap checks
    pairs = np.array(np.triu_indices(n, k=1)).T

    def non_overlap_constraints(params):
        # Constraint: (x_i - x_j)^2 + (y_i - y_j)^2 - (r_i + r_j)^2 >= 0
        pos = params.reshape(n, 3)[:, :2]
        radii = params[2::3]
        
        pos_i, pos_j = pos[pairs[:, 0]], pos[pairs[:, 1]]
        radii_i, radii_j = radii[pairs[:, 0]], radii[pairs[:, 1]]
        
        dist_sq = np.sum((pos_i - pos_j)**2, axis=1)
        radii_sum_sq = (radii_i + radii_j)**2
        
        return dist_sq - radii_sum_sq

    cons = [
        # Containment constraints (vectorized for efficiency)
        # x_i - r_i >= 0
        {'type': 'ineq', 'fun': lambda p: p[x_idx] - p[r_idx]},
        # 1 - x_i - r_i >= 0
        {'type': 'ineq', 'fun': lambda p: 1.0 - p[x_idx] - p[r_idx]},
        # y_i - r_i >= 0
        {'type': 'ineq', 'fun': lambda p: p[y_idx] - p[r_idx]},
        # 1 - y_i - r_i >= 0
        {'type': 'ineq', 'fun': lambda p: 1.0 - p[y_idx] - p[r_idx]},
        # Non-overlap constraints
        {'type': 'ineq', 'fun': non_overlap_constraints}
    ]
    
    # --- 4. Bounds ---
    # 0 <= x_i, y_i <= 1
    # 0 <= r_i <= 0.5 (theoretical max for a single circle)
    lower_bounds = np.tile([0, 0, 0], n)
    upper_bounds = np.tile([1, 1, 0.5], n)
    bounds = Bounds(lower_bounds, upper_bounds)

    # --- 5. Optimization ---
    # SLSQP is suitable for this type of non-linear constrained problem.
    result = minimize(
        objective,
        initial_guess,
        method='SLSQP',
        bounds=bounds,
        constraints=cons,
        options={'maxiter': 1000, 'ftol': 1e-9, 'disp': False}
    )

    if result.success:
        final_params = result.x
        return final_params.reshape((n, 3))
    else:
        # If optimization fails, return the last attempted parameters.
        # This may be a partially optimized, but possibly invalid, solution.
        # For this problem, it's better than returning zeros.
        return result.x.reshape((n, 3))


# EVOLVE-BLOCK-END
