# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize


def circle_packing32() -> np.ndarray:
    """
    Places 32 non-overlapping circles in the unit square to maximize the sum of radii.

    This is achieved using a two-stage optimization process with 'SLSQP':
    1.  Stage 1 (Uniform Radii): An optimal arrangement for 32 circles of the *same*
        radius is found. This simplifies the problem to find a good spatial distribution.
        The initial guess is a slightly perturbed 6x6 grid (with corners removed) to
        break symmetry and avoid trivial local minima.
    2.  Stage 2 (Variable Radii): The result from Stage 1 (optimized centers and the
        uniform radius) is used as a high-quality initial guess for the full problem,
        where each circle's radius can vary independently.

    This staged approach guides the optimizer to a more promising region of the
    solution space, significantly improving the final sum of radii.
    """
    n_circles = 32

    # --- Initial Guess Generation ---
    coords = []
    for i in range(6):
        for j in range(6):
            if (i == 0 or i == 5) and (j == 0 or j == 5):
                continue
            coords.append([(i + 0.5) / 6.0, (j + 0.5) / 6.0])
    initial_centers = np.array(coords)

    # Add small deterministic noise to break symmetry, helping escape local minima
    rng = np.random.default_rng(42)
    noise = rng.uniform(-0.01, 0.01, initial_centers.shape)
    initial_centers += noise

    # --- Stage 1: Optimization with Uniform Radii ---
    def stage1_objective(params):
        # params are [x0, y0, ..., x31, y31, r]
        return -params[-1]  # Minimize -r to maximize r

    def stage1_constraints(params):
        centers = params[:-1].reshape((n_circles, 2))
        r = params[-1]
        
        # Non-overlap constraints: dist_sq >= (2r)^2
        dist_sq = np.sum((centers[:, np.newaxis, :] - centers[np.newaxis, :, :])**2, axis=-1)
        indices = np.triu_indices(n_circles, k=1)
        overlap_cons = dist_sq[indices] - (2 * r)**2
        
        # Boundary constraints
        boundary_cons = np.concatenate([
            centers[:, 0] - r, 1.0 - centers[:, 0] - r,
            centers[:, 1] - r, 1.0 - centers[:, 1] - r
        ])
        return np.concatenate([overlap_cons, boundary_cons])

    x0_stage1 = np.append(initial_centers.flatten(), 1.0 / 13.0)
    bounds_stage1 = [(0.0, 1.0)] * (2 * n_circles) + [(0.0, 0.5)]
    cons_stage1 = {'type': 'ineq', 'fun': stage1_constraints}

    result_stage1 = minimize(
        stage1_objective, x0_stage1, method='SLSQP', bounds=bounds_stage1,
        constraints=cons_stage1, options={'maxiter': 2000, 'ftol': 1e-9}
    )
    
    # --- Stage 2: Full Optimization with Variable Radii ---
    # Use Stage 1 result as a high-quality initial guess for Stage 2.
    # Instead of a uniform radius, we create a non-uniform initial guess for
    # the radii based on a geometric heuristic: a circle's available space is
    # determined by its distance to its nearest neighbors and the container walls.
    # This provides a much better starting point for the local optimizer.
    from scipy.spatial import distance_matrix

    optimized_centers = result_stage1.x[:-1].reshape((n_circles, 2))
    uniform_radius = result_stage1.x[-1]

    # Heuristic for initial radii in Stage 2:
    # 1. For each circle, find the average distance to its k nearest neighbors.
    k_neighbors = 6  # Inspired by hexagonal packing
    dists = distance_matrix(optimized_centers, optimized_centers)
    np.fill_diagonal(dists, np.inf)  # Ignore distance to self
    dists.sort(axis=1)
    avg_neighbor_dist = np.mean(dists[:, :k_neighbors], axis=1)

    # 2. For each circle, find the distance to the nearest wall.
    dist_to_wall_x = np.minimum(optimized_centers[:, 0], 1.0 - optimized_centers[:, 0])
    dist_to_wall_y = np.minimum(optimized_centers[:, 1], 1.0 - optimized_centers[:, 1])
    dist_to_wall = np.minimum(dist_to_wall_x, dist_to_wall_y)

    # 3. The initial radius estimate is limited by half the neighbor distance and the wall distance.
    estimated_radii = np.minimum(avg_neighbor_dist / 2.0, dist_to_wall)

    # 4. Scale the estimates to preserve the total sum of radii from Stage 1.
    # This ensures we start from a point of similar "quality" but with better direction.
    total_radius_stage1 = n_circles * uniform_radius
    if np.sum(estimated_radii) > 1e-9: # Avoid division by zero
        scaling_factor = total_radius_stage1 / np.sum(estimated_radii)
        initial_radii_stage2 = estimated_radii * scaling_factor
    else: # Fallback in an unlikely edge case
        initial_radii_stage2 = np.full(n_circles, uniform_radius)

    x0_stage2 = np.hstack((optimized_centers, initial_radii_stage2[:, np.newaxis])).flatten()

    def stage2_objective(params):
        return -np.sum(params[2::3])

    def stage2_constraints(params):
        circles = params.reshape((n_circles, 3))
        centers, radii = circles[:, :2], circles[:, 2]
        
        # Non-overlap constraints
        dist_sq = np.sum((centers[:, np.newaxis, :] - centers[np.newaxis, :, :])**2, axis=-1)
        radii_sum_sq = (radii[:, np.newaxis] + radii[np.newaxis, :])**2
        indices = np.triu_indices(n_circles, k=1)
        overlap_cons = dist_sq[indices] - radii_sum_sq[indices]
        
        # Boundary constraints
        boundary_cons = np.concatenate([
            centers[:, 0] - radii, 1.0 - centers[:, 0] - radii,
            centers[:, 1] - radii, 1.0 - centers[:, 1] - radii
        ])
        return np.concatenate([overlap_cons, boundary_cons])

    bounds_stage2 = []
    for _ in range(n_circles):
        bounds_stage2.extend([(0.0, 1.0), (0.0, 1.0), (0.0, 0.5)])
    
    cons_stage2 = {'type': 'ineq', 'fun': stage2_constraints}
    
    result_stage2 = minimize(
        stage2_objective, x0_stage2, method='SLSQP', bounds=bounds_stage2,
        constraints=cons_stage2, options={'maxiter': 5000, 'ftol': 1e-11, 'disp': False}
    )

    # --- Format and Return the Result ---
    final_circles = result_stage2.x.reshape((n_circles, 3))
    return final_circles


# EVOLVE-BLOCK-END
