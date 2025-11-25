# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
from scipy.spatial.distance import pdist


def circle_packing32() -> np.ndarray:
    """
    Generates an optimized arrangement of 32 non-overlapping circles within a unit square.
    This version uses a two-stage optimization process to achieve a better result:
    1.  Find an optimal packing for 32 circles of EQUAL radius to establish a good
        spatial distribution.
    2.  Use this dense, uniform packing as the starting point for the main optimization
        problem of maximizing the SUM of VARIABLE radii.

    This approach helps the optimizer avoid poor local minima and find a superior solution.
    It also uses a squared-distance formulation for the non-overlap constraint for
    better numerical stability.

    Returns:
        np.ndarray: An array of shape (32, 3) where each row represents a circle's
                    (x, y, r) coordinates and radius.
    """
    n = 32

    # --- STAGE 1: Find a good spatial arrangement with equal radii ---
    # This pre-optimization step finds a dense packing of same-sized circles.
    # The variables are [x0, y0, ..., xn-1, yn-1, r_uniform].
    def objective_stage1(params):
        return -params[-1]  # Maximize the uniform radius r

    def constraints_stage1(params):
        positions = params[:-1].reshape((n, 2))
        r = params[-1]
        
        c_contain = np.concatenate([
            positions[:, 0] - r, 1 - positions[:, 0] - r,
            positions[:, 1] - r, 1 - positions[:, 1] - r
        ])
        
        # Use squared distances to avoid sqrt, which is better for the optimizer.
        # dist_sq >= (2r)^2  =>  dist_sq - 4r^2 >= 0
        if n > 1:
            dist_sq = pdist(positions, 'sqeuclidean')
            c_overlap = dist_sq - 4.0 * r**2
            return np.concatenate([c_contain, c_overlap])
        return c_contain

    # Initial guess for Stage 1: a phyllotaxis (sunflower) pattern mapped to a square.
    # This provides a more uniform and dense initial distribution than a grid,
    # helping the optimizer find a better initial packing in Stage 1.
    golden_angle = np.pi * (3. - np.sqrt(5.))
    indices = np.arange(n)
    theta = indices * golden_angle
    r_disk = np.sqrt(indices / float(n)) # Radius for the disk to be mapped

    # Convert polar coordinates on a disk to cartesian
    x_disk = r_disk * np.cos(theta)
    y_disk = r_disk * np.sin(theta)

    # Use a concentric mapping to transform points from a disk to a square [-1,1]x[-1,1].
    # This ensures points are distributed evenly across the entire square area.
    # Reference: Shirley & Chiu, "A Low Distortion Map Between Disk and Square"
    x_square = np.zeros_like(x_disk)
    y_square = np.zeros_like(y_disk)
    
    # Identify points in the 'east-west' sectors vs 'north-south' sectors
    mask_ew = np.abs(x_disk) > np.abs(y_disk)
    mask_ns = ~mask_ew
    
    # Map points in east-west sectors
    u_ew, v_ew = x_disk[mask_ew], y_disk[mask_ew]
    x_square[mask_ew] = np.sign(u_ew) * r_disk[mask_ew]
    y_square[mask_ew] = np.sign(u_ew) * r_disk[mask_ew] * (v_ew / u_ew)
    
    # Map points in north-south sectors, handling division by zero for the origin
    u_ns, v_ns = x_disk[mask_ns], y_disk[mask_ns]
    with np.errstate(divide='ignore', invalid='ignore'):
        ratio = u_ns / v_ns
        ratio = np.nan_to_num(ratio) # handles origin where v_ns=0
    x_square[mask_ns] = np.sign(v_ns) * r_disk[mask_ns] * ratio
    y_square[mask_ns] = np.sign(v_ns) * r_disk[mask_ns]
    
    # Map from [-1,1]x[-1,1] to [0,1]x[0,1]
    points_x = (x_square + 1) / 2.0
    points_y = (y_square + 1) / 2.0
    
    x0_s1_pos = np.vstack([points_x, points_y]).T.flatten().tolist()
    
    # Estimate initial radius based on area and a reasonable packing density
    r_init_s1 = np.sqrt(0.7 / (n * np.pi))
    x0_s1 = np.array(x0_s1_pos + [r_init_s1])

    bounds_s1 = [(0, 1)] * (2 * n) + [(0, 0.5)]
    cons_s1 = [{'type': 'ineq', 'fun': constraints_stage1}]
    
    # Run a short optimization for Stage 1.
    res_s1 = minimize(objective_stage1, x0_s1, method='SLSQP', bounds=bounds_s1,
                      constraints=cons_s1, options={'maxiter': 250, 'ftol': 1e-7})

    # Use the result of Stage 1 as the initial guess for Stage 2.
    if res_s1.success:
        best_pos = res_s1.x[:-1].reshape((n, 2))
        best_r = res_s1.x[-1]
        initial_circles = np.hstack([best_pos, np.full((n, 1), best_r)])
    else:
        # Fallback to the original grid if Stage 1 fails.
        initial_circles = np.zeros((n, 3))
        count = 0
        for i in range(grid_size):
            for j in range(grid_size):
                if count >= n: break
                initial_circles[count] = [(2*i+1)*r_init_s1, (2*j+1)*r_init_s1, r_init_s1]
                count += 1
            if count >= n: break
    
    x0 = initial_circles.flatten()

    # --- STAGE 2: Main optimization to maximize sum of variable radii ---
    def objective(params):
        radii = params[2::3]
        return -np.sum(radii)

    def constraints(params):
        circles = params.reshape((n, 3))
        positions = circles[:, :2]
        radii = circles[:, 2]

        c_contain = np.concatenate([
            positions[:, 0] - radii, 1 - positions[:, 0] - radii,
            positions[:, 1] - radii, 1 - positions[:, 1] - radii
        ])

        # Use squared distances here as well for consistency and stability.
        # dist_sq >= (ri + rj)^2  =>  dist_sq - (ri + rj)^2 >= 0
        if n > 1:
            dist_sq = pdist(positions, 'sqeuclidean')
            i, j = np.triu_indices(n, k=1)
            radii_sum_pairs = radii[i] + radii[j]
            c_overlap = dist_sq - radii_sum_pairs**2
            return np.concatenate([c_contain, c_overlap])
        return c_contain

    bounds = []
    for _ in range(n):
        bounds.append((0, 1))    # Bound for x
        bounds.append((0, 1))    # Bound for y
        bounds.append((0, 0.5))  # Bound for r

    cons = [{'type': 'ineq', 'fun': constraints}]
    
    # Give the main optimization more iterations and tighter tolerance to fine-tune the solution.
    # Give the main optimization more iterations to better fine-tune the solution,
    # capitalizing on the superior starting point from the phyllotaxis pattern.
    res = minimize(objective, x0, method='SLSQP', bounds=bounds, constraints=cons,
                   options={'maxiter': 3000, 'ftol': 1e-9, 'disp': False})

    if res.success:
        return res.x.reshape((n, 3))
    else:
        # If the main optimization fails, return the result from Stage 1,
        # which is guaranteed to be a valid, dense packing.
        return initial_circles


# EVOLVE-BLOCK-END
