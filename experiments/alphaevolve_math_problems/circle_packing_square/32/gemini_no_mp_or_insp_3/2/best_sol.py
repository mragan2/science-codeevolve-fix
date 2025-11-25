# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
import numba


@numba.jit(nopython=True, fastmath=True)
def _fast_non_overlap_constraints(circles: np.ndarray) -> np.ndarray:
    """
    Numba-accelerated function to compute pairwise non-overlap constraints.
    (xi-xj)^2 + (yi-yj)^2 - (ri+rj)^2 >= 0
    """
    n = circles.shape[0]
    num_pairs = n * (n - 1) // 2
    non_overlap = np.empty(num_pairs, dtype=np.float64)
    k = 0
    for i in range(n):
        xi, yi, ri = circles[i]
        for j in range(i + 1, n):
            xj, yj, rj = circles[j]
            d_sq = (xi - xj)**2 + (yi - yj)**2
            r_sum_sq = (ri + rj)**2
            non_overlap[k] = d_sq - r_sum_sq
            k += 1
    return non_overlap


def circle_packing32() -> np.ndarray:
    """
    Places 32 non-overlapping circles in the unit square in order to maximize the sum of radii.
    This is achieved by setting up and solving a constrained optimization problem, starting
    from a good initial guess based on a grid layout.
    """
    n = 32

    # Define objective function (minimize negative sum of radii).
    def objective(p):
        return -np.sum(p[2::3])

    # Define a single, efficient constraint function.
    def all_constraints(p):
        circles = p.reshape((n, 3))
        x, y, r = circles.T

        # a) Boundary constraints (vectorized)
        # ri <= xi <= 1-ri  &  ri <= yi <= 1-ri
        boundary_cons = np.concatenate([x - r, (1.0 - x) - r, y - r, (1.0 - y) - r])
        
        # b) Radius non-negativity
        radius_cons = r

        # c) Non-overlap constraints (using Numba-JIT for speed)
        non_overlap_cons = _fast_non_overlap_constraints(circles)

        return np.concatenate([boundary_cons, radius_cons, non_overlap_cons])

    cons = [{'type': 'ineq', 'fun': all_constraints}]

    # Define bounds for each variable (x, y, r).
    bounds = []
    for _ in range(n):
        # x, y coordinates must be within [0,1]. Radii must be positive and <= 0.5
        bounds.extend([(0.0, 1.0), (0.0, 1.0), (1e-6, 0.5)]) # Changed lower bound for radius to 1e-6 to avoid numerical issues with zero radius

    best_sum_radii = -np.inf
    best_circles = np.zeros((n, 3)) # Initialize with zeros as fallback

    num_starts = 100 # Further increased number of multi-start optimization runs for more exhaustive exploration
    
    # Generate the base initial grid using a hexagonal lattice for a better start.
    # A hexagonal grid is the densest packing arrangement in 2D.
    points = []
    # Estimate the number of rows needed for a roughly square aspect ratio
    n_rows_est = int(np.ceil(np.sqrt(n * 2.0 / np.sqrt(3.0))))
    
    v_dist = 1.0 / n_rows_est
    h_dist = 2.0 * v_dist / np.sqrt(3.0)
    r_init = h_dist / 2.0

    # Generate points on the lattice, creating a superset to select from
    for i in range(n_rows_est + 2):
        y = v_dist / 2.0 + (i - 1) * v_dist
        n_cols_est = int(1.0 / h_dist) + 2
        for j in range(n_cols_est):
            x = h_dist / 2.0 + (j - 1) * h_dist
            if i % 2 == 1:
                x += h_dist / 2.0
            points.append([x, y, r_init])
    
    base_initial_circles = np.array(points)

    # Center the grid of points inside the unit square
    if len(base_initial_circles) > 0:
        min_x, max_x = np.min(base_initial_circles[:, 0]), np.max(base_initial_circles[:, 0])
        base_initial_circles[:, 0] += (1.0 - (max_x + min_x)) / 2.0
        min_y, max_y = np.min(base_initial_circles[:, 1]), np.max(base_initial_circles[:, 1])
        base_initial_circles[:, 1] += (1.0 - (max_y + min_y)) / 2.0

    # Select the n points closest to the center of the square
    # First, filter out any points that may have been pushed outside during centering
    valid_mask = np.all((base_initial_circles[:, :2] > r_init) & (base_initial_circles[:, :2] < 1 - r_init), axis=1)
    base_initial_circles = base_initial_circles[valid_mask]

    if len(base_initial_circles) >= n:
        center = np.array([0.5, 0.5])
        dist_sq = np.sum((base_initial_circles[:, :2] - center)**2, axis=1)
        indices = np.argsort(dist_sq)[:n]
        base_initial_circles = base_initial_circles[indices]
    else:
        # Fallback to the simpler square grid if hex generation fails to produce enough points
        print(f"Warning: Hexagonal grid generation produced {len(base_initial_circles)}/{n} points. Falling back to square grid.")
        base_initial_circles = np.zeros((n, 3))
        grid_dim = 6
        r_init_fb = 1.0 / (2 * grid_dim)
        step = 1.0 / grid_dim
        count = 0
        for i in range(grid_dim):
            for j in range(grid_dim):
                if count < n:
                    x = step / 2.0 + i * step
                    y = step / 2.0 + j * step
                    base_initial_circles[count] = [x, y, r_init_fb]
                    count += 1
    
    # Set a global seed for reproducibility of the multi-start sequence
    np.random.seed(42) 

    for start_idx in range(num_starts):
        # Create a new perturbed initial guess for each start
        x0_perturbed = base_initial_circles.flatten().copy()
        
        noise_strength = 0.008 # Slightly reduced perturbation strength to better preserve hexagonal structure while still providing diversity
        noise = np.random.uniform(-noise_strength, noise_strength, n * 3)
        
        x0_perturbed += noise
        
        # Clip radii first to ensure they are positive and within max bounds [1e-6, 0.5]
        x0_perturbed[2::3] = np.clip(x0_perturbed[2::3], 1e-6, 0.5) # Ensure radii are not too small or too large

        # Ensure the perturbed guess remains feasible for boundary constraints
        # Clip x, y coordinates so that r <= x <= 1-r and r <= y <= 1-r
        for i in range(n):
            current_r = x0_perturbed[3 * i + 2] # Use the now-perturbed and clipped radius
            x0_perturbed[3 * i] = np.clip(x0_perturbed[3 * i], current_r, 1.0 - current_r)
            x0_perturbed[3 * i + 1] = np.clip(x0_perturbed[3 * i + 1], current_r, 1.0 - current_r)

        # Run the SLSQP optimizer with more iterations.
        # The faster constraint evaluation allows for a more thorough search.
        res = minimize(objective, x0_perturbed, method='SLSQP', bounds=bounds, constraints=cons,
                       options={'maxiter': 1500, 'ftol': 1e-9, 'disp': False}) # Further increased maxiter for deeper local search

        if res.success:
            current_sum_radii = -res.fun
            if current_sum_radii > best_sum_radii:
                best_sum_radii = current_sum_radii
                best_circles = res.x.reshape((n, 3))
                # Optional: print progress for debugging/monitoring
                # print(f"Start {start_idx}: New best sum_radii = {best_sum_radii:.6f}")
        # else:
            # Optional: print failure for debugging
            # print(f"Start {start_idx}: Optimization failed: {res.message}")
    
    # If no optimization succeeded in finding a better solution than the initial -inf,
    # return the base_initial_circles as a robust fallback.
    if best_sum_radii == -np.inf:
        print("All optimization runs failed or found no improvement. Returning base grid.")
        return base_initial_circles
    else:
        return best_circles


# EVOLVE-BLOCK-END
