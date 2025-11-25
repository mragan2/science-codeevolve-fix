# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
from scipy.stats import qmc
import numba


def circle_packing32() -> np.ndarray:
    """
    Places 32 non-overlapping circles in the unit square to maximize the sum of radii.
    This is achieved using a two-stage optimization process with SLSQP.

    Stage 1: Maximize the minimum radius. This finds a good, uniform-radius
             packing which serves as an excellent starting point for the next stage.
             The variables are (x_1..n, y_1..n, r_uniform).

    Stage 2: Maximize the sum of radii. Starting from the Stage 1 solution, this
             stage allows radii to vary, finding a more optimal non-uniform packing.
             The variables are (x_1, y_1, r_1, ... x_n, y_n, r_n).

    This hybrid strategy is more robust and effective than a single optimization run
    from a random-like starting point.
    
    Returns:
        circles: np.array of shape (32,3), where the i-th row (x,y,r) stores the
                 (x,y) coordinates of the i-th circle of radius r.
    """
    n = 32
    seed = 42

    # --- STAGE 1: Maximize Minimum Radius (find a good uniform packing) ---
    @numba.jit(nopython=True, cache=True)
    def _get_overlap_constraints_uniform(p1_flat):
        # p1_flat is (x1, ..., xn, y1, ..., yn, r)
        n_circles = (len(p1_flat) - 1) // 2
        xs = p1_flat[:n_circles]
        ys = p1_flat[n_circles:2*n_circles]
        r = p1_flat[-1]
        
        num_pairs = n_circles * (n_circles - 1) // 2
        c_overlap = np.empty(num_pairs, dtype=np.float64)
        k = 0
        radii_sum_sq = (2 * r)**2
        for i in range(n_circles):
            for j in range(i + 1, n_circles):
                dist_sq = (xs[i] - xs[j])**2 + (ys[i] - ys[j])**2
                c_overlap[k] = dist_sq - radii_sum_sq
                k += 1
        return c_overlap

    def _objective1(p1):
        # p1 is (x1..xn, y1..yn, r), we want to maximize r
        return -p1[-1]

    def _constraints1(p1):
        xs = p1[:n]
        ys = p1[n:2*n]
        r = p1[-1]
        c_contain = np.concatenate((xs - r, 1.0 - xs - r, ys - r, 1.0 - ys - r))
        c_overlap = _get_overlap_constraints_uniform(p1)
        return np.concatenate((c_contain, c_overlap))

    # Generate initial centers on a 4x8 grid with some perturbation for Stage 1
    grid_rows = 4
    grid_cols = 8
    
    # Calculate center coordinates for a grid
    x_coords = np.linspace(0.5 / grid_cols, 1 - 0.5 / grid_cols, grid_cols)
    y_coords = np.linspace(0.5 / grid_rows, 1 - 0.5 / grid_rows, grid_rows)
    
    X, Y = np.meshgrid(x_coords, y_coords)
    initial_centers = np.vstack([X.ravel(), Y.ravel()]).T
    
    # Add a small random perturbation to break perfect symmetry and help optimization
    rng_init = np.random.default_rng(seed=seed)
    initial_centers += rng_init.uniform(-0.01, 0.01, size=initial_centers.shape)
    
    # Ensure centers remain within reasonable bounds after perturbation
    initial_centers = np.clip(initial_centers, 0.01, 0.99)
    
    # Initial radius guess for uniform packing. Max possible for 4x8 is 0.0625.
    # Starting slightly below this allows room for growth.
    initial_r1 = 0.06 
    
    p1_0 = np.concatenate([initial_centers[:, 0], initial_centers[:, 1], [initial_r1]])
    bounds1 = [(0, 1)] * (2 * n) + [(0, 0.5)]
    cons1 = [{'type': 'ineq', 'fun': _constraints1}]
    
    res1 = minimize(_objective1, p1_0, method='SLSQP',
                    bounds=bounds1, constraints=cons1,
                    options={'maxiter': 500, 'ftol': 1e-7})

    # --- STAGE 2: Maximize Sum of Radii (refine the packing) ---
    @numba.jit(nopython=True, cache=True)
    def _get_overlap_constraints(p_flat):
        num_pairs = n * (n - 1) // 2
        c_overlap = np.empty(num_pairs, dtype=np.float64)
        k = 0
        for i in range(n):
            i_idx = i * 3
            xi, yi, ri = p_flat[i_idx], p_flat[i_idx+1], p_flat[i_idx+2]
            for j in range(i + 1, n):
                j_idx = j * 3
                xj, yj, rj = p_flat[j_idx], p_flat[j_idx+1], p_flat[j_idx+2]
                dist_sq = (xi - xj)**2 + (yi - yj)**2
                radii_sum_sq = (ri + rj)**2
                c_overlap[k] = dist_sq - radii_sum_sq
                k += 1
        return c_overlap

    def _objective2(p):
        return -np.sum(p[2::3])

    def _constraints2(p):
        xs, ys, rs = p[0::3], p[1::3], p[2::3]
        c_contain = np.concatenate((xs - rs, 1.0 - xs - rs, ys - rs, 1.0 - ys - rs))
        c_overlap = _get_overlap_constraints(p)
        return np.concatenate((c_contain, c_overlap))

    # Build initial guess for Stage 2 from the result of Stage 1
    p1_final = res1.x if res1.success else p1_0
    
    # Initialize baseline for Stage 2 from Stage 1 result
    p2_0_baseline = np.zeros(n * 3)
    p2_0_baseline[0::3] = p1_final[:n]
    p2_0_baseline[1::3] = p1_final[n:2*n]
    p2_0_baseline[2::3] = p1_final[-1]  # Start all radii at the uniform value from Stage 1

    best_sum_radii = -np.inf
    best_circles_data = None
    
    # If Stage 1 was successful, use its result as an initial best feasible solution
    if res1.success:
        # Check initial feasibility using Stage 2 constraints
        if np.all(_constraints2(p2_0_baseline) >= -1e-8): # Allowing a small tolerance
            best_sum_radii = np.sum(p2_0_baseline[2::3])
            best_circles_data = p2_0_baseline
    
    # Fallback if Stage 1 failed or its result is somehow not feasible for Stage 2
    if best_circles_data is None:
        p2_0_baseline[2::3] = 0.001 # Set a very small radius for all circles
        best_sum_radii = n * 0.001
        best_circles_data = p2_0_baseline


    num_restarts = 10 # Increased for more thorough exploration
    rng_restarts = np.random.default_rng(seed=seed + 1) # Separate RNG for restarts
    
    # Max iterations for Stage 2
    max_iter2 = 5000 # Increased for more thorough optimization

    for restart_idx in range(num_restarts):
        p2_0_current = np.copy(p2_0_baseline)
        
        # Perturb positions
        p2_0_current[0::3] += rng_restarts.uniform(-0.01, 0.01, size=n)
        p2_0_current[1::3] += rng_restarts.uniform(-0.01, 0.01, size=n)
        
        # Perturb radii significantly
        initial_radii_factor = (1 + rng_restarts.uniform(-0.15, 0.15, size=n)) # Wider range for radii perturbation
        p2_0_current[2::3] = p2_0_current[2::3] * initial_radii_factor
        
        # Clip all parameters to their valid ranges
        # Centers: [0, 1]. Radii: [1e-6, 0.5].
        p2_0_current[0::3] = np.clip(p2_0_current[0::3], 0, 1)
        p2_0_current[1::3] = np.clip(p2_0_current[1::3], 0, 1)
        p2_0_current[2::3] = np.clip(p2_0_current[2::3], 1e-6, 0.5)

        # Define bounds for Stage 2 optimization
        bounds2 = []
        for _ in range(n):
            # Centers can be between 0 and 1. Radii must be positive.
            bounds2.extend([(0, 1), (0, 1), (1e-6, 0.5)])

        cons2 = [{'type': 'ineq', 'fun': _constraints2}]

        result = minimize(_objective2, p2_0_current, method='SLSQP',
                          bounds=bounds2, constraints=cons2,
                          options={'maxiter': max_iter2, 'ftol': 1e-9, 'disp': False})

        if result.success or result.message == 'Iteration limit exceeded':
            # Verify explicit constraint satisfaction before considering it a valid solution
            constraint_violations = _constraints2(result.x)
            # Allow a small tolerance for constraint violations due to floating point arithmetic
            if np.all(constraint_violations >= -1e-7): 
                current_sum_radii = np.sum(result.x[2::3])
                if current_sum_radii > best_sum_radii:
                    best_sum_radii = current_sum_radii
                    best_circles_data = result.x
        
    final_circles = best_circles_data.reshape((n, 3))
    
    return final_circles


# EVOLVE-BLOCK-END
