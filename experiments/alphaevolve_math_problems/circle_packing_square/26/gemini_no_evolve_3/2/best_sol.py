# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize, Bounds
from scipy.spatial.distance import pdist


def circle_packing26() -> np.ndarray:
    """
    Generates an optimal arrangement of 26 non-overlapping circles
    within a unit square [0,1] x [0,1], maximizing the sum of their radii.

    This is achieved via a two-stage optimization process using SLSQP:
    1.  Maximize the radius of 26 equal-sized circles to find a stable
        and well-spaced initial configuration.
    2.  Use the result from stage 1 as a starting point to optimize
        individual radii and positions to maximize the total sum of radii.
    """
    n = 26
    seed = 42  # For reproducibility
    np.random.seed(seed)

    # --- Stage 1: Maximize the radius of 26 equal circles ---
    def objective_stage1(p):
        # p = [x0, y0, ..., x(n-1), y(n-1), r]
        return -p[-1]  # Maximize r

    def constraints_stage1(p):
        centers = p[:-1].reshape(n, 2)
        r = p[-1]
        
        cons = []
        # Boundary: r <= x <= 1-r and r <= y <= 1-r
        cons.extend(centers[:, 0] - r)
        cons.extend(1 - centers[:, 0] - r)
        cons.extend(centers[:, 1] - r)
        cons.extend(1 - centers[:, 1] - r)
        
        # Overlap: (xi-xj)^2 + (yi-yj)^2 >= (2r)^2
        if n > 1:
            dist_sq = pdist(centers)**2
            overlap_cons = dist_sq - (2 * r)**2
            cons.extend(overlap_cons)
        
        return np.array(cons)

    # Initial guess: Jittered 5x6 grid for centers
    coords = []
    nx, ny = 5, 6
    xs = np.linspace(0.1, 0.9, nx)
    ys = np.linspace(0.1, 0.9, ny)
    for i in range(nx):
        for j in range(ny):
            coords.append([xs[i], ys[j]])
    initial_centers = np.array(coords[:n])
    initial_centers += np.random.uniform(-0.01, 0.01, initial_centers.shape)
    p1_0 = np.append(initial_centers.ravel(), 0.1) # Start with r=0.1

    # Bounds for Stage 1 variables
    lb1 = np.zeros(2 * n + 1)
    ub1 = np.ones(2 * n + 1)
    ub1[-1] = 0.5
    bounds1 = Bounds(lb1, ub1)

    # Run Stage 1 optimization
    res1 = minimize(objective_stage1, p1_0, method='SLSQP', bounds=bounds1,
                    constraints={'type': 'ineq', 'fun': constraints_stage1},
                    options={'maxiter': 500, 'disp': False, 'ftol': 1e-7})

    p1_res = res1.x if res1.success and np.all(np.isfinite(res1.x)) else p1_0

    # --- Stage 2: Maximize sum of radii, starting from Stage 1 result ---
    def objective_stage2(p):
        # p = [x0, y0, r0, ..., x(n-1), y(n-1), r(n-1)]
        return -np.sum(p[2::3])

    def constraints_stage2(p):
        p_reshaped = p.reshape(n, 3)
        centers = p_reshaped[:, :2]
        radii = p_reshaped[:, 2]

        cons = []
        # Boundary constraints
        cons.extend(centers[:, 0] - radii)
        cons.extend(1 - centers[:, 0] - radii)
        cons.extend(centers[:, 1] - radii)
        cons.extend(1 - centers[:, 1] - radii)

        # Overlap constraints: (xi-xj)^2+(yi-yj)^2 >= (ri+rj)^2
        if n > 1:
            dist_sq = pdist(centers)**2
            iu = np.triu_indices(n, k=1)
            radii_sum = radii[iu[0]] + radii[iu[1]]
            overlap_cons = dist_sq - radii_sum**2
            cons.extend(overlap_cons)

        return np.array(cons)

    # Initial guess for Stage 2 from Stage 1 result
    stage1_centers = p1_res[:-1].reshape(n, 2)
    stage1_r = p1_res[-1]
    p2_0 = np.zeros(3 * n)
    p2_0[0::3] = stage1_centers[:, 0]
    p2_0[1::3] = stage1_centers[:, 1]
    p2_0[2::3] = stage1_r

    # Bounds for Stage 2 variables
    lb2 = np.tile([0, 0, 0], n)
    ub2 = np.tile([1, 1, 0.5], n)
    bounds2 = Bounds(lb2, ub2)

    # Run Stage 2 optimization
    res2 = minimize(objective_stage2, p2_0, method='SLSQP', bounds=bounds2,
                    constraints={'type': 'ineq', 'fun': constraints_stage2},
                    options={'maxiter': 1000, 'disp': False, 'ftol': 1e-9})
    
    final_p = res2.x if res2.success and np.all(np.isfinite(res2.x)) else p2_0
    
    return final_p.reshape(n, 3)


# EVOLVE-BLOCK-END
