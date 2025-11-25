# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize, NonlinearConstraint
from numba import jit


@jit(nopython=True, cache=True)
def stage1_constraints_numba(v: np.ndarray, N: int) -> np.ndarray:
    """Numba-optimized constraints for equal-sized circles."""
    r = v[-1]
    centers = v[:-1].reshape((N, 2))
    
    # Containment: r <= x,y <= 1-r
    containment_c = np.empty(4 * N)
    for i in range(N):
        containment_c[i]       = centers[i, 0] - r
        containment_c[i + N]   = 1 - centers[i, 0] - r
        containment_c[i + 2*N] = centers[i, 1] - r
        containment_c[i + 3*N] = 1 - centers[i, 1] - r
        
    # Non-overlap: dist(ci, cj) >= 2*r
    num_pairs = N * (N - 1) // 2
    overlap_c = np.empty(num_pairs)
    k = 0
    if N > 1:
        for i in range(N):
            for j in range(i + 1, N):
                dist_sq = (centers[i, 0] - centers[j, 0])**2 + (centers[i, 1] - centers[j, 1])**2
                overlap_c[k] = np.sqrt(dist_sq) - 2 * r
                k += 1
    
    return np.concatenate((containment_c, overlap_c))


@jit(nopython=True, cache=True)
def constraints_numba(x: np.ndarray, N: int) -> np.ndarray:
    """Numba-optimized constraints for variable-sized circles."""
    circles = x.reshape((N, 3))
    centers = circles[:, :2]
    radii = circles[:, 2]

    # Containment: ri <= xi,yi <= 1-ri
    containment_c = np.empty(4 * N)
    for i in range(N):
        containment_c[i]       = centers[i, 0] - radii[i]
        containment_c[i + N]   = 1 - centers[i, 0] - radii[i]
        containment_c[i + 2*N] = centers[i, 1] - radii[i]
        containment_c[i + 3*N] = 1 - centers[i, 1] - radii[i]

    # Non-overlap: dist(ci, cj) >= ri + rj
    num_pairs = N * (N - 1) // 2
    overlap_c = np.empty(num_pairs)
    k = 0
    if N > 1:
        for i in range(N):
            for j in range(i + 1, N):
                dist_sq = (centers[i, 0] - centers[j, 0])**2 + (centers[i, 1] - centers[j, 1])**2
                overlap_c[k] = np.sqrt(dist_sq) - (radii[i] + radii[j])
                k += 1
            
    return np.concatenate((containment_c, overlap_c))


def circle_packing32() -> np.ndarray:
    """
    Places 32 non-overlapping circles in the unit square in order to maximize the sum of radii.
    This is achieved by using a two-stage optimization with scipy.optimize.minimize (SLSQP).
    Stage 1: Find an optimal packing of equal-sized circles to establish a good spatial layout.
    Stage 2: Use this layout as a starting point to optimize for the maximal sum of variable radii.
    This version uses a sunflower seed initial guess and Numba-jitted constraint functions.

    Returns:
        circles: np.array of shape (32,3), where the i-th row (x,y,r) stores the
                 (x,y) coordinates of the i-th circle of radius r.
    """
    N = 32
    np.random.seed(42)  # For reproducibility

    # --- STAGE 1: Find optimal packing for equal radii (maximin problem) ---
    # This provides a robust initial guess for the main problem by first finding a
    # good spatial configuration.

    def stage1_objective(v: np.ndarray) -> float:
        """Objective is to maximize the single radius r, equivalent to minimizing -r."""
        return -v[-1]

    # Initial guess for Stage 1 (sunflower seed arrangement for a better start)
    indices = np.arange(N)
    phi = (1 + np.sqrt(5)) / 2  # Golden ratio
    theta = 2 * np.pi * indices / phi
    r_sun = np.sqrt(indices / (N - 1))
    centers_init_s1 = 0.5 + 0.45 * np.column_stack([r_sun * np.cos(theta), r_sun * np.sin(theta)])
    r_init_s1 = 0.08  # A reasonable starting radius
    v0_s1 = np.append(centers_init_s1.ravel(), r_init_s1)

    bounds_s1 = [(0.0, 1.0)] * (2 * N) + [(0.0, 0.5)]

    stage1_nlc = NonlinearConstraint(lambda v: stage1_constraints_numba(v, N), 0, np.inf)
    res_s1 = minimize(
        stage1_objective,
        v0_s1,
        method='SLSQP',
        bounds=bounds_s1,
        constraints=[stage1_nlc],
        options={'maxiter': 2000, 'ftol': 1e-9, 'disp': False}
    )

    # --- STAGE 2: Optimize for the sum of radii using Stage 1 result ---
    
    def objective(x: np.ndarray) -> float:
        """Computes the negative sum of radii."""
        return -np.sum(x[2::3])

    if res_s1.success:
        best_centers = res_s1.x[:-1].reshape((N, 2))
        best_r_uniform = res_s1.x[-1]
        x0 = np.zeros(N * 3)
        x0[0::3] = best_centers[:, 0]
        x0[1::3] = best_centers[:, 1]
        x0[2::3] = best_r_uniform
    else:
        # Fallback to original sunflower guess if Stage 1 fails
        radii_init = np.full(N, r_init_s1)
        x0 = np.zeros(N * 3)
        x0[0::3] = centers_init_s1[:, 0]
        x0[1::3] = centers_init_s1[:, 1]
        x0[2::3] = radii_init

    bounds = []
    for _ in range(N):
        bounds.extend([(0.0, 1.0), (0.0, 1.0), (0.0, 0.5)])

    nonlinear_constraint = NonlinearConstraint(lambda x: constraints_numba(x, N), 0, np.inf)
    res = minimize(
        objective,
        x0,
        method='SLSQP',
        bounds=bounds,
        constraints=[nonlinear_constraint],
        options={'maxiter': 5000, 'ftol': 1e-10, 'disp': False} # Increased iterations and precision
    )

    if res.success:
        circles = res.x.reshape((N, 3))
    else:
        # If optimization fails, return the best guess we have.
        circles = x0.reshape((N, 3))

    return circles


# EVOLVE-BLOCK-END
