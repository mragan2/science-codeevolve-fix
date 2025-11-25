# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize, differential_evolution, NonlinearConstraint
from scipy.spatial.distance import pdist, squareform
from numba import jit

# --- Constants ---
N_CIRCLES = 32

# --- Helper functions for optimization ---

@jit(nopython=True, fastmath=True)
def _numba_overlap_cons(coords: np.ndarray, radii: np.ndarray) -> np.ndarray:
    """
    Numba-accelerated calculation of non-overlap constraints.
    Constraint is of the form: (xi-xj)^2 + (yi-yj)^2 - (ri+rj)^2 >= 0
    """
    n = coords.shape[0]
    num_constraints = n * (n - 1) // 2
    overlap_cons = np.empty(num_constraints, dtype=np.float64)
    k = 0
    for i in range(n):
        for j in range(i + 1, n):
            dist_sq = (coords[i, 0] - coords[j, 0])**2 + (coords[i, 1] - coords[j, 1])**2
            radii_sum = radii[i] + radii[j]
            overlap_cons[k] = dist_sq - radii_sum**2
            k += 1
    return overlap_cons


@jit(nopython=True, fastmath=True)
def _numba_relax(coords: np.ndarray, radii: np.ndarray, steps: int) -> np.ndarray:
    """
    Numba-accelerated physics-based relaxation of circle positions.
    Resolves overlaps and boundary violations to create a better initial guess.
    Radii are kept constant during relaxation.
    """
    n = coords.shape[0]
    for _ in range(steps):
        # 1. Resolve overlaps by pushing circles apart
        for i in range(n):
            for j in range(i + 1, n):
                d_vec_x = coords[j, 0] - coords[i, 0]
                d_vec_y = coords[j, 1] - coords[i, 1]
                dist_sq = d_vec_x**2 + d_vec_y**2
                radii_sum = radii[i] + radii[j]
                
                if dist_sq < radii_sum**2 and dist_sq > 1e-12:
                    dist = np.sqrt(dist_sq)
                    overlap = radii_sum - dist
                    # Push each circle by half the overlap distance + a small epsilon
                    push = overlap / 2.0 + 1e-6 
                    move_x = (d_vec_x / dist) * push
                    move_y = (d_vec_y / dist) * push
                    coords[i, 0] -= move_x
                    coords[i, 1] -= move_y
                    coords[j, 0] += move_x
                    coords[j, 1] += move_y

        # 2. Enforce boundary constraints by clamping positions
        for i in range(n):
            # Clamp x coordinate
            if coords[i, 0] < radii[i]:
                coords[i, 0] = radii[i]
            elif coords[i, 0] > 1.0 - radii[i]:
                coords[i, 0] = 1.0 - radii[i]
            # Clamp y coordinate
            if coords[i, 1] < radii[i]:
                coords[i, 1] = radii[i]
            elif coords[i, 1] > 1.0 - radii[i]:
                coords[i, 1] = 1.0 - radii[i]
            
    return coords


def _objective(params: np.ndarray) -> float:
    """Objective function to minimize: the negative sum of radii."""
    radii = params.reshape((N_CIRCLES, 3))[:, 2]
    return -np.sum(radii)

def _constraints(params: np.ndarray) -> np.ndarray:
    """
    Constraint function returning all constraint values.
    For SLSQP, constraints are of the form C_j >= 0.
    A positive returned value means the constraint is satisfied.
    """
    circles = params.reshape((N_CIRCLES, 3))
    coords = circles[:, :2]
    radii = circles[:, 2]

    # 1. Containment constraints:
    containment_cons = np.concatenate([
        coords[:, 0] - radii,
        1 - coords[:, 0] - radii,
        coords[:, 1] - radii,
        1 - coords[:, 1] - radii
    ])

    # 2. Non-overlap constraints (calculated with Numba for performance)
    overlap_cons = _numba_overlap_cons(coords, radii)

    return np.concatenate([containment_cons, overlap_cons])


def circle_packing32() -> np.ndarray:
    """
    Places 32 non-overlapping circles in the unit square to maximize the sum of radii.
    This implementation uses a hybrid global-local optimization strategy:
    1. An initial guess is generated from a relaxed, differentiated hexagonal grid.
    2. Differential Evolution (a global optimizer) is run with a limited budget
       to find a superior starting region, leveraging parallel processing.
    3. The better of the initial guess and the DE result is used to seed a
       multi-start local search using SLSQP with adaptive perturbations.
    4. A final high-precision SLSQP refinement is performed on the best solution found.
    """
    # 1. Initial Guess: A refined hexagonal grid arrangement
    np.random.seed(42)
    r_base = 0.088
    rows_hex = 7
    cols_hex = 5
    points = []
    x_spacing_unit = 2.0
    y_spacing_unit = np.sqrt(3.0)
    for i in range(rows_hex):
        for j in range(cols_hex):
            x = j * x_spacing_unit
            y = i * y_spacing_unit
            if i % 2 == 1:
                x += x_spacing_unit / 2
            points.append([x, y])
    points = np.array(points)
    grid_center = np.mean(points, axis=0)
    dist_from_center = np.linalg.norm(points - grid_center, axis=1)
    indices_to_keep = np.argsort(dist_from_center)[:N_CIRCLES]
    points = points[indices_to_keep]
    min_x_pattern, max_x_pattern = np.min(points[:, 0]), np.max(points[:, 0])
    min_y_pattern, max_y_pattern = np.min(points[:, 1]), np.max(points[:, 1])
    available_packing_side = 1.0 - 2 * r_base
    pattern_width = max_x_pattern - min_x_pattern
    pattern_height = max_y_pattern - min_y_pattern
    scale_factor = available_packing_side / max(pattern_width, pattern_height, 1e-9)
    points[:, 0] = (points[:, 0] - min_x_pattern) * scale_factor + r_base
    points[:, 1] = (points[:, 1] - min_y_pattern) * scale_factor + r_base
    center_of_square = np.array([0.5, 0.5])
    distances_from_center = np.linalg.norm(points - center_of_square, axis=1)
    min_dist, max_dist = np.min(distances_from_center), np.max(distances_from_center)
    dist_range = max_dist - min_dist
    if dist_range < 1e-6:
        norm_dist = np.zeros(N_CIRCLES)
    else:
        norm_dist = (distances_from_center - min_dist) / dist_range
    radius_bonus = 0.02 # Increased from 0.015 to encourage more radius differentiation
    initial_radii = r_base + radius_bonus * norm_dist
    initial_circles = np.hstack([points, initial_radii.reshape(-1, 1)])
    relaxed_coords = _numba_relax(initial_circles[:, :2].copy(), initial_circles[:, 2].copy(), steps=500)
    x0_initial_hex = np.hstack([relaxed_coords, initial_radii.reshape(-1, 1)]).flatten()

    # 2. Define bounds for optimization variables
    bounds = []
    for _ in range(N_CIRCLES):
        bounds.extend([(0.0, 1.0), (0.0, 1.0), (1e-6, 0.5)])

    # 3. Global search phase with Differential Evolution
    de_nlc = NonlinearConstraint(_constraints, 0, np.inf)
    de_res = differential_evolution(
        _objective,
        bounds,
        constraints=de_nlc,
        maxiter=150, # Increased from 60 for more thorough global exploration
        popsize=30, # Increased from 20 for a larger population
        strategy='best1bin',
        recombination=0.8,
        polish=True,
        seed=42,
        workers=-1,
        updating='immediate'
    )

    initial_hex_score = -_objective(x0_initial_hex)
    de_score = -de_res.fun if hasattr(de_res, 'fun') and de_res.success else -np.inf

    if de_score > initial_hex_score:
        x0_base_for_restarts = de_res.x
    else:
        x0_base_for_restarts = x0_initial_hex

    # 4. Local search phase: Multi-start SLSQP
    cons = {'type': 'ineq', 'fun': _constraints}
    num_restarts = 20 # Increased from 15 to account for potentially more diverse starting points
    best_sum_radii = -np.inf
    best_circles = None

    for i in range(num_restarts):
        x0_current = x0_base_for_restarts.copy()
        if i > 0:
            decay_factor = 0.88**(i - 1)
            current_perturb_pos = 0.03 * decay_factor
            current_perturb_rad = 0.02 * decay_factor
            x0_current[::3] += np.random.uniform(-current_perturb_pos, current_perturb_pos, N_CIRCLES)
            x0_current[1::3] += np.random.uniform(-current_perturb_pos, current_perturb_pos, N_CIRCLES)
            x0_current[2::3] += np.random.uniform(-current_perturb_rad, current_perturb_rad, N_CIRCLES)
            for k in range(N_CIRCLES):
                x_idx, y_idx, r_idx = 3*k, 3*k+1, 3*k+2
                x0_current[x_idx] = np.clip(x0_current[x_idx], bounds[x_idx][0], bounds[x_idx][1])
                x0_current[y_idx] = np.clip(x0_current[y_idx], bounds[y_idx][0], bounds[y_idx][1])
                x0_current[r_idx] = np.clip(x0_current[r_idx], bounds[r_idx][0], bounds[r_idx][1])
            
            perturbed_circles = x0_current.reshape((N_CIRCLES, 3))
            relaxed_coords = _numba_relax(perturbed_circles[:, :2].copy(), perturbed_circles[:, 2].copy(), steps=200)
            x0_current = np.hstack([relaxed_coords, perturbed_circles[:, 2].reshape(-1, 1)]).flatten()

        res = minimize(
            fun=_objective, x0=x0_current, method='SLSQP', bounds=bounds, constraints=cons,
            options={'maxiter': 2000, 'ftol': 1e-12, 'disp': False, 'eps': 1e-9}
        )
        current_sum_radii = -res.fun
        if res.success and current_sum_radii > best_sum_radii:
            best_sum_radii = current_sum_radii
            best_circles = res.x.reshape((N_CIRCLES, 3))
        elif not res.success and current_sum_radii > best_sum_radii:
            best_sum_radii = current_sum_radii
            best_circles = res.x.reshape((N_CIRCLES, 3))

    if best_circles is None:
        best_circles = x0_base_for_restarts.reshape((N_CIRCLES, 3))

    # 5. Final high-precision refinement
    final_refinement_res = minimize(
        fun=_objective, x0=best_circles.flatten(), method='SLSQP', bounds=bounds, constraints=cons,
        options={'maxiter': 5000, 'ftol': 1e-13, 'disp': False, 'eps': 1e-10}
    )
    final_sum_radii = -final_refinement_res.fun
    if final_refinement_res.success and final_sum_radii > best_sum_radii:
        best_circles = final_refinement_res.x.reshape((N_CIRCLES, 3))

    return best_circles


# EVOLVE-BLOCK-END
