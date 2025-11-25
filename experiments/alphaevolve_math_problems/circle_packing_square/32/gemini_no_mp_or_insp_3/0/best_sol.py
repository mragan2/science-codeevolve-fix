# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
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
    #    r_i <= x_i <= 1 - r_i  =>  x_i - r_i >= 0  AND  1 - x_i - r_i >= 0
    #    r_i <= y_i <= 1 - r_i  =>  y_i - r_i >= 0  AND  1 - y_i - r_i >= 0
    containment_cons = np.concatenate([
        coords[:, 0] - radii,
        1 - coords[:, 0] - radii,
        coords[:, 1] - radii,
        1 - coords[:, 1] - radii
    ])

    # 2. Non-overlap constraints:
    #    sqrt((x_i-x_j)^2 + (y_i-y_j)^2) >= r_i + r_j
    #    (x_i-x_j)^2 + (y_i-y_j)^2 - (r_i+r_j)^2 >= 0
    
    # 2. Non-overlap constraints (calculated with Numba for performance)
    overlap_cons = _numba_overlap_cons(coords, radii)

    return np.concatenate([containment_cons, overlap_cons])


def circle_packing32() -> np.ndarray:
    """
    Places 32 non-overlapping circles in the unit square to maximize the sum of radii.
    This implementation uses scipy's SLSQP optimizer starting from a hexagonal grid layout,
    which is denser and provides a better initial guess.
    """
    # 1. Initial Guess: A refined hexagonal grid arrangement
    # We aim for a hexagonal grid that provides a dense, non-overlapping starting point.
    # For N=32, a 7x5 grid (35 points) is a good base to select from.

    # Set a fixed random seed for reproducibility of perturbations later
    np.random.seed(42)

    # Base radius for initial hexagonal packing - slightly adjusted from previous guess
    # More aggressive r_base based on theoretical packing density.
    r_base = 0.088 # Adjusted from 0.082

    # Grid dimensions for the hexagonal pattern (will generate 35 points, then take 32)
    rows_hex = 7
    cols_hex = 5

    points = []
    # Generate a full 7x5=35 point hexagonal grid
    x_spacing_unit = 2.0
    y_spacing_unit = np.sqrt(3.0)

    for i in range(rows_hex):
        for j in range(cols_hex):
            x = j * x_spacing_unit
            y = i * y_spacing_unit
            if i % 2 == 1: # Stagger odd rows
                x += x_spacing_unit / 2
            points.append([x, y])
    
    # Select the 32 points closest to the geometric center of the grid
    # This creates a more compact and symmetric initial arrangement.
    points = np.array(points)
    grid_center = np.mean(points, axis=0)
    dist_from_center = np.linalg.norm(points - grid_center, axis=1)
    indices_to_keep = np.argsort(dist_from_center)[:N_CIRCLES]
    points = points[indices_to_keep]

    # Calculate current bounds of the generated pattern (unit-scaled)
    min_x_pattern, max_x_pattern = np.min(points[:, 0]), np.max(points[:, 0])
    min_y_pattern, max_y_pattern = np.min(points[:, 1]), np.max(points[:, 1])

    # Calculate actual available packing space for centers (1 - 2*r_base)
    available_packing_side = 1.0 - 2 * r_base

    # Scale factors to fit the pattern into the available packing space while maintaining aspect ratio
    # We scale the larger dimension of the pattern to fit, and apply the same scale to the other dimension.
    # This prevents distortion and ensures non-overlap relative to r_base.
    pattern_width = max_x_pattern - min_x_pattern
    pattern_height = max_y_pattern - min_y_pattern
    
    # Avoid division by zero if pattern is a single point or line
    scale_factor = available_packing_side / max(pattern_width, pattern_height, 1e-9)
    
    points[:, 0] = (points[:, 0] - min_x_pattern) * scale_factor + r_base
    points[:, 1] = (points[:, 1] - min_y_pattern) * scale_factor + r_base

    # 1a. Differentiate initial radii based on distance from the center.
    # This encourages a multi-scale solution, which is common in optimal packings.
    # Circles near the boundary are given a slightly larger initial radius.
    center_of_square = np.array([0.5, 0.5])
    distances_from_center = np.linalg.norm(points - center_of_square, axis=1)
    min_dist, max_dist = np.min(distances_from_center), np.max(distances_from_center)
    dist_range = max_dist - min_dist
    
    if dist_range < 1e-6:
        norm_dist = np.zeros(N_CIRCLES)
    else:
        norm_dist = (distances_from_center - min_dist) / dist_range

    radius_bonus = 0.015  # Tunable parameter for radius differentiation
    initial_radii = r_base + radius_bonus * norm_dist
    initial_circles = np.hstack([points, initial_radii.reshape(-1, 1)])

    # 1b. Relax the initial guess to resolve overlaps before optimization.
    # This uses a simple physics-based simulation to push overlapping circles apart,
    # providing a much better and near-feasible starting point for SLSQP.
    relaxed_coords = _numba_relax(initial_circles[:, :2].copy(), initial_circles[:, 2].copy(), steps=500)
    initial_circles = np.hstack([relaxed_coords, initial_radii.reshape(-1, 1)])
    
    x0_initial_hex = initial_circles.flatten()

    # 2. Bounds for each variable [x, y, r] for each circle
    # 0 <= x, y <= 1
    # 0 <= r <= 0.5 (theoretical max for one circle)
    bounds = []
    for _ in range(N_CIRCLES):
        bounds.extend([(0.0, 1.0), (0.0, 1.0), (1e-6, 0.5)]) # Minimum radius 1e-6 to avoid numerical issues

    # 3. Constraints dictionary for the optimizer
    cons = {'type': 'ineq', 'fun': _constraints}

    # 4. Run multiple SLSQP optimizations with perturbed initial guesses
    # This acts as a 'multi-start' local search to escape shallow local minima.
    num_restarts = 15 # Increased for more thorough exploration
    best_sum_radii = -np.inf
    best_circles = None

    for i in range(num_restarts):
        x0_current = x0_initial_hex.copy()
        if i > 0: # Perturb for subsequent runs
            # Use an adaptive perturbation scale that decays with each restart.
            # This allows for broad exploration initially and fine-tuning later.
            decay_factor = 0.88**(i - 1)
            current_perturb_pos = 0.03 * decay_factor # Start larger
            current_perturb_rad = 0.02 * decay_factor # Start larger

            # Add noise to x,y positions
            x0_current[::3] += np.random.uniform(-current_perturb_pos, current_perturb_pos, N_CIRCLES)
            x0_current[1::3] += np.random.uniform(-current_perturb_pos, current_perturb_pos, N_CIRCLES)
            
            # Add noise to radii
            x0_current[2::3] += np.random.uniform(-current_perturb_rad, current_perturb_rad, N_CIRCLES)
            
            # Clamp perturbed values to their respective bounds
            for k in range(N_CIRCLES):
                x_idx, y_idx, r_idx = 3*k, 3*k+1, 3*k+2
                x0_current[x_idx] = np.clip(x0_current[x_idx], bounds[x_idx][0], bounds[x_idx][1])
                x0_current[y_idx] = np.clip(x0_current[y_idx], bounds[y_idx][0], bounds[y_idx][1])
                x0_current[r_idx] = np.clip(x0_current[r_idx], bounds[r_idx][0], bounds[r_idx][1])

            # Re-relax the perturbed configuration to ensure it's a good starting point
            perturbed_circles = x0_current.reshape((N_CIRCLES, 3))
            relaxed_coords = _numba_relax(
                perturbed_circles[:, :2].copy(),
                perturbed_circles[:, 2].copy(),
                steps=200  # Fewer steps needed for re-relaxation
            )
            x0_current = np.hstack([relaxed_coords, perturbed_circles[:, 2].reshape(-1, 1)]).flatten()

        # Run SLSQP
        res = minimize(
            fun=_objective,
            x0=x0_current,
            method='SLSQP',
            bounds=bounds,
            constraints=cons,
            options={'maxiter': 2000, 'ftol': 1e-12, 'disp': False, 'eps': 1e-9} # Increased maxiter, tighter ftol, adjusted eps
        )

        current_sum_radii = -res.fun # objective is negative sum of radii
        if res.success and current_sum_radii > best_sum_radii:
            best_sum_radii = current_sum_radii
            best_circles = res.x.reshape((N_CIRCLES, 3))
        elif not res.success and current_sum_radii > best_sum_radii:
            # Even if not fully converged, if it's better, keep it.
            best_sum_radii = current_sum_radii
            best_circles = res.x.reshape((N_CIRCLES, 3))
    
    # If no successful optimization found a better result, fallback to the last best, or initial.
    if best_circles is None:
        best_circles = x0_initial_hex.reshape((N_CIRCLES, 3)) # Fallback to the initial hexagonal guess

    # 5. Final high-precision refinement of the best found solution
    # This polishes the best result from the multi-start search to achieve maximum precision.
    if best_circles is not None:
        final_refinement_res = minimize(
            fun=_objective,
            x0=best_circles.flatten(),
            method='SLSQP',
            bounds=bounds,
            constraints=cons,
            options={'maxiter': 5000, 'ftol': 1e-13, 'disp': False, 'eps': 1e-10}
        )
        
        final_sum_radii = -final_refinement_res.fun
        # Only accept the refined result if it's an improvement and valid
        if final_refinement_res.success and final_sum_radii > best_sum_radii:
            best_circles = final_refinement_res.x.reshape((N_CIRCLES, 3))

    return best_circles


# EVOLVE-BLOCK-END
