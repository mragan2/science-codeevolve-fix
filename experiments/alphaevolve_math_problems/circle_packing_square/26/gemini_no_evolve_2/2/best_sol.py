# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize, NonlinearConstraint
import numba

# Numba-jitted helper function for fast pairwise overlap constraint calculation.
# This is the performance bottleneck of the optimization.
@numba.jit(nopython=True, fastmath=True)
def _fast_overlap_constraints(p_flat: np.ndarray, n: int) -> np.ndarray:
    """
    Calculates the squared distance minus squared radius sum for all pairs of circles.
    A positive value means no overlap.
    """
    p = p_flat.reshape(n, 3)
    num_pairs = n * (n - 1) // 2
    overlap = np.empty(num_pairs, dtype=np.float64)
    k = 0
    for i in range(n):
        for j in range(i + 1, n):
            # (xi-xj)^2 + (yi-yj)^2 - (ri+rj)^2 >= 0
            dist_sq = (p[i, 0] - p[j, 0])**2 + (p[i, 1] - p[j, 1])**2
            r_sum_sq = (p[i, 2] + p[j, 2])**2
            overlap[k] = dist_sq - r_sum_sq
            k += 1
    return overlap


def circle_packing26() -> np.ndarray:
    """
    Generates an optimal arrangement of 26 non-overlapping circles
    within a unit square [0,1] x [0,1], maximizing the sum of their radii.
    
    This uses a multi-restart optimization strategy with diverse starting
    configurations to improve solution quality beyond single local optimization.
    """
    n = 26
    np.random.seed(42)  # For reproducible results

    # Objective function to maximize the sum of radii (by minimizing its negative).
    def objective(p: np.ndarray) -> float:
        return -np.sum(p[2::3])

    # Constraint function: all returned values must be >= 0 for a feasible solution.
    def constraints_func(p: np.ndarray) -> np.ndarray:
        circles = p.reshape(n, 3)
        x, y, r = circles.T
        
        # 1. Containment constraints
        containment = np.concatenate([x - r, 1 - x - r, y - r, 1 - y - r])
        
        # 2. Non-overlap constraints
        overlap = _fast_overlap_constraints(p, n)
        
        return np.concatenate([containment, overlap])

    def generate_initial_guess(strategy: str) -> np.ndarray:
        """Generate different starting configurations for multi-restart optimization."""
        if strategy == 'grid':
            # Original grid strategy with improved radius
            r_init = 0.085  # Slightly larger initial radius
            nx, ny = 5, 6
            x_coords = np.linspace(r_init, 1 - r_init, nx)
            y_coords = np.linspace(r_init, 1 - r_init, ny)
            grid_x, grid_y = np.meshgrid(x_coords, y_coords)
            coords = np.vstack([grid_x.ravel(), grid_y.ravel()]).T[:n]
            radii = np.full((n, 1), r_init)
            
        elif strategy == 'corner_focused':
            # Place larger circles in corners, smaller ones in center
            coords = []
            radii = []
            
            # Corner circles (4 large ones)
            corner_r = 0.15
            corner_positions = [
                [corner_r, corner_r], [1-corner_r, corner_r],
                [corner_r, 1-corner_r], [1-corner_r, 1-corner_r]
            ]
            coords.extend(corner_positions)
            radii.extend([corner_r] * 4)
            
            # Fill remaining space with smaller circles
            remaining = n - 4
            r_small = 0.07
            for i in range(remaining):
                # Random positions avoiding corners
                x = np.random.uniform(r_small + 0.2, 1 - r_small - 0.2)
                y = np.random.uniform(r_small + 0.2, 1 - r_small - 0.2)
                coords.append([x, y])
                radii.append(r_small)
                
            coords = np.array(coords)
            radii = np.array(radii).reshape(-1, 1)
            
        elif strategy == 'hierarchical':
            # Multi-scale approach: few large, many small
            coords = []
            radii = []
            
            # 6 medium circles
            medium_r = 0.12
            for i in range(6):
                x = np.random.uniform(medium_r, 1 - medium_r)
                y = np.random.uniform(medium_r, 1 - medium_r)
                coords.append([x, y])
                radii.append(medium_r)
            
            # 20 small circles
            small_r = 0.06
            for i in range(20):
                x = np.random.uniform(small_r, 1 - small_r)
                y = np.random.uniform(small_r, 1 - small_r)
                coords.append([x, y])
                radii.append(small_r)
                
            coords = np.array(coords)
            radii = np.array(radii).reshape(-1, 1)
            
        else:  # 'random'
            # Pure random initialization
            r_init = 0.08
            coords = np.random.uniform(r_init, 1 - r_init, (n, 2))
            radii = np.full((n, 1), r_init)
        
        return np.hstack([coords, radii]).flatten()

    # Define bounds and constraints
    bounds_list = []
    for _ in range(n):
        bounds_list.extend([(0.0, 1.0), (0.0, 1.0), (0.0, 0.5)])
    
    nonlinear_constraint = NonlinearConstraint(constraints_func, 0, np.inf)

    # Multi-restart optimization with different strategies
    strategies = ['grid', 'corner_focused', 'hierarchical', 'random']
    best_result = None
    best_objective = float('inf')

    for strategy in strategies:
        x0 = generate_initial_guess(strategy)
        
        # Run optimization with enhanced settings
        res = minimize(
            objective,
            x0,
            method='SLSQP',
            bounds=bounds_list,
            constraints=[nonlinear_constraint],
            options={'maxiter': 800, 'ftol': 1e-10, 'disp': False}
        )
        
        # Keep track of best result
        if res.success and res.fun < best_objective:
            best_objective = res.fun
            best_result = res

    # If all strategies failed, fall back to simple grid approach
    if best_result is None:
        x0 = generate_initial_guess('grid')
        best_result = minimize(
            objective,
            x0,
            method='SLSQP',
            bounds=bounds_list,
            constraints=[nonlinear_constraint],
            options={'maxiter': 500, 'ftol': 1e-9, 'disp': False}
        )

    final_circles = best_result.x.reshape(n, 3)
    return final_circles


# EVOLVE-BLOCK-END
