# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize, differential_evolution
from scipy.spatial.distance import pdist, squareform


def circle_packing26() -> np.ndarray:
    """
    Finds an optimal arrangement of 26 non-overlapping circles in a unit square
    to maximize the sum of their radii, using multi-stage optimization.

    Returns:
        circles: np.array of shape (26,3), where the i-th row (x,y,r) stores the
                 (x,y) coordinates of the i-th circle of radius r.
    """
    n = 26
    np.random.seed(42)  # For reproducible results

    # The state vector 'x' is a flat array of [x0, y0, r0, x1, y1, r1, ...]
    # Objective function: maximize sum of radii -> minimize -sum(radii)
    def objective(x):
        radii = x[2::3]
        return -np.sum(radii)

    # All constraints must be >= 0 for SLSQP's 'ineq' type
    def constraints_func(x):
        circles = x.reshape((n, 3))
        positions = circles[:, :2]
        radii = circles[:, 2]

        # 1. Boundary constraints:
        #    ri <= xi <= 1-ri  =>  xi - ri >= 0  and  1 - xi - ri >= 0
        #    ri <= yi <= 1-ri  =>  yi - ri >= 0  and  1 - yi - ri >= 0
        # These are vectorized for efficiency.
        c_boundary_x1 = positions[:, 0] - radii
        c_boundary_x2 = 1 - positions[:, 0] - radii
        c_boundary_y1 = positions[:, 1] - radii
        c_boundary_y2 = 1 - positions[:, 1] - radii
        
        # 2. Non-overlap constraints:
        #    sqrt((xi-xj)^2 + (yi-yj)^2) >= ri + rj
        #    (xi-xj)^2 + (yi-yj)^2 - (ri+rj)^2 >= 0
        # Using squared distances avoids sqrt, which is better for optimization.
        
        # Pairwise squared Euclidean distances between circle centers
        dist_sq = squareform(pdist(positions, 'sqeuclidean'))
        
        # Pairwise sum of radii, squared
        radii_sum = radii[:, np.newaxis] + radii
        radii_sum_sq = radii_sum**2
        
        # We only need the upper triangle of the matrix to avoid duplicate constraints
        # and self-comparisons (i.e., i < j).
        indices = np.triu_indices(n, k=1)
        c_overlap = dist_sq[indices] - radii_sum_sq[indices]

        return np.concatenate([
            c_boundary_x1, c_boundary_x2, c_boundary_y1, c_boundary_y2,
            c_overlap
        ])

    # Define constraints for the optimizer
    cons = [{'type': 'ineq', 'fun': constraints_func}]

    # Bounds for variables: 0 <= x,y <= 1 and 0 <= r <= 0.5
    bounds = []
    for i in range(n):
        bounds.append((0, 1))    # x_i
        bounds.append((0, 1))    # y_i
        bounds.append((0, 0.5))  # r_i

    def create_hierarchical_initialization():
        """Create initialization with varied circle sizes and strategic placement"""
        initial_circles = np.zeros((n, 3))
        
        # Strategy: Place larger circles first, then fill with smaller ones
        # Corner positions for large circles (exploit boundary advantages)
        corner_positions = np.array([
            [0.15, 0.15], [0.85, 0.15], [0.15, 0.85], [0.85, 0.85],  # corners
            [0.5, 0.15], [0.5, 0.85], [0.15, 0.5], [0.85, 0.5]       # edges
        ])
        
        # Size hierarchy: 4 large, 8 medium, 14 small circles
        large_r = 0.12
        medium_r = 0.08
        small_r = 0.05
        
        # Place large circles at corners/edges
        for i in range(min(8, n)):
            if i < len(corner_positions):
                initial_circles[i, :2] = corner_positions[i]
                initial_circles[i, 2] = large_r if i < 4 else medium_r
        
        # Fill remaining positions with grid + perturbation
        remaining = n - min(8, n)
        if remaining > 0:
            # Create a denser grid for remaining circles
            grid_size = int(np.ceil(np.sqrt(remaining * 1.5)))
            x_coords = np.linspace(0.2, 0.8, grid_size)
            y_coords = np.linspace(0.2, 0.8, grid_size)
            xx, yy = np.meshgrid(x_coords, y_coords)
            grid_points = np.stack([xx.ravel(), yy.ravel()], axis=-1)
            
            # Add perturbation and assign small radii
            perturbation = (np.random.rand(remaining, 2) - 0.5) * 0.05
            start_idx = min(8, n)
            initial_circles[start_idx:start_idx+remaining, :2] = grid_points[:remaining] + perturbation
            initial_circles[start_idx:start_idx+remaining, 2] = small_r
        
        return initial_circles

    def create_grid_initialization():
        """Create standard grid initialization as fallback"""
        nx, ny = 5, 6
        x_coords = np.linspace(1/(2*nx), 1 - 1/(2*nx), nx)
        y_coords = np.linspace(1/(2*ny), 1 - 1/(2*ny), ny)
        xx, yy = np.meshgrid(x_coords, y_coords)
        grid_points = np.stack([xx.ravel(), yy.ravel()], axis=-1)

        initial_circles = np.zeros((n, 3))
        initial_r = 0.5 / ny 
        
        perturbation = (np.random.rand(n, 2) - 0.5) * (initial_r / 10)
        initial_circles[:, :2] = grid_points[:n] + perturbation
        initial_circles[:, 2] = initial_r
        
        return initial_circles

    # Multi-stage optimization: try different approaches and select best
    best_result = None
    best_sum = -np.inf
    
    # Stage 1: Hierarchical initialization with SLSQP
    try:
        initial_circles = create_hierarchical_initialization()
        x0 = initial_circles.flatten()
        
        res = minimize(objective, x0, method='SLSQP', bounds=bounds, constraints=cons, 
                       options={'maxiter': 1500, 'ftol': 1e-9, 'disp': False})
        
        current_sum = -res.fun
        if current_sum > best_sum:
            best_sum = current_sum
            best_result = res.x.reshape((n, 3))
    except:
        pass
    
    # Stage 2: Grid initialization with SLSQP
    try:
        initial_circles = create_grid_initialization()
        x0 = initial_circles.flatten()
        
        res = minimize(objective, x0, method='SLSQP', bounds=bounds, constraints=cons, 
                       options={'maxiter': 1500, 'ftol': 1e-9, 'disp': False})
        
        current_sum = -res.fun
        if current_sum > best_sum:
            best_sum = current_sum
            best_result = res.x.reshape((n, 3))
    except:
        pass
    
    # Stage 3: Differential Evolution for global exploration (shorter run)
    try:
        res = differential_evolution(objective, bounds, constraints=cons, 
                                   seed=42, maxiter=200, popsize=10, 
                                   polish=True, disp=False)
        
        current_sum = -res.fun
        if current_sum > best_sum:
            best_sum = current_sum
            best_result = res.x.reshape((n, 3))
    except:
        pass
    
    # Stage 4: Refinement - use best result as starting point for final optimization
    if best_result is not None:
        try:
            x0 = best_result.flatten()
            res = minimize(objective, x0, method='SLSQP', bounds=bounds, constraints=cons, 
                           options={'maxiter': 2000, 'ftol': 1e-10, 'disp': False})
            
            current_sum = -res.fun
            if current_sum > best_sum:
                best_sum = current_sum
                best_result = res.x.reshape((n, 3))
        except:
            pass
    
    # Fallback if all methods fail
    if best_result is None:
        initial_circles = create_grid_initialization()
        best_result = initial_circles

    return best_result


# EVOLVE-BLOCK-END
