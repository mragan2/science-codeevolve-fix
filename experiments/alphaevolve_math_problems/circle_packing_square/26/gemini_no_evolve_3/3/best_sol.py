# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import differential_evolution, minimize
from numba import njit
import time

N_CIRCLES = 26
PENALTY_FACTOR = 2000.0 # Increased penalty factor for better constraint enforcement
OVERLAP_PENALTY_POWER = 2.5 # Slightly higher power for overlap penalty
BOUNDARY_PENALTY_POWER = 2.0 # Power for boundary penalty (squared is common)
MIN_RADIUS = 1e-6 # Minimum allowed radius to prevent numerical issues
MAX_RADIUS = 0.5 # Maximum possible radius for a circle in a unit square

@njit(cache=True)
def _unpack_params(params: np.ndarray) -> np.ndarray:
    """
    Unpacks a 1D array of parameters [x1, y1, r1, x2, y2, r2, ...]
    into a 2D array of circles [(x1, y1, r1), ...].
    """
    # Assuming params are ordered as [x_coords, y_coords, r_coords]
    x_coords = params[:N_CIRCLES]
    y_coords = params[N_CIRCLES:2*N_CIRCLES]
    r_coords = params[2*N_CIRCLES:]
    
    circles = np.empty((N_CIRCLES, 3), dtype=params.dtype)
    circles[:, 0] = x_coords
    circles[:, 1] = y_coords
    circles[:, 2] = r_coords
    return circles

@njit(cache=True)
def _boundary_constraint(params: np.ndarray, idx: int, constraint_type: int) -> float:
    """
    Numba-compatible boundary constraint function.
    constraint_type: 0=x-r, 1=y-r, 2=1-x-r, 3=1-y-r, 4=r-MIN_RADIUS
    """
    x = params[idx]
    y = params[N_CIRCLES + idx]
    r = params[2*N_CIRCLES + idx]
    
    if constraint_type == 0:
        return x - r
    elif constraint_type == 1:
        return y - r
    elif constraint_type == 2:
        return 1.0 - x - r
    elif constraint_type == 3:
        return 1.0 - y - r
    else:  # constraint_type == 4
        return r - MIN_RADIUS

@njit(cache=True)
def _overlap_constraint(params: np.ndarray, i: int, j: int) -> float:
    """
    Numba-compatible overlap constraint function.
    Returns distance_squared - (r1 + r2)^2, must be >= 0 for no overlap.
    """
    x1, y1, r1 = params[i], params[N_CIRCLES + i], params[2*N_CIRCLES + i]
    x2, y2, r2 = params[j], params[N_CIRCLES + j], params[2*N_CIRCLES + j]
    dx = x1 - x2
    dy = y1 - y2
    dist_sq = dx*dx + dy*dy
    min_dist_sq = (r1 + r2)**2
    return dist_sq - min_dist_sq

@njit(cache=True)
def _calculate_violations(circles: np.ndarray) -> float:
    """
    Calculates the sum of squared violations for boundary and overlap constraints.
    """
    violations = 0.0

    # Boundary violations
    for i in range(N_CIRCLES):
        x, y, r = circles[i]
        
        # Radii must be positive
        if r < MIN_RADIUS:
            violations += (MIN_RADIUS - r)**BOUNDARY_PENALTY_POWER
        
        # Containment within unit square [0,1]x[0,1]
        violations += max(0.0, r - x)**BOUNDARY_PENALTY_POWER
        violations += max(0.0, r - y)**BOUNDARY_PENALTY_POWER
        violations += max(0.0, x + r - 1.0)**BOUNDARY_PENALTY_POWER
        violations += max(0.0, y + r - 1.0)**BOUNDARY_PENALTY_POWER

    # Overlap violations
    for i in range(N_CIRCLES):
        for j in range(i + 1, N_CIRCLES):
            x1, y1, r1 = circles[i]
            x2, y2, r2 = circles[j]

            dx = x1 - x2
            dy = y1 - y2
            dist_sq = dx*dx + dy*dy
            min_dist = r1 + r2
            min_dist_sq = min_dist*min_dist
            
            if dist_sq < min_dist_sq:
                # Add penalty if circles overlap
                # The penalty is proportional to the square of the overlap depth
                violations += (min_dist_sq - dist_sq)**OVERLAP_PENALTY_POWER

    return violations

@njit(cache=True)
def _objective_function(params: np.ndarray) -> float:
    """
    Objective function to minimize for circle packing.
    Returns -sum(radii) + penalty for violations.
    """
    circles = _unpack_params(params)
    
    # Sum of radii (to be maximized, so we negate it for minimization)
    sum_radii = np.sum(circles[:, 2])

    # Calculate constraint violations
    violations = _calculate_violations(circles)
    
    return -sum_radii + PENALTY_FACTOR * violations

def _initial_population_multiscale(n_circles: int) -> np.ndarray:
    """
    Generates an initial population using multi-scale packing strategies.
    Combines corner optimization, hierarchical sizing, and structured patterns.
    """
    initial_pop = []
    np.random.seed(42)
    
    # Configuration 1: Corner-optimized with large anchor circles
    def create_corner_config():
        circles = []
        
        # Place 4 large corner circles (theoretical max radius ≈ 0.354 for corner placement)
        corner_r = 0.12  # Conservative corner radius
        corner_positions = [
            (corner_r, corner_r),           # bottom-left
            (1.0 - corner_r, corner_r),     # bottom-right  
            (corner_r, 1.0 - corner_r),     # top-left
            (1.0 - corner_r, 1.0 - corner_r) # top-right
        ]
        
        for x, y in corner_positions:
            circles.append([x, y, corner_r])
        
        # Place medium circles along edges
        edge_r = 0.08
        # Bottom edge
        circles.append([0.5, edge_r, edge_r])
        # Top edge  
        circles.append([0.5, 1.0 - edge_r, edge_r])
        # Left edge
        circles.append([edge_r, 0.5, edge_r])
        # Right edge
        circles.append([1.0 - edge_r, 0.5, edge_r])
        
        # Fill remaining space with smaller circles in a grid pattern
        remaining = n_circles - len(circles)
        small_r = 0.05
        
        # Create a 4x5 grid for the remaining circles in the center region
        grid_x = np.linspace(0.25, 0.75, 4)
        grid_y = np.linspace(0.25, 0.75, 5)
        
        for i, x in enumerate(grid_x):
            for j, y in enumerate(grid_y):
                if len(circles) >= n_circles:
                    break
                # Offset every other row slightly for better packing
                x_pos = x + (0.02 if j % 2 == 1 else 0.0)
                circles.append([x_pos, y, small_r])
        
        # Fill any remaining slots
        while len(circles) < n_circles:
            circles.append([0.5 + np.random.uniform(-0.2, 0.2), 
                          0.5 + np.random.uniform(-0.2, 0.2), 
                          MIN_RADIUS])
        
        return np.array(circles[:n_circles])
    
    # Configuration 2: Optimized hexagonal pattern
    def create_hex_config():
        circles = []
        r_hex = 0.075  # Slightly larger than previous attempt
        
        # 5 rows: [5, 5, 5, 5, 6] configuration
        rows = [5, 5, 5, 5, 6]
        y_spacing = r_hex * np.sqrt(3)
        
        total_height = (len(rows) - 1) * y_spacing + 2 * r_hex
        start_y = (1.0 - total_height) / 2.0 + r_hex
        
        for row_idx, num_cols in enumerate(rows):
            y = start_y + row_idx * y_spacing
            
            # Alternate row offset for hexagonal packing
            x_offset = r_hex if row_idx % 2 == 1 else 0.0
            x_spacing = 2 * r_hex
            
            total_width = (num_cols - 1) * x_spacing + 2 * r_hex
            start_x = (1.0 - total_width) / 2.0 + r_hex + x_offset
            
            for col_idx in range(num_cols):
                if len(circles) >= n_circles:
                    break
                x = start_x + col_idx * x_spacing
                circles.append([x, y, r_hex])
        
        return np.array(circles[:n_circles])
    
    # Configuration 3: Size-graded configuration
    def create_graded_config():
        circles = []
        
        # Large circles (4 circles, r ≈ 0.15)
        large_r = 0.15
        large_positions = [(0.15, 0.15), (0.85, 0.15), (0.15, 0.85), (0.85, 0.85)]
        for x, y in large_positions:
            circles.append([x, y, large_r])
        
        # Medium circles (8 circles, r ≈ 0.09)  
        medium_r = 0.09
        medium_positions = [
            (0.5, 0.09), (0.09, 0.5), (0.91, 0.5), (0.5, 0.91),  # edges
            (0.3, 0.3), (0.7, 0.3), (0.3, 0.7), (0.7, 0.7)       # inner positions
        ]
        for x, y in medium_positions:
            circles.append([x, y, medium_r])
        
        # Small circles (14 circles, r ≈ 0.04)
        small_r = 0.04
        remaining = n_circles - len(circles)
        
        # Fill remaining space with small circles
        for i in range(remaining):
            # Use a quasi-random distribution
            x = 0.2 + 0.6 * ((i * 0.618033988749895) % 1.0)  # Golden ratio spacing
            y = 0.2 + 0.6 * ((i * 0.381966011250105) % 1.0)  
            circles.append([x, y, small_r])
        
        return np.array(circles[:n_circles])
    
    # Add the three base configurations
    for config_func in [create_corner_config, create_hex_config, create_graded_config]:
        config = config_func()
        initial_pop.append(np.hstack((config[:,0], config[:,1], config[:,2])))
        
        # Add perturbed version
        perturbed = config.copy()
        perturbed[:, :2] += np.random.uniform(-0.015, 0.015, size=(n_circles, 2))
        perturbed[:, 2] += np.random.uniform(-0.01, 0.01, size=n_circles)
        
        # Ensure bounds
        for i in range(n_circles):
            r = np.clip(perturbed[i, 2], MIN_RADIUS, MAX_RADIUS)
            perturbed[i, 0] = np.clip(perturbed[i, 0], r, 1.0 - r)
            perturbed[i, 1] = np.clip(perturbed[i, 1], r, 1.0 - r)
            perturbed[i, 2] = r
        
        initial_pop.append(np.hstack((perturbed[:,0], perturbed[:,1], perturbed[:,2])))
    
    # Add random configurations with size diversity
    for _ in range(4):
        # Create size-diverse random configuration
        sizes = np.concatenate([
            np.full(4, 0.12),  # 4 large circles
            np.full(8, 0.07),  # 8 medium circles  
            np.full(14, 0.03)  # 14 small circles
        ])
        np.random.shuffle(sizes)
        
        positions = np.random.uniform(0.05, 0.95, size=(n_circles, 2))
        
        # Ensure position bounds given radii
        for i in range(n_circles):
            r = sizes[i]
            positions[i, 0] = np.clip(positions[i, 0], r, 1.0 - r)
            positions[i, 1] = np.clip(positions[i, 1], r, 1.0 - r)
        
        initial_pop.append(np.hstack((positions[:,0], positions[:,1], sizes)))
    
    return np.array(initial_pop)


def circle_packing26() -> np.ndarray:
    """
    Places 26 non-overlapping circles in the unit square in order to maximize the sum of radii.

    Returns:
        circles: np.array of shape (26,3), where the i-th row (x,y,r) stores the (x,y) coordinates of the i-th circle of radius r.
    """
    np.random.seed(42) # For reproducibility

    # Define bounds for x, y, r
    # params are [x1..xN, y1..yN, r1..rN]
    # So N_CIRCLES x-bounds, N_CIRCLES y-bounds, N_CIRCLES r-bounds
    bounds = [(MIN_RADIUS, 1.0 - MIN_RADIUS) for _ in range(N_CIRCLES)] * 2 # x and y bounds
    bounds += [(MIN_RADIUS, MAX_RADIUS) for _ in range(N_CIRCLES)] # r bounds

    # Generate an initial population for differential_evolution
    initial_pop = _initial_population_multiscale(N_CIRCLES)
    
    # --- Step 1: Global Optimization with Differential Evolution ---
    de_result = differential_evolution(
        _objective_function,
        bounds,
        init=initial_pop, # Provide our structured initial population
        maxiter=4000, # Increased iterations for better convergence
        popsize=40,   # Increased population size for better exploration
        mutation=(0.3, 0.9), # Slightly more conservative mutation
        recombination=0.8, # Higher crossover probability  
        strategy='best1bin', # Common strategy
        seed=42, # For reproducibility
        workers=-1, # Use all available CPU cores
        disp=False, # Set to True to see progress
        polish=False, # Don't polish yet, save for SLSQP
        tol=0.0001, # Tighter tolerance for global search
        atol=0.0001
    )

    best_params_de = de_result.x
    
    # --- Step 2: Local Optimization with SLSQP (Sequential Least Squares Programming) ---
    # This refines the solution found by differential_evolution.
    # SLSQP can handle explicit constraints, which are more precise than penalty methods
    # for final convergence.

    # Define constraints for SLSQP using numba-compatible functions
    slsqp_constraints = []

    # 1. Boundary constraints: r <= x, r <= y, x+r <= 1, y+r <= 1
    # Form: fun(x) >= 0 for inequality constraints
    for i in range(N_CIRCLES):
        for constraint_type in range(5):  # 5 different boundary constraint types
            slsqp_constraints.append({
                'type': 'ineq', 
                'fun': lambda p, idx=i, ctype=constraint_type: _boundary_constraint(p, idx, ctype)
            })

    # 2. Overlap constraints: (x_i - x_j)^2 + (y_i - y_j)^2 - (r_i + r_j)^2 >= 0
    for i in range(N_CIRCLES):
        for j in range(i + 1, N_CIRCLES):
            slsqp_constraints.append({
                'type': 'ineq', 
                'fun': lambda p, i=i, j=j: _overlap_constraint(p, i, j)
            })

    # Objective function for SLSQP is simply -sum(radii)
    # The constraints handle the validity.
    def _slsqp_objective(params: np.ndarray) -> float:
        return -np.sum(params[2*N_CIRCLES:]) # Sum of radii

    # Run SLSQP from the DE result with adaptive parameters
    slsqp_result = minimize(
        _slsqp_objective,
        best_params_de, # Start from the best DE solution
        method='SLSQP',
        bounds=bounds,
        constraints=slsqp_constraints,
        options={
            'ftol': 1e-10,      # Very tight function tolerance
            'maxiter': 3000,    # More iterations for complex constraints
            'disp': False,
            'eps': 1e-8         # Smaller step size for gradient estimation
        }
    )

    # If SLSQP failed or found a worse solution, fall back to DE result
    final_params = slsqp_result.x
    if not slsqp_result.success or _slsqp_objective(slsqp_result.x) > _slsqp_objective(best_params_de):
        # Fallback to DE result if SLSQP did not improve or failed
        final_params = best_params_de
        # print("SLSQP did not improve or failed, using DE result.")
    # else:
        # print(f"SLSQP improved objective from {-_slsqp_objective(best_params_de):.6f} to {-_slsqp_objective(slsqp_result.x):.6f}")


    # Unpack the final parameters into circles format (x, y, r)
    final_circles = np.zeros((N_CIRCLES, 3))
    final_circles[:, 0] = final_params[:N_CIRCLES]
    final_circles[:, 1] = final_params[N_CIRCLES:2*N_CIRCLES]
    final_circles[:, 2] = final_params[2*N_CIRCLES:]

    # Ensure radii are at least MIN_RADIUS and within MAX_RADIUS (final clip)
    final_circles[:, 2] = np.clip(final_circles[:, 2], MIN_RADIUS, MAX_RADIUS)
    # Ensure positions are within bounds, accounting for radius (final clip)
    for i in range(N_CIRCLES):
        r = final_circles[i, 2]
        final_circles[i, 0] = np.clip(final_circles[i, 0], r, 1.0 - r)
        final_circles[i, 1] = np.clip(final_circles[i, 1], r, 1.0 - r)


    # Final check of violations (for debugging/validation)
    final_violations = _calculate_violations(final_circles)
    if final_violations > 1e-7: # A small tolerance for floating point errors
        print(f"Warning: Final solution has residual violations: {final_violations}")
        # If there are significant violations, it might mean the SLSQP didn't converge perfectly
        # or the penalty factor was too low for DE.
        # For now, we accept it as the best effort.
        # It's also possible that the final clip slightly moved circles into violation,
        # but this is a common post-processing step to ensure strict bounds.


    return final_circles


# EVOLVE-BLOCK-END
