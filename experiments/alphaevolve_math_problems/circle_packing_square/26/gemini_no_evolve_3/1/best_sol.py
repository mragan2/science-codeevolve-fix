# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize, Bounds
from numba import njit
import time # Added for potential timing, though not directly used in the final return.

# Numba-accelerated function for pairwise squared distances
@njit(cache=True)
def _pairwise_distances_sq(coords):
    n = coords.shape[0]
    dist_sq = np.zeros((n, n), dtype=coords.dtype)
    for i in range(n):
        for j in range(i + 1, n):
            dx = coords[i, 0] - coords[j, 0]
            dy = coords[i, 1] - coords[j, 1]
            d_sq = dx*dx + dy*dy
            dist_sq[i, j] = d_sq
            dist_sq[j, i] = d_sq
    return dist_sq

def objective_function(params):
    """
    Objective function for scipy.optimize.minimize.
    params: flattened array [x1, y1, r1, x2, y2, r2, ...]
    Maximizing sum of radii is equivalent to minimizing negative sum of radii.
    """
    radii = params[2::3]
    return -np.sum(radii)

def all_constraints_func(params, n_circles):
    """
    Returns an array of all inequality constraint values, g(x) >= 0.
    """
    x = params[0::3]
    y = params[1::3]
    r = params[2::3]

    constraints_values = []

    # Boundary constraints: r <= x <= 1-r and r <= y <= 1-r
    # Equivalent to: x - r >= 0, 1 - x - r >= 0, y - r >= 0, 1 - y - r >= 0
    constraints_values.extend(x - r)
    constraints_values.extend(1.0 - x - r)
    constraints_values.extend(y - r)
    constraints_values.extend(1.0 - y - r)

    # Non-overlap constraints: (xi-xj)^2 + (yi-yj)^2 >= (ri + rj)^2
    coords = params.reshape(-1, 3)[:, :2]
    
    # Use numba-accelerated distance calculation
    dist_sq = _pairwise_distances_sq(coords)

    for i in range(n_circles):
        for j in range(i + 1, n_circles):
            # Calculate (ri + rj)^2
            min_dist_sq = (r[i] + r[j])**2
            current_dist_sq = dist_sq[i, j]
            constraints_values.append(current_dist_sq - min_dist_sq)

    return np.array(constraints_values)

def circle_packing26() -> np.ndarray:
    n_circles = 26
    
    # Initial guess: A quasi-hexagonal pattern (5,6,5,6,4 circles per row)
    # This pattern is known to be a good starting point for N=26.
    r_base = 0.08 # A reasonable initial radius for the pattern's internal structure
    centers_initial = []
    
    rows_pattern = [5, 6, 5, 6, 4]
    
    # Step 1: Generate relative coordinates for the hexagonal pattern
    y_current_relative = r_base
    for row_idx, count in enumerate(rows_pattern):
        # Stagger x-offset for alternating rows for hexagonal packing.
        # Rows with even indices (0, 2, 4) start at x=r_base.
        # Rows with odd indices (1, 3) start at x=r_base * 2 (offset by r_base).
        x_start_relative = r_base if row_idx % 2 == 0 else r_base * 2
        
        for i in range(count):
            centers_initial.append([x_start_relative + i * 2 * r_base, y_current_relative, r_base])
        
        y_current_relative += np.sqrt(3) * r_base # Hexagonal row height

    # Step 2: Scale and center the initial pattern within the [0,1] square
    all_x_rel = np.array([c[0] for c in centers_initial])
    all_y_rel = np.array([c[1] for c in centers_initial])
    
    min_x_rel = np.min(all_x_rel) - r_base # Leftmost edge of bounding box
    max_x_rel = np.max(all_x_rel) + r_base # Rightmost edge
    min_y_rel = np.min(all_y_rel) - r_base # Bottommost edge
    max_y_rel = np.max(all_y_rel) + r_base # Topmost edge
    
    width_rel = max_x_rel - min_x_rel
    height_rel = max_y_rel - min_y_rel
    
    # Determine scale factor to fit the pattern's bounding box into [0,1]
    scale_factor = 1.0 / max(width_rel, height_rel)
    
    # Apply scaling to positions and radii
    scaled_centers = []
    for cx, cy, cr in centers_initial:
        scaled_r = cr * scale_factor
        scaled_cx = (cx - min_x_rel) * scale_factor # Shift to start from 0, then scale
        scaled_cy = (cy - min_y_rel) * scale_factor
        scaled_centers.append([scaled_cx, scaled_cy, scaled_r])
    
    # After scaling, the pattern is now in [0,1]x[0,1] but its bounding box might not be centered.
    # Recalculate bounding box for centering to ensure it's truly centered in the unit square.
    all_x_scaled = np.array([c[0] for c in scaled_centers])
    all_y_scaled = np.array([c[1] for c in scaled_centers])
    
    # The radii are all the same at this point, so use any radius for bounding box calculation.
    current_r_scaled = scaled_centers[0][2] 
    
    min_x_final = np.min(all_x_scaled) - current_r_scaled
    max_x_final = np.max(all_x_scaled) + current_r_scaled
    min_y_final = np.min(all_y_scaled) - current_r_scaled
    max_y_final = np.max(all_y_scaled) + current_r_scaled
    
    width_final = max_x_final - min_x_final
    height_final = max_y_final - min_y_final
    
    # Calculate final shifts to center the pattern in the 1x1 square
    x_shift_final = (1.0 - width_final) / 2.0 - min_x_final
    y_shift_final = (1.0 - height_final) / 2.0 - min_y_final
    
    initial_params = np.array(scaled_centers).flatten()
    
    # Apply final centering shifts
    for i in range(n_circles):
        initial_params[i*3] += x_shift_final
        initial_params[i*3+1] += y_shift_final
    
    # Ensure initial radii are positive (should be due to scaling, but as a safeguard)
    initial_params[2::3] = np.maximum(initial_params[2::3], 1e-6)

    # Define bounds for x, y, r
    param_bounds = []
    for i in range(n_circles):
        param_bounds.extend([(0.0, 1.0), (0.0, 1.0), (1e-6, 0.5)]) # (x, y, r)
    bounds = Bounds([b[0] for b in param_bounds], [b[1] for b in param_bounds])

    # Define constraints using the single function approach
    constraints = [{'type': 'ineq', 'fun': all_constraints_func, 'args': (n_circles,)}]

    # Run SLSQP optimization
    res = minimize(
        objective_function,
        initial_params,
        args=(), # No args for objective_function directly, n_circles passed to constraint
        method='SLSQP',
        bounds=bounds,
        constraints=constraints,
        options={'ftol': 1e-8, 'maxiter': 10000, 'disp': False} # Increased maxiter and precision
    )
    
    if not res.success:
        # If optimization fails, print a message. The `res.x` will still contain the best
        # solution found before termination, which might still be good.
        print(f"SLSQP optimization failed: {res.message}")
        
    optimized_params = res.x
    
    # Reshape the output to (n_circles, 3)
    circles = optimized_params.reshape(n_circles, 3)

    return circles


# EVOLVE-BLOCK-END
