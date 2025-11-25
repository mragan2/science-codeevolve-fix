# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import differential_evolution, minimize, minimize_scalar
import numba

# --- Constants ---
UNIT_HEX_SIDE_LENGTH = 1.0
SQRT3 = np.sqrt(3.0)
N_HEX = 11 # For clarity and consistency
EPSILON = 1e-12 # Tighter tolerance for geometric comparisons and containment buffer (from Inspiration 2)

# Precompute trig values for speed in MERH and other calculations
# COS30 = SQRT3 / 2 # Not used in current implementation
# SIN30 = 0.5       # Not used in current implementation
# COS150 = -SQRT3 / 2 # Not explicitly used after refactor
# SIN150 = 0.5        # Not explicitly used after refactor

# --- Numba-accelerated Geometric Primitives & MERH Calculation ---
@numba.jit(nopython=True, cache=True)
def _calculate_hexagon_vertices_rad(center_x, center_y, angle_rad, side_length=UNIT_HEX_SIDE_LENGTH):
    """
    Numba-accelerated function to get hexagon vertices.
    angle_rad defines the angle of one of its vertices relative to the positive x-axis.
    """
    vertices = np.empty((6, 2), dtype=np.float64)
    for k in range(6):
        angle = angle_rad + k * np.pi / 3
        vertices[k, 0] = center_x + side_length * np.cos(angle)
        vertices[k, 1] = center_y + side_length * np.sin(angle)
    return vertices

@numba.jit(nopython=True, cache=True)
def _get_hexagon_axes_rad(angle_rad):
    """
    Returns 3 unique normal vectors for a hexagon.
    angle_rad defines the angle of one of its vertices relative to the positive x-axis.
    The normals to the sides are at angles: angle_rad + pi/6, angle_rad + pi/2, angle_rad + 5*pi/6.
    """
    axes_angles_rad = np.array([np.pi/6, np.pi/2, 5*np.pi/6], dtype=np.float64) + angle_rad
    axes = np.empty((3, 2), dtype=np.float64)
    axes[:, 0] = np.cos(axes_angles_rad)
    axes[:, 1] = np.sin(axes_angles_rad)
    return axes

@numba.jit(nopython=True, cache=True)
def _project_polygon_on_axis(vertices, axis):
    """Projects polygon vertices onto an axis and returns min/max scalar projections."""
    min_proj = np.dot(vertices[0], axis)
    max_proj = min_proj
    for i in range(1, vertices.shape[0]):
        proj = np.dot(vertices[i], axis)
        if proj < min_proj: min_proj = proj
        if proj > max_proj: max_proj = proj
    return min_proj, max_proj

@numba.jit(nopython=True, cache=True)
def check_hexagon_overlap_sat_rad(h1_x, h1_y, h1_angle_rad, h2_x, h2_y, h2_angle_rad, side_length=UNIT_HEX_SIDE_LENGTH):
    """
    Returns True if two hexagons overlap using the Separating Axis Theorem (SAT).
    Optimized with Numba. Angles are in radians.
    """
    # Quick bounding circle check: If distance between centers is >= 2 * circumradius, no overlap.
    # Circumradius of a unit hexagon is UNIT_HEX_SIDE_LENGTH.
    center_dist_sq = (h1_x - h2_x)**2 + (h1_y - h2_y)**2
    if center_dist_sq >= (2 * side_length - EPSILON)**2: 
        return False
    
    h1_vertices = _calculate_hexagon_vertices_rad(h1_x, h1_y, h1_angle_rad, side_length)
    h2_vertices = _calculate_hexagon_vertices_rad(h2_x, h2_y, h2_angle_rad, side_length)
    
    axes1 = _get_hexagon_axes_rad(h1_angle_rad)
    axes2 = _get_hexagon_axes_rad(h2_angle_rad)
    all_axes = np.vstack((axes1, axes2))
    
    for i in range(all_axes.shape[0]):
        axis = all_axes[i]
        min1, max1 = _project_polygon_on_axis(h1_vertices, axis)
        min2, max2 = _project_polygon_on_axis(h2_vertices, axis)
        # Check for overlap on the current axis (if projections do not overlap, polygons are separated)
        if max1 < min2 + EPSILON or max2 < min1 + EPSILON:
            return False
    return True

@numba.jit(nopython=True, cache=True)
def _calculate_max_R_for_fixed_outer_angle_numba(all_vertices, outer_angle_rad, SQRT3_val):
    """
    Numba-accelerated function to calculate the side length of the minimal enclosing regular hexagon
    centered at (0,0) with a *fixed* outer angle, given a set of vertices.
    Assumes all_vertices are already centered around (0,0).
    """
    if all_vertices.shape[0] == 0:
        return 0.0

    # Rotate vertices so the outer hexagon becomes 'flat-top' relative to our calculation axes
    cos_neg_angle = np.cos(-outer_angle_rad)
    sin_neg_angle = np.sin(-outer_angle_rad)
    
    rotated_vertices_x = all_vertices[:, 0] * cos_neg_angle - all_vertices[:, 1] * sin_neg_angle
    rotated_vertices_y = all_vertices[:, 0] * sin_neg_angle + all_vertices[:, 1] * cos_neg_angle
    
    max_R = 0.0
    for i in range(all_vertices.shape[0]):
        vx = rotated_vertices_x[i]
        vy = rotated_vertices_y[i]
        
        # These are the formulas for a point (vx, vy) to be inside a flat-top hexagon
        # centered at (0,0) with side length R. We find the minimum R for this point.
        R_candidate = max(
            np.abs(vy) * 2 / SQRT3_val,
            np.abs(SQRT3_val * vx - vy) / SQRT3_val,
            np.abs(SQRT3_val * vx + vy) / SQRT3_val
        )
        if R_candidate > max_R:
            max_R = R_candidate
    return max_R

def find_min_enclosing_hexagon_R(all_vertices):
    """
    Finds the side length of the minimum enclosing regular hexagon
    for a given set of vertices by optimizing the outer hexagon's rotation.
    This function wraps a numba-jitted calculation with scipy's minimize_scalar.
    """
    if all_vertices.shape[0] == 0:
        return 0.0
    
    # Define a local function to be minimized by scipy.optimize.minimize_scalar
    def get_R_for_outer_angle_rad(outer_angle_rad):
        # Pass numpy arrays to numba functions.
        return _calculate_max_R_for_fixed_outer_angle_numba(all_vertices, outer_angle_rad, SQRT3)

    # Optimize outer_angle_rad using minimize_scalar.
    # Search range for angle is [0, pi/3) (0 to 60 degrees) due to 6-fold rotational symmetry.
    res_angle = minimize_scalar(get_R_for_outer_angle_rad, bounds=(0, np.pi/3), method='bounded', tol=1e-7)
    
    # Add a small epsilon to ensure strict containment due to floating point precision.
    return res_angle.fun + EPSILON

# --- Objective Function for the Optimizer ---
def fitness_function(params, n_hexagons): # Added n_hexagons arg for generality
    """
    Objective function to be minimized. It returns the outer hexagon side length
    for a given configuration, with a large penalty for overlaps.
    This version dynamically centers the packing before evaluation, making the
    objective function translation-invariant, and optimizes outer hexagon orientation.
    """
    params_reshaped = params.reshape((n_hexagons, 3))

    # --- Dynamic Centering ---
    # Calculate centroid of all inner hexagon centers and subtract it
    # This makes the objective function translation-invariant
    center_coords = params_reshaped[:, :2]
    centroid = np.mean(center_coords, axis=0)
    
    # Store centered hexagon data for overlap and vertex generation
    centered_hex_data = np.empty_like(params_reshaped)
    centered_hex_data[:, :2] = center_coords - centroid
    centered_hex_data[:, 2] = params_reshaped[:, 2] # Angles remain the same (in radians)

    # Constraint check: No overlaps between any pair of hexagons
    # Using Numba-accelerated SAT check
    for i in range(n_hexagons):
        for j in range(i + 1, n_hexagons):
            if check_hexagon_overlap_sat_rad(centered_hex_data[i, 0], centered_hex_data[i, 1], centered_hex_data[i, 2],
                                             centered_hex_data[j, 0], centered_hex_data[j, 1], centered_hex_data[j, 2],
                                             UNIT_HEX_SIDE_LENGTH): # EPSILON is now a global constant used internally
                return 1e9 # Large penalty for invalid configurations

    # Objective: Calculate the side length of the minimum enclosing hexagon,
    # optimizing for its orientation.
    all_vertices_list = []
    for i in range(n_hexagons):
        # Use the Numba-accelerated vertex generation
        all_vertices_list.append(_calculate_hexagon_vertices_rad(
            centered_hex_data[i, 0], centered_hex_data[i, 1], centered_hex_data[i, 2], UNIT_HEX_SIDE_LENGTH
        ))
    
    if not all_vertices_list: # Handle case of 0 hexagons
        return 0.0
    
    all_vertices = np.vstack(all_vertices_list)
    
    # Find the side length of the minimal enclosing hexagon,
    # optimizing the outer hexagon's rotation.
    return find_min_enclosing_hexagon_R(all_vertices)

# --- Utility for generating a good initial population ---
def generate_initial_population_11_hexagons(popsize, bounds, n_hexagons):
    """
    Generates a diverse initial population for 11 hexagons, combining several strategies:
    1. A '1+6+4' pattern (from Inspiration 1)
    2. A '3-5-3' pattern with a rotated central hex (from Inspiration 2/3)
    3. Perturbed versions of these smart guesses
    4. Purely random individuals within bounds
    Returns a numpy array of shape (popsize, n_variables) where n_variables = n_hexagons * 3.
    """
    n_variables = n_hexagons * 3
    initial_population = np.zeros((popsize, n_variables), dtype=np.float64)
    
    # --- Smart Guess 1: 1+6+4 pattern (from Inspiration 1) ---
    guess_1_params = np.zeros((n_hexagons, 3))
    
    # Central hexagon (0,0)
    guess_1_params[0, :] = [0, 0, 0] 

    # First ring of 6 hexagons, centers at distance SQRT3 from origin (s*sqrt(3))
    # Angles are 0 for these for initial alignment (pointy-top orientation)
    ring_dist_1 = SQRT3 * UNIT_HEX_SIDE_LENGTH
    for i in range(6):
        angle = i * np.pi / 3
        guess_1_params[i+1, :] = [ring_dist_1 * np.cos(angle), ring_dist_1 * np.sin(angle), 0]

    # Outer 4 hexagons: Placed to fill out a more hexagonal shape.
    # These coordinates are chosen to be more aligned with the hexagonal axes
    # and are often rotated by 30 degrees (pi/6 radians).
    rotation_outer = np.pi / 6 
    outer_dist = 3.0 * UNIT_HEX_SIDE_LENGTH # A common optimal result for N=11 has some hexagons at ~3.0 distance.
    guess_1_params[7, :] = [-outer_dist, 0, rotation_outer] # Left
    guess_1_params[8, :] = [outer_dist, 0, rotation_outer]  # Right
    guess_1_params[9, :] = [0, outer_dist * SQRT3 / 2, rotation_outer] # Top (approx 2.598)
    guess_1_params[10, :] = [0, -outer_dist * SQRT3 / 2, rotation_outer] # Bottom (approx 2.598)
    
    initial_population[0] = guess_1_params.flatten()

    # --- Smart Guess 2: 3-5-3 pattern (from Inspiration 2/3) ---
    guess_2_params = np.zeros((n_hexagons, 3))
    
    d_c = SQRT3 * UNIT_HEX_SIDE_LENGTH # Distance between centers of adjacent axis-aligned hexagons
    row_v_dist = 1.5 * UNIT_HEX_SIDE_LENGTH # Vertical distance between rows for unit hexagons
    
    # Row 1 (3 hexagons)
    guess_2_params[0, :] = [-d_c, row_v_dist, 0]
    guess_2_params[1, :] = [0, row_v_dist, 0]
    guess_2_params[2, :] = [d_c, row_v_dist, 0]

    # Row 2 (5 hexagons)
    guess_2_params[3, :] = [-2*d_c, 0, 0]
    guess_2_params[4, :] = [-d_c, 0, 0]
    guess_2_params[5, :] = [0, 0, np.pi/6] # Central hex rotated by 30 degrees (pi/6 rad)
    guess_2_params[6, :] = [d_c, 0, 0]
    guess_2_params[7, :] = [2*d_c, 0, 0]

    # Row 3 (3 hexagons)
    guess_2_params[8, :] = [-d_c, -row_v_dist, 0]
    guess_2_params[9, :] = [0, -row_v_dist, 0]
    guess_2_params[10, :] = [d_c, -row_v_dist, 0]

    if popsize > 1:
        initial_population[1] = guess_2_params.flatten()
    
    # --- Perturbed versions of smart guesses and random individuals ---
    num_smart_guesses = 2 if popsize > 1 else 1 # How many smart guesses we start with
    num_perturbed = (popsize - num_smart_guesses) // 2 # Half of remaining population are perturbed
    
    perturb_scale_xy = 0.5
    perturb_scale_angle = np.pi / 8
    
    for i in range(num_smart_guesses, num_smart_guesses + num_perturbed):
        base_guess = initial_population[(i % num_smart_guesses)].copy() # Alternate between smart guesses
        perturbed_guess = base_guess.reshape((n_hexagons, 3))
        for j in range(n_hexagons):
            perturbed_guess[j, 0] += np.random.uniform(-perturb_scale_xy, perturb_scale_xy)
            perturbed_guess[j, 1] += np.random.uniform(-perturb_scale_xy, perturb_scale_xy)
            perturbed_guess[j, 2] = np.fmod(perturbed_guess[j, 2] + np.random.uniform(-perturb_scale_angle, perturb_scale_angle), 2 * np.pi)
            if perturbed_guess[j, 2] < 0:
                perturbed_guess[j, 2] += 2 * np.pi
        initial_population[i] = perturbed_guess.flatten()

    # The remaining individuals are purely random within bounds
    for i in range(num_smart_guesses + num_perturbed, popsize):
        for j, (low, high) in enumerate(bounds):
            initial_population[i, j] = np.random.uniform(low, high)
            
    return initial_population

# --- Main Constructor Function ---
def hexagon_packing_11():
    """ 
    Constructs an optimized packing of 11 disjoint unit regular hexagons inside 
    a larger regular hexagon by minimizing the outer hexagon's side length.
    
    Returns:
        inner_hex_data: np.ndarray of shape (11,3), with rows (x, y, angle_degrees).
        outer_hex_data: np.ndarray of shape (3,) for the outer hexagon (x,y,angle_degree).
        outer_hex_side_length: float, the side length of the optimized outer hexagon.
    """
    n_hexagons = N_HEX # Use the global constant for N_HEX
    num_variables = n_hexagons * 3 # Each hexagon has (x, y, angle_rad)

    # Define search space bounds for 33 variables (11 hexagons * [x, y, angle_rad])
    # Bounds are set slightly larger to allow exploration, considering benchmark R is ~3.93
    bounds = []
    for _ in range(n_hexagons):
        bounds.extend([(-8, 8), (-8, 8), (0, 2 * np.pi)]) # x,y range, angle in radians [0, 2pi)

    # Generate the initial population using the new sophisticated strategy
    popsize = 200 # Increased popsize for better global search (from Inspiration 1)
    np.random.seed(42) # Seed numpy's random generator for reproducibility
    initial_population = generate_initial_population_11_hexagons(popsize, bounds, n_hexagons)

    # Stage 1: Global search with Differential Evolution
    de_result = differential_evolution(
        fitness_function,
        bounds,
        args=(n_hexagons,), # Pass n_hexagons to the fitness function
        maxiter=18000,       # Increased for more thorough global search (from Inspiration 1)
        popsize=popsize,
        seed=42,             # For reproducible results of the DE algorithm itself
        workers=-1,          # Use all available CPU cores for parallelization
        updating='deferred',
        init=initial_population,
        tol=1e-6,            # Tighter tolerance for DE convergence (from Inspiration 1)
        atol=1e-6,           # Tighter absolute tolerance for DE convergence (from Inspiration 1)
        disp=False,          # Set to False to reduce console output and potentially improve speed
        mutation=(0.8, 1.0),
        recombination=0.9,
        polish=False         # Local polish done separately for more control
    )
    
    if de_result.fun >= 1e9: # Check against the high penalty value
        raise RuntimeError(f"Differential Evolution failed to find a valid non-overlapping solution. Best fun: {de_result.fun}")

    # Stage 2: Local refinement using the best solution from the global search
    # print(f"Starting local optimization from DE best result (objective: {de_result.fun:.6f})...") # Removed disp output
    local_result = minimize(
        fun=fitness_function,
        x0=de_result.x, # Start from the best solution found by DE
        args=(n_hexagons,), # Pass n_hexagons to the fitness function
        method='L-BFGS-B', # A robust method for bounded, gradient-free optimization
        bounds=bounds,
        options={'maxiter': 5000, 'ftol': 1e-13, 'gtol': 1e-11} # Tighter tolerances and increased maxiter for local search
    )
    
    # Choose the best result between global and local optimization
    if local_result.success and local_result.fun < de_result.fun:
        # print(f"Local optimization improved result from {de_result.fun:.6f} to {local_result.fun:.6f}") # Removed disp output
        final_params = local_result.x
        final_fun = local_result.fun
    else:
        # print(f"Local optimization did not improve result, keeping DE's best ({de_result.fun:.6f})") # Removed disp output
        final_params = de_result.x
        final_fun = de_result.fun
    
    if final_fun >= 1e9: # Check for the 'inf' penalty
        raise RuntimeError(f"Final optimization failed to find a valid non-overlapping solution. Best fun: {final_fun}")

    # --- Post-processing for Final Output ---
    inner_hex_data_rad = final_params.reshape((n_hexagons, 3))
    
    # Recenter inner hexagons for output consistency (centroid at (0,0))
    center_of_mass = np.mean(inner_hex_data_rad[:, :2], axis=0)
    inner_hex_data_rad[:, :2] -= center_of_mass

    # Recalculate the outer hexagon's side length and optimal angle for the final, centered configuration
    final_vertices_list = []
    for cx, cy, angle_rad in inner_hex_data_rad:
        final_vertices_list.append(_calculate_hexagon_vertices_rad(cx, cy, angle_rad, UNIT_HEX_SIDE_LENGTH))
    final_vertices = np.vstack(final_vertices_list)

    outer_hex_side_length = find_min_enclosing_hexagon_R(final_vertices) # This includes EPSILON
    
    # To get the outer_hex_angle_rad, we need to run minimize_scalar again on the final vertices
    # (find_min_enclosing_hexagon_R only returns the minimum R, not the angle)
    def get_R_for_angle_final(outer_angle_rad):
        return _calculate_max_R_for_fixed_outer_angle_numba(final_vertices, outer_angle_rad, SQRT3)
    
    res_angle_final = minimize_scalar(get_R_for_angle_final, bounds=(0, np.pi/3), method='bounded', tol=1e-7)
    final_outer_hex_angle_rad = res_angle_final.x

    # Prepare final output: convert angles to degrees
    inner_hex_data = inner_hex_data_rad.copy()
    inner_hex_data[:, 2] = np.rad2deg(inner_hex_data_rad[:, 2] % (2 * np.pi)) # Ensure angles are within [0, 360)
    
    # Outer hexagon is centered at (0,0) for consistent output, with its optimized rotation
    outer_hex_data = np.array([0, 0, np.rad2deg(final_outer_hex_angle_rad)]) 
    
    return inner_hex_data, outer_hex_data, outer_hex_side_length
# EVOLVE-BLOCK-END