# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import differential_evolution, minimize, minimize_scalar
# Removed shapely.geometry.Polygon - will use Numba-accelerated SAT instead
from numba import njit # Added numba for JIT compilation

# --- Numba-accelerated Geometric Primitives & Objective Components ---

# Pre-calculate base hexagon angles (from Inspiration 1 & 3)
_base_hexagon_angles = np.arange(6, dtype=np.float64) * np.pi / 3
UNIT_HEX_SIDE_LENGTH = 1.0 # Define as a constant for clarity and potential Numba usage (from Insp1/3)
SQRT3 = np.sqrt(3.0) # Define as a constant for clarity and Numba usage (from Insp1/3)
EPSILON_SAT = 1e-9 # Epsilon for SAT collision detection and floating point robustness (from Insp1/3)

@njit(nopython=True, cache=True) # JIT compilation for performance (from Inspiration 3, added nopython=True)
def get_hexagon_vertices(center_x, center_y, angle_rad, side_length=UNIT_HEX_SIDE_LENGTH): # Use constant
    """Returns a (6, 2) numpy array of hexagon vertices."""
    vertices = np.empty((6, 2), dtype=np.float64)
    angles = _base_hexagon_angles + angle_rad # Use pre-calculated base angles
    for i in range(6):
        vertices[i, 0] = center_x + side_length * np.cos(angles[i])
        vertices[i, 1] = center_y + side_length * np.sin(angles[i])
    return vertices

@njit(nopython=True, cache=True) # JIT compilation for performance (from Inspiration 3, added nopython=True)
def calculate_outer_hexagon_side_length(all_vertices):
    """
    Calculates the side length of the minimal enclosing regular hexagon
    centered at (0,0) with a flat top/bottom, given a set of vertices.
    """
    if all_vertices.shape[0] == 0:
        return 0.0
    
    # Use global constant SQRT3 (from Insp1/3)
    term1 = np.max(np.abs(all_vertices[:, 1]) * 2.0 / SQRT3) # Added .0 for float literal, consistent with Insp1/3
    term2 = np.max(np.abs(all_vertices[:, 0]) + np.abs(all_vertices[:, 1]) / SQRT3)
    
    return max(term1, term2)

# --- Angle-Optimized Outer Hexagon Calculation (from Inspirations 1 & 2) ---

@njit(nopython=True, cache=True)
def _calculate_max_R_from_rotated_vertices(rotated_vertices):
    """
    Calculates minimum outer hexagon side length for a set of points, assuming
    outer hex is centered at (0,0) and flat-topped (after point rotation).
    This is the core calculation used by the angle optimizer.
    """
    max_R = 0.0
    # The side length R for a single point (x,y) is max(|y|*2/sqrt(3), |x|+|y|/sqrt(3)).
    # We can check these two terms for all vertices to find the overall maximum.
    term1_max = 0.0
    term2_max = 0.0
    for i in range(rotated_vertices.shape[0]):
        vx, vy = rotated_vertices[i, 0], rotated_vertices[i, 1]
        abs_vy = np.abs(vy)
        term1_max = max(term1_max, abs_vy * 2.0 / SQRT3)
        term2_max = max(term2_max, np.abs(vx) + abs_vy / SQRT3)
    
    return max(term1_max, term2_max)

def get_outer_hex_side_length_optimized_angle(all_vertices):
    """
    Calculates the side length of the minimum enclosing regular hexagon by
    optimizing for the outer hexagon's rotation angle using minimize_scalar.
    (Adopted from Inspirations 1 & 2).
    """
    if all_vertices.shape[0] == 0:
        return 0.0, 0.0 # Return side_length and angle

    # Define an inner function for minimize_scalar
    def _get_R_for_angle(angle_deg):
        # We rotate the points in the opposite direction of the hexagon's rotation
        angle_rad = np.deg2rad(-angle_deg)
        cos_a, sin_a = np.cos(angle_rad), np.sin(angle_rad)
        
        # Manually create rotation matrix and apply it for Numba compatibility
        rotated_vertices = np.empty_like(all_vertices)
        rotated_vertices[:, 0] = all_vertices[:, 0] * cos_a - all_vertices[:, 1] * sin_a
        rotated_vertices[:, 1] = all_vertices[:, 0] * sin_a + all_vertices[:, 1] * cos_a
        
        return _calculate_max_R_from_rotated_vertices(rotated_vertices)
    
    # Search for optimal outer hex rotation in range [0, 60) degrees due to 6-fold symmetry
    res_angle = minimize_scalar(
        _get_R_for_angle, 
        bounds=(0, 60), 
        method='bounded', 
        options={'xatol': EPSILON_SAT}
    )
    
    return res_angle.fun, res_angle.x # Return both the side length and the optimal angle

# --- Numba-accelerated SAT Collision Detection (from Inspiration 3) ---

@njit(nopython=True, cache=True) # Added nopython=True
def _project_polygon(vertices, axis):
    """Projects polygon vertices onto an axis and returns min/max scalars."""
    min_proj = np.dot(vertices[0], axis)
    max_proj = min_proj
    for i in range(1, 6):
        proj = np.dot(vertices[i], axis)
        if proj < min_proj:
            min_proj = proj
        elif proj > max_proj:
            max_proj = proj
    return min_proj, max_proj

@njit(nopython=True, cache=True) # Added nopython=True
def _check_overlap_sat(hex1_vertices, hex2_vertices):
    """
    Checks for overlap between two convex hexagons using the Separating Axis Theorem (SAT).
    Returns True if overlaps, False otherwise. Uses EPSILON_SAT for "touching". (from Insp1/3)
    """
    axes = np.empty((6, 2), dtype=np.float64)
    
    # Axes from hex1 (3 unique normals)
    for i in range(3):
        p1 = hex1_vertices[i]
        p2 = hex1_vertices[i + 1]
        edge = p2 - p1
        axes[i, 0] = -edge[1]
        axes[i, 1] = edge[0]
    
    # Axes from hex2 (3 unique normals)
    for i in range(3):
        p1 = hex2_vertices[i]
        p2 = hex2_vertices[i + 1]
        edge = p2 - p1
        axes[i+3, 0] = -edge[1]
        axes[i+3, 1] = edge[0]

    for i in range(6):
        axis = axes[i]
        norm = np.sqrt(axis[0]**2 + axis[1]**2)
        if norm < EPSILON_SAT: # Use constant EPSILON_SAT (from Insp1/3)
            continue # Avoid division by zero for degenerate edges
        axis /= norm

        min1, max1 = _project_polygon(hex1_vertices, axis)
        min2, max2 = _project_polygon(hex2_vertices, axis)

        # Check for separation along this axis (with a small epsilon for floating point robustness)
        if max1 < min2 + EPSILON_SAT or max2 < min1 + EPSILON_SAT: # Use constant EPSILON_SAT (from Insp1/3)
            return False  # Found a separating axis, no overlap

    return True  # No separating axis found, polygons overlap

# --- Numba-accelerated Objective Function ---

def fitness_function(params, n_hexagons):
    """
    Objective function to be minimized. It returns the outer hexagon side length
    for a given configuration, with a large penalty for overlaps.
    This version dynamically centers the packing before evaluation, making the
    objective function translation-invariant. It uses Numba-accelerated
    geometric primitives and SAT for collision detection (from Inspiration 3).
    """
    params_reshaped = params.reshape((n_hexagons, 3))

    # --- Dynamic Centering ---
    center_coords = params_reshaped[:, :2]
    centroid = np.mean(center_coords, axis=0)
    centered_coords = center_coords - centroid
    
    # Generate all hexagon vertices using the Numba-accelerated function
    all_vertices_list = [
        get_hexagon_vertices(cx, cy, angle) 
        for (cx, cy), angle in zip(centered_coords, params_reshaped[:, 2])
    ]

    # Constraint check: No overlaps between any pair of hexagons
    for i in range(n_hexagons):
        for j in range(i + 1, n_hexagons):
            # Bounding circle pre-check (from Inspiration Program 1)
            # Two unit hexagons (side_length=1) can only overlap if their centers are closer than 2*side_length = 2.
            # (2*side_length)^2 = 4. Subtract a small epsilon for floating point safety.
            c1 = centered_coords[i]
            c2 = centered_coords[j]
            dist_sq = (c1[0] - c2[0])**2 + (c1[1] - c2[1])**2
            max_center_dist_sq = (2 * UNIT_HEX_SIDE_LENGTH - EPSILON_SAT)**2 # Use constants (from Insp1/3)
            if dist_sq >= max_center_dist_sq: 
                continue # No overlap, skip detailed SAT check

            # If bounding circles overlap, perform the more precise SAT check
            if _check_overlap_sat(all_vertices_list[i], all_vertices_list[j]):
                return 1e9 # Large penalty for invalid configurations

    # Objective: Calculate the side length of the enclosing hexagon
    all_vertices_stacked = np.vstack(all_vertices_list)
    return calculate_outer_hexagon_side_length(all_vertices_stacked)

# --- Utility for generating a good initial population (from Inspiration Program 1) ---
def generate_initial_population_11_hexagons(popsize, bounds, n_hexagons):
    """
    Generates a diverse initial population for 11 hexagons, combining several strategies:
    1. A '1+6+4' pattern 
    2. A '3-5-3' pattern with a rotated central hex
    3. Perturbed versions of these smart guesses
    4. Purely random individuals within bounds
    Returns a numpy array of shape (popsize, n_variables).
    """
    n_variables = n_hexagons * 3
    initial_population = np.zeros((popsize, n_variables), dtype=np.float64)
    
    # --- Smart Guess 1: 1+6+4 pattern ---
    guess_1_params = np.zeros((n_hexagons, 3))
    guess_1_params[0, :] = [0, 0, 0] # Central hexagon
    
    # First ring of 6 hexagons
    ring_dist_1 = SQRT3 * UNIT_HEX_SIDE_LENGTH
    for i in range(6):
        angle = i * np.pi / 3
        guess_1_params[i+1, :] = [ring_dist_1 * np.cos(angle), ring_dist_1 * np.sin(angle), 0]

    # Outer 4 hexagons (common for N=11 optimal results)
    rotation_outer = np.pi / 6 
    outer_dist = 3.0 * UNIT_HEX_SIDE_LENGTH 
    guess_1_params[7, :] = [-outer_dist, 0, rotation_outer] 
    guess_1_params[8, :] = [outer_dist, 0, rotation_outer]  
    guess_1_params[9, :] = [0, outer_dist * SQRT3 / 2, rotation_outer] 
    guess_1_params[10, :] = [0, -outer_dist * SQRT3 / 2, rotation_outer] 
    
    initial_population[0] = guess_1_params.flatten()

    # --- Smart Guess 2: 3-5-3 pattern (from previous target iterations) ---
    guess_2_params = np.zeros((n_hexagons, 3))
    
    d_c = SQRT3 * UNIT_HEX_SIDE_LENGTH
    row_v_dist = 1.5 * UNIT_HEX_SIDE_LENGTH
    
    guess_2_params[0, :] = [-d_c, row_v_dist, 0]
    guess_2_params[1, :] = [0, row_v_dist, 0]
    guess_2_params[2, :] = [d_c, row_v_dist, 0]
    guess_2_params[3, :] = [-2*d_c, 0, 0]
    guess_2_params[4, :] = [-d_c, 0, 0]
    guess_2_params[5, :] = [0, 0, np.pi/6] # Central hex rotated by 30 degrees
    guess_2_params[6, :] = [d_c, 0, 0]
    guess_2_params[7, :] = [2*d_c, 0, 0]
    guess_2_params[8, :] = [-d_c, -row_v_dist, 0]
    guess_2_params[9, :] = [0, -row_v_dist, 0]
    guess_2_params[10, :] = [d_c, -row_v_dist, 0]

    if popsize > 1:
        initial_population[1] = guess_2_params.flatten()
    
    # --- Perturbed versions of smart guesses and random individuals ---
    num_smart_guesses = 2 if popsize > 1 else 1
    num_perturbed = (popsize - num_smart_guesses) // 2
    
    perturb_scale_xy = 0.75 # Larger perturbation for more exploration
    perturb_scale_angle = np.pi / 6 # Larger perturbation for more exploration
    
    for i in range(num_smart_guesses, num_smart_guesses + num_perturbed):
        base_guess_idx = (i % num_smart_guesses) # Alternate between smart guesses
        base_guess = initial_population[base_guess_idx].copy() 
        perturbed_guess = base_guess.reshape((n_hexagons, 3))
        for j in range(n_hexagons):
            perturbed_guess[j, 0] += np.random.uniform(-perturb_scale_xy, perturb_scale_xy)
            perturbed_guess[j, 1] += np.random.uniform(-perturb_scale_xy, perturb_scale_xy)
            perturbed_guess[j, 2] = np.fmod(perturbed_guess[j, 2] + np.random.uniform(-perturb_scale_angle, perturb_scale_angle), 2 * np.pi)
            if perturbed_guess[j, 2] < 0: # Ensure angle is non-negative
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
    Constructs an optimized packing of 11 disjoint unit regular hexagons using a
    Numba-accelerated engine and a rigorous hybrid optimization strategy
    to beat the SOTA benchmark.
    """
    n_hexagons = 11
    
    # Define search space bounds for 33 variables (11 hexagons * [x, y, angle_rad])
    bounds = []
    for _ in range(n_hexagons):
        bounds.extend([(-8, 8), (-8, 8), (0, 2 * np.pi)])

    # Generate the initial population using the new sophisticated strategy (from Insp1/3)
    popsize = 200 # Increased popsize for better global search (from Insp1/3)
    np.random.seed(42) # Seed numpy's random generator for reproducibility
    initial_population = generate_initial_population_11_hexagons(popsize, bounds, n_hexagons)

    # --- Stage 1: Global Optimization with Differential Evolution ---
    # print("Starting global optimization (Differential Evolution)...") # Re-enable for debugging if needed (from Insp1/3)
    de_result = differential_evolution(
        fitness_function,
        bounds,
        args=(n_hexagons,),
        maxiter=20000,       # Increased for most exhaustive global search (from Insp 3)
        popsize=popsize,     # Large population size for diversity
        seed=42,             # For reproducible DE algorithm behavior
        workers=-1,          # Use all available CPU cores for parallelization
        updating='deferred', # Recommended for parallel execution
        init=initial_population, # Provide our custom, diversified initial population
        tol=1e-6,            # Tight tolerance for DE convergence
        atol=1e-6,           # Tight absolute tolerance for DE convergence
        disp=False,          # Set to False to reduce console output and slightly improve speed
        mutation=(0.8, 1.0), # Tuned mutation for aggressive exploration
        recombination=0.9,   # Tuned recombination for good mixing
        polish=True          # Use default polishing, consistent with best inspiration (Insp 3)
    )
    
    if de_result.fun >= 1e9:
        raise RuntimeError(f"Global optimization (DE) failed to find a valid solution. Best fun: {de_result.fun}")

    # --- Stage 2: Local Refinement with L-BFGS-B ---
    # print(f"Starting local optimization from DE best result (objective: {de_result.fun:.6f})...") # Re-enable for debugging if needed (from Insp1/3)
    local_result = minimize(
        fitness_function,
        de_result.x, # Start local search from the best DE solution
        args=(n_hexagons,),
        method='L-BFGS-B',
        bounds=bounds,
        options={'ftol': 1e-15, 'gtol': 1e-10, 'maxiter': 15000} # Increased for most thorough local refinement (from Insp 3)
    )
    
    # --- Select the Best Result from the Two Stages ---
    if local_result.success and local_result.fun < de_result.fun:
        final_params = local_result.x
        # print(f"Local optimization improved result from {de_result.fun:.6f} to {local_result.fun:.6f}") # Re-enable for debugging if needed (from Insp1/3)
    else:
        final_params = de_result.x
        # print(f"Local optimization did not improve result, keeping DE's best ({de_result.fun:.6f})") # Re-enable for debugging if needed (from Insp1/3)

    # Final check to ensure the chosen solution is valid
    if fitness_function(final_params, n_hexagons) >= 1e9:
        raise RuntimeError(f"Final optimization failed to find a valid non-overlapping solution.")

    # --- Post-processing for Final Output ---
    inner_hex_data = final_params.reshape((n_hexagons, 3))
    
    # Recenter the entire packing at the origin for consistency in output,
    # as the fitness function already handles dynamic centering.
    center_of_mass = np.mean(inner_hex_data[:, :2], axis=0)
    inner_hex_data[:, :2] -= center_of_mass
    
    # Recalculate the final, accurate side length and optimal outer hex angle after centering.
    # This uses the angle-optimized calculation from Inspirations 1 & 2.
    final_vertices = np.vstack([
        get_hexagon_vertices(cx, cy, angle_rad) for cx, cy, angle_rad in inner_hex_data
    ])
    outer_hex_side_length, outer_hex_angle_deg = get_outer_hex_side_length_optimized_angle(final_vertices)

    # Convert inner hex angles from radians to degrees for the specified output format.
    inner_hex_data[:, 2] = np.rad2deg(inner_hex_data[:, 2])

    # The outer hexagon is canonically centered at (0,0) with its optimized angle.
    outer_hex_data = np.array([0, 0, outer_hex_angle_deg])
    
    return inner_hex_data, outer_hex_data, outer_hex_side_length
# EVOLVE-BLOCK-END