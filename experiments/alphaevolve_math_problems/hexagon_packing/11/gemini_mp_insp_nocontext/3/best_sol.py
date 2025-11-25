# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import differential_evolution, minimize
from numba import njit

# --- Global Constants for Numba Compatibility and Clarity ---
_base_hexagon_angles = np.arange(6, dtype=np.float64) * np.pi / 3
UNIT_HEX_SIDE_LENGTH = 1.0
SQRT3 = np.sqrt(3.0)
EPSILON_SAT = 1e-9

# --- Geometric Primitives & Objective Components ---

@njit(nopython=True, cache=True)
def get_hexagon_vertices(center_x, center_y, angle_rad, side_length=UNIT_HEX_SIDE_LENGTH):
    """Returns a (6, 2) numpy array of hexagon vertices, optimized with Numba."""
    angles = _base_hexagon_angles + angle_rad
    vertices = np.empty((6, 2), dtype=np.float64)
    vertices[:, 0] = center_x + side_length * np.cos(angles)
    vertices[:, 1] = center_y + side_length * np.sin(angles)
    return vertices

@njit(nopython=True, cache=True)
def calculate_outer_hexagon_side_length(all_vertices):
    """
    Calculates the side length of the minimal enclosing regular hexagon
    centered at (0,0) with a flat top/bottom, given a set of vertices.
    Optimized with Numba.
    """
    if all_vertices.shape[0] == 0:
        return 0.0
    
    term1 = np.max(np.abs(all_vertices[:, 1]) * 2.0 / SQRT3)
    term2 = np.max(np.abs(all_vertices[:, 0]) + np.abs(all_vertices[:, 1]) / SQRT3)
    
    return max(term1, term2)

# --- Numba-accelerated SAT Collision Detection ---

@njit(nopython=True, cache=True)
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

@njit(nopython=True, cache=True)
def _check_overlap_sat(hex1_vertices, hex2_vertices):
    """
    Checks for overlap between two convex hexagons using the Separating Axis Theorem (SAT).
    Returns True if overlaps, False otherwise. Uses EPSILON_SAT for "touching".
    """
    axes = np.empty((6, 2), dtype=np.float64)
    
    for i in range(3):
        edge = hex1_vertices[i + 1] - hex1_vertices[i]
        axes[i, 0] = -edge[1]
        axes[i, 1] = edge[0]
    
    for i in range(3):
        edge = hex2_vertices[i + 1] - hex2_vertices[i]
        axes[i+3, 0] = -edge[1]
        axes[i+3, 1] = edge[0]

    for i in range(6):
        axis = axes[i]
        norm = np.sqrt(axis[0]**2 + axis[1]**2)
        if norm < EPSILON_SAT:
            continue 
        axis /= norm

        min1, max1 = _project_polygon(hex1_vertices, axis)
        min2, max2 = _project_polygon(hex2_vertices, axis)

        if max1 < min2 + EPSILON_SAT or max2 < min1 + EPSILON_SAT:
            return False

    return True

# --- Objective Function for the Optimizer ---

def fitness_function(params, n_hexagons):
    """
    Objective function to be minimized. It returns the outer hexagon side length
    for a given configuration, with a large penalty for overlaps.
    This version uses Numba-accelerated SAT for high-performance collision detection.
    """
    params_reshaped = params.reshape((n_hexagons, 3))

    center_coords = params_reshaped[:, :2]
    centroid = np.mean(center_coords, axis=0)
    centered_coords = center_coords - centroid
    
    all_vertices_list = [
        get_hexagon_vertices(cx, cy, angle) 
        for (cx, cy), angle in zip(centered_coords, params_reshaped[:, 2])
    ]

    max_center_dist_sq = (2 * UNIT_HEX_SIDE_LENGTH - EPSILON_SAT)**2 
    for i in range(n_hexagons):
        for j in range(i + 1, n_hexagons):
            c1 = centered_coords[i]
            c2 = centered_coords[j]
            dist_sq = (c1[0] - c2[0])**2 + (c1[1] - c2[1])**2
            if dist_sq >= max_center_dist_sq: 
                continue

            if _check_overlap_sat(all_vertices_list[i], all_vertices_list[j]):
                return 1e9

    all_vertices = np.vstack(all_vertices_list)
    return calculate_outer_hexagon_side_length(all_vertices)

# --- Utility for generating a good initial population ---
def generate_initial_population_11_hexagons(popsize, bounds, n_hexagons):
    """
    Generates a diverse initial population for 11 hexagons, combining several strategies:
    1. A '1+6+4' pattern 
    2. A '3-5-3' pattern with a rotated central hex
    3. Perturbed versions of these smart guesses
    4. Purely random individuals within bounds
    """
    n_variables = n_hexagons * 3
    initial_population = np.zeros((popsize, n_variables), dtype=np.float64)
    
    guess_1_params = np.zeros((n_hexagons, 3))
    guess_1_params[0, :] = [0, 0, 0]
    ring_dist_1 = SQRT3 * UNIT_HEX_SIDE_LENGTH
    for i in range(6):
        angle = i * np.pi / 3
        guess_1_params[i+1, :] = [ring_dist_1 * np.cos(angle), ring_dist_1 * np.sin(angle), 0]
    rotation_outer = np.pi / 6 
    outer_dist = 3.0 * UNIT_HEX_SIDE_LENGTH 
    guess_1_params[7, :] = [-outer_dist, 0, rotation_outer] 
    guess_1_params[8, :] = [outer_dist, 0, rotation_outer]  
    guess_1_params[9, :] = [0, outer_dist * SQRT3 / 2, rotation_outer] 
    guess_1_params[10, :] = [0, -outer_dist * SQRT3 / 2, rotation_outer] 
    initial_population[0] = guess_1_params.flatten()

    guess_2_params = np.zeros((n_hexagons, 3))
    d_c = SQRT3 * UNIT_HEX_SIDE_LENGTH
    row_v_dist = 1.5 * UNIT_HEX_SIDE_LENGTH
    guess_2_params[0, :] = [-d_c, row_v_dist, 0]
    guess_2_params[1, :] = [0, row_v_dist, 0]
    guess_2_params[2, :] = [d_c, row_v_dist, 0]
    guess_2_params[3, :] = [-2*d_c, 0, 0]
    guess_2_params[4, :] = [-d_c, 0, 0]
    guess_2_params[5, :] = [0, 0, np.pi/6]
    guess_2_params[6, :] = [d_c, 0, 0]
    guess_2_params[7, :] = [2*d_c, 0, 0]
    guess_2_params[8, :] = [-d_c, -row_v_dist, 0]
    guess_2_params[9, :] = [0, -row_v_dist, 0]
    guess_2_params[10, :] = [d_c, -row_v_dist, 0]

    if popsize > 1:
        initial_population[1] = guess_2_params.flatten()
    
    num_smart_guesses = 2 if popsize > 1 else 1
    num_perturbed = (popsize - num_smart_guesses) // 2
    perturb_scale_xy = 0.75
    perturb_scale_angle = np.pi / 6
    
    for i in range(num_smart_guesses, num_smart_guesses + num_perturbed):
        base_guess = initial_population[(i % num_smart_guesses)].copy()
        perturbed_guess = base_guess.reshape((n_hexagons, 3))
        for j in range(n_hexagons):
            perturbed_guess[j, 0] += np.random.uniform(-perturb_scale_xy, perturb_scale_xy)
            perturbed_guess[j, 1] += np.random.uniform(-perturb_scale_xy, perturb_scale_xy)
            perturbed_guess[j, 2] = np.fmod(perturbed_guess[j, 2] + np.random.uniform(-perturb_scale_angle, perturb_scale_angle), 2 * np.pi)
            if perturbed_guess[j, 2] < 0:
                perturbed_guess[j, 2] += 2 * np.pi
        initial_population[i] = perturbed_guess.flatten()

    for i in range(num_smart_guesses + num_perturbed, popsize):
        for j, (low, high) in enumerate(bounds):
            initial_population[i, j] = np.random.uniform(low, high)
            
    return initial_population

# --- Main Constructor Function ---

def hexagon_packing_11():
    """ 
    Constructs an optimized packing of 11 disjoint unit regular hexagons using a
    Numba-accelerated engine and a rigorous hybrid optimization strategy.
    """
    n_hexagons = 11
    
    bounds = []
    for _ in range(n_hexagons):
        bounds.extend([(-8, 8), (-8, 8), (0, 2 * np.pi)])

    popsize = 200
    np.random.seed(42)
    initial_population = generate_initial_population_11_hexagons(popsize, bounds, n_hexagons)

    # --- Stage 1: Global Optimization with Differential Evolution ---
    de_result = differential_evolution(
        fitness_function,
        bounds,
        args=(n_hexagons,),
        maxiter=20000, # Increased from 15000 for a final exhaustive global search
        popsize=popsize,
        seed=42,
        workers=-1,
        updating='deferred',
        init=initial_population,
        tol=1e-6,
        atol=1e-6,
        disp=False,
        mutation=(0.8, 1.0),
        recombination=0.9
    )
    
    if de_result.fun >= 1e9:
        raise RuntimeError(f"Global optimization (DE) failed to find a valid solution. Best fun: {de_result.fun}")

    # --- Stage 2: Local Refinement with L-BFGS-B ---
    local_result = minimize(
        fitness_function,
        de_result.x,
        args=(n_hexagons,),
        method='L-BFGS-B',
        bounds=bounds,
        options={'ftol': 1e-15, 'gtol': 1e-10, 'maxiter': 15000} # Increased from 12000 for a final precision push
    )
    
    if local_result.success and local_result.fun < de_result.fun:
        final_params = local_result.x
    else:
        final_params = de_result.x

    if fitness_function(final_params, n_hexagons) >= 1e9:
        raise RuntimeError(f"Final optimization failed to find a valid non-overlapping solution.")

    # --- Post-processing for Final Output ---
    inner_hex_data = final_params.reshape((n_hexagons, 3))
    
    center_of_mass = np.mean(inner_hex_data[:, :2], axis=0)
    inner_hex_data[:, :2] -= center_of_mass
    
    final_vertices = np.vstack([
        get_hexagon_vertices(cx, cy, angle_rad) for cx, cy, angle_rad in inner_hex_data
    ])
    outer_hex_side_length = calculate_outer_hexagon_side_length(final_vertices)

    inner_hex_data[:, 2] = np.rad2deg(inner_hex_data[:, 2])

    outer_hex_data = np.array([0, 0, 0])
    
    return inner_hex_data, outer_hex_data, outer_hex_side_length
# EVOLVE-BLOCK-END