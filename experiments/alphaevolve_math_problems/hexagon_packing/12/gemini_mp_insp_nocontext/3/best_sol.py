# EVOLVE-BLOCK-START
import numpy as np
import time
from shapely.geometry import Polygon
from itertools import combinations
from scipy.optimize import differential_evolution, minimize
import warnings
from numba import jit

# Suppress potential Shapely warnings about GEOS being unstable
warnings.filterwarnings("ignore", category=UserWarning)

# --- Constants for Hexagon Geometry ---
SQRT3 = np.sqrt(3.0)
HEX_SIDE_LENGTH = 1.0
HEX_INRADIUS = HEX_SIDE_LENGTH * SQRT3 / 2.0  # Distance from center to midpoint of side (apothem)
HEX_CIRCUMRADIUS = HEX_SIDE_LENGTH          # Distance from center to vertex (radius)

# Base vertices for a "flat-top" unit hexagon, used by JIT function
HEX_BASE_VERTICES_JIT = np.array([
    [1.0, 0.0],
    [0.5, SQRT3/2.0],
    [-0.5, SQRT3/2.0],
    [-1.0, 0.0],
    [-0.5, -SQRT3/2.0],
    [0.5, -SQRT3/2.0]
])

@jit(nopython=True, cache=True)
def _get_hexagon_vertices_jit(center_x, center_y, angle_degrees, side_length):
    """ JIT-compiled function to calculate hexagon vertices for speed. """
    angle_rad = np.radians(angle_degrees)
    cos_a, sin_a = np.cos(angle_rad), np.sin(angle_rad)
    
    base_verts = HEX_BASE_VERTICES_JIT * side_length
    
    rotated_translated_vertices = np.empty_like(base_verts)
    for i in range(6):
        x, y = base_verts[i, 0], base_verts[i, 1]
        rotated_translated_vertices[i, 0] = x * cos_a - y * sin_a + center_x
        rotated_translated_vertices[i, 1] = x * sin_a + y * cos_a + center_y
    return rotated_translated_vertices

def create_hexagon_polygon(center_x, center_y, angle_degrees, side_length=HEX_SIDE_LENGTH):
    """ Creates a shapely Polygon, using JIT helper for vertex calculation. """
    vertices = _get_hexagon_vertices_jit(center_x, center_y, angle_degrees, side_length)
    return Polygon(vertices)

@jit(nopython=True, cache=True)
def _calculate_outer_hexagon_side_length_0deg_jit(all_vertices_array):
    """ JIT-compiled R calculation for a 0-degree outer hexagon. """
    if all_vertices_array.shape[0] == 0:
        return 0.0
    
    s3 = np.sqrt(3.0)
    max_r = 0.0
    for i in range(all_vertices_array.shape[0]):
        x, y = all_vertices_array[i, 0], all_vertices_array[i, 1]
        term1 = 2 * np.abs(y) / s3
        term2 = np.abs(s3 * x + y) / s3
        term3 = np.abs(s3 * x - y) / s3
        
        current_max = max(term1, term2, term3)
        if current_max > max_r:
            max_r = current_max
    return max_r

@jit(nopython=True, cache=True)
def _calculate_outer_hexagon_side_length_30deg_jit(all_vertices_array):
    """ JIT-compiled R calculation for a 30-degree outer hexagon. """
    if all_vertices_array.shape[0] == 0:
        return 0.0
    
    s3 = np.sqrt(3.0)
    max_r = 0.0
    for i in range(all_vertices_array.shape[0]):
        x, y = all_vertices_array[i, 0], all_vertices_array[i, 1]
        term1 = 2 * np.abs(x) / s3
        term2 = np.abs(x + s3 * y) / s3
        term3 = np.abs(x - s3 * y) / s3
        
        current_max = max(term1, term2, term3)
        if current_max > max_r:
            max_r = current_max
    return max_r

def calculate_min_enclosing_hexagon_side_length(all_vertices_array):
    """
    Calculates the minimum side length of a regular outer hexagon (centered at origin)
    that encloses all inner hexagons, using JIT-compiled helpers.
    Returns the minimum R and the corresponding optimal angle (0 or 30).
    """
    if not all_vertices_array.size:
        return 0.0, 0 # Return R and angle

    r_0deg = _calculate_outer_hexagon_side_length_0deg_jit(all_vertices_array)
    r_30deg = _calculate_outer_hexagon_side_length_30deg_jit(all_vertices_array)

    if r_0deg <= r_30deg:
        return r_0deg, 0
    else:
        return r_30deg, 30

def objective_function_full(params):
    """
    Objective function for 12 hexagons without symmetry constraints.
    Refactored for performance: JIT R calculation first, then shapely for checks.
    """
    inner_hex_data = params.reshape(12, 3)
    
    # 1. Generate all vertices first using the fast JIT function
    all_vertices_list = [_get_hexagon_vertices_jit(x, y, theta, HEX_SIDE_LENGTH) for x, y, theta in inner_hex_data]
    all_vertices_np = np.vstack(all_vertices_list)

    # 2. Calculate smallest enclosing hexagon side length R_outer
    R_outer, _ = calculate_min_enclosing_hexagon_side_length(all_vertices_np)
    
    # 3. Now create shapely polygons and perform the expensive non-overlap check
    inner_polygons = [Polygon(verts) for verts in all_vertices_list]
    for h1, h2 in combinations(inner_polygons, 2):
        if h1.buffer(-1e-9).intersects(h2.buffer(-1e-9)):
            return 1e9 # Penalty for overlaps

    return R_outer


def objective_function_symmetric(params):
    """
    Objective function enforcing 6-fold rotational symmetry.
    Refactored for performance: JIT R calculation first, then shapely for checks.
    """
    r1, alpha1_deg, theta1_rel, r2, alpha2_deg, theta2_rel = params
    
    all_vertices_list = []
    hex_data_list = [] # Store params to create polygons later if needed

    # Generate data for 12 hexagons
    base_params = [(r1, alpha1_deg, theta1_rel), (r2, alpha2_deg, theta2_rel)]
    for r, alpha_deg, theta_rel in base_params:
        for i in range(6):
            rotation_angle = i * 60.0
            current_alpha_rad = np.radians(alpha_deg + rotation_angle)
            cx = r * np.cos(current_alpha_rad)
            cy = r * np.sin(current_alpha_rad)
            abs_theta = theta_rel + rotation_angle
            hex_data_list.append((cx, cy, abs_theta))
    
    # 1. Generate all vertices using the fast JIT function
    for cx, cy, abs_theta in hex_data_list:
        all_vertices_list.append(_get_hexagon_vertices_jit(cx, cy, abs_theta, HEX_SIDE_LENGTH))
    all_vertices_np = np.vstack(all_vertices_list)

    # 2. Calculate smallest enclosing hexagon side length R_outer
    R_outer, _ = calculate_min_enclosing_hexagon_side_length(all_vertices_np)

    # 3. Now create shapely polygons and perform the expensive non-overlap check
    inner_polygons = [Polygon(verts) for verts in all_vertices_list]
    for h1, h2 in combinations(inner_polygons, 2):
        if h1.buffer(-1e-9).intersects(h2.buffer(-1e-9)):
            return 1e9 # Penalty for overlaps

    return R_outer


def hexagon_packing_12():
    """
    Constructs an optimal packing of 12 disjoint unit regular hexagons inside a larger regular hexagon.
    This is achieved via a two-stage optimization process:
    1. A global search assuming D6 symmetry to efficiently find a near-optimal configuration.
    2. A local refinement on all 36 parameters (breaking symmetry) to fine-tune the solution.
    """
    start_time = time.time()
    
    # --- Stage 1: Global Search with D6 Symmetry ---
    
    # Widen bounds slightly to ensure good guesses are not on the edge.
    bounds_symmetric = [
        (HEX_INRADIUS * 1.0, HEX_INRADIUS * 2.8), # r1 (inner ring radius)
        (0.0, 60.0),                               # alpha1_deg
        (0.0, 60.0),                               # theta1_rel
        (HEX_INRADIUS * 2.8, HEX_INRADIUS * 4.8), # r2 (outer ring radius)
        (0.0, 60.0),                               # alpha2_deg
        (0.0, 60.0)                                # theta2_rel
    ]
    
    # Inspired by Inspirations 2 & 3: Inject a good initial guess.
    good_guess = np.array([SQRT3, 0.0, 30.0, 2.0 * SQRT3, 30.0, 0.0])
    popsize = 80
    
    rng = np.random.default_rng(42)
    bounds_array = np.array(bounds_symmetric)
    initial_population = rng.uniform(bounds_array[:, 0], bounds_array[:, 1], size=(popsize, len(bounds_symmetric)))
    initial_population[0] = good_guess # Inject the guess

    # Global optimization with more aggressive parameters, enabled by JIT speedup.
    result_de = differential_evolution(
        objective_function_symmetric,
        bounds_symmetric,
        strategy='best1bin',
        maxiter=8000,
        popsize=popsize,
        mutation=(0.8, 1.2),
        recombination=0.9,
        tol=1e-6,
        disp=False,
        workers=-1,
        seed=42,
        init=initial_population
    )
    
    optimized_params_sym = result_de.x
    r_outer_sym = result_de.fun

    # Local refinement on the symmetric parameters
    if r_outer_sym < 1e8:
        result_min_sym = minimize(
            objective_function_symmetric,
            optimized_params_sym,
            method='L-BFGS-B',
            bounds=bounds_symmetric,
            tol=1e-9
        )
        if result_min_sym.success and result_min_sym.fun < r_outer_sym:
            optimized_params_sym = result_min_sym.x
            r_outer_sym = result_min_sym.fun

    if r_outer_sym > 1e8:
        print("Error: Global symmetric search failed to find a valid packing.")
        return np.zeros((12, 3)), np.array([0, 0, 0]), 1e9

    # --- Reconstruct the full 12-hexagon configuration from the symmetric solution ---
    r1, alpha1_deg, theta1_rel, r2, alpha2_deg, theta2_rel = optimized_params_sym
    inner_hex_data_list = []
    base_params = [(r1, alpha1_deg, theta1_rel), (r2, alpha2_deg, theta2_rel)]
    for r, alpha_deg, theta_rel in base_params:
        for i in range(6):
            rot_angle = i * 60.0
            alpha_rad = np.radians(alpha_deg + rot_angle)
            inner_hex_data_list.append([r * np.cos(alpha_rad), r * np.sin(alpha_rad), theta_rel + rot_angle])
    
    initial_guess_full = np.array(inner_hex_data_list)

    # --- Stage 2: Local Refinement without Symmetry Constraints ---
    max_abs_coord = np.max(np.abs(initial_guess_full[:, :2])) + HEX_CIRCUMRADIUS * 2.0
    bounds_full = [(-max_abs_coord, max_abs_coord), (-max_abs_coord, max_abs_coord), (-360.0, 360.0)] * 12

    result_min_full = minimize(
        objective_function_full,
        initial_guess_full.flatten(),
        method='L-BFGS-B',
        bounds=bounds_full,
        tol=1e-9,
        options={'maxiter': 5000, 'ftol': 1e-12, 'gtol': 1e-8}
    )
    
    # Determine the best configuration found
    if result_min_full.success and result_min_full.fun < r_outer_sym:
        inner_hex_data = result_min_full.x.reshape(12, 3)
    else:
        inner_hex_data = initial_guess_full

    # --- Final Calculation of Outer Hexagon ---
    all_final_vertices = np.vstack([
        _get_hexagon_vertices_jit(x, y, theta, HEX_SIDE_LENGTH) for x, y, theta in inner_hex_data
    ])
    final_outer_side_length, final_outer_hex_angle_degrees = calculate_min_enclosing_hexagon_side_length(all_final_vertices)
    
    outer_hex_data = np.array([0, 0, final_outer_hex_angle_degrees])
    
    return inner_hex_data, outer_hex_data, final_outer_side_length
# EVOLVE-BLOCK-END