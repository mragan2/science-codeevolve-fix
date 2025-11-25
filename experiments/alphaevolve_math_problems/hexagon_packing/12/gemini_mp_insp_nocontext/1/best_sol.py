# EVOLVE-BLOCK-START
import numpy as np
import time
from shapely.geometry import Polygon
from itertools import combinations
from scipy.spatial.distance import cdist
from scipy.optimize import differential_evolution, minimize
import warnings
from numba import jit # New import for JIT compilation
import math # Needed for math.radians, math.cos, math.sin in JIT-compiled functions

# Suppress potential Shapely warnings about GEOS being unstable
warnings.filterwarnings("ignore", category=UserWarning)

# --- Constants for Hexagon Geometry ---
SQRT3 = np.sqrt(3.0)
HEX_SIDE_LENGTH = 1.0
HEX_INRADIUS = HEX_SIDE_LENGTH * SQRT3 / 2.0  # Distance from center to midpoint of side (apothem)
HEX_CIRCUMRADIUS = HEX_SIDE_LENGTH          # Distance from center to vertex (radius)

# JIT-compiled helper to get hexagon vertices
@jit(nopython=True, cache=True)
def _get_hexagon_vertices_jit(center_x, center_y, angle_degrees, side_length):
    """
    JIT-compiled function to calculate the vertices of a regular hexagon.
    angle_degrees=0 means a vertex is on the positive x-axis (pointy-top, consistent with inspirations).
    Returns a (6, 2) numpy array of vertices.
    """
    vertices = np.empty((6, 2), dtype=np.float64)
    angle_rad = np.radians(angle_degrees)
    
    for i in range(6):
        current_vertex_angle = np.radians(i * 60.0) + angle_rad
        vertices[i, 0] = side_length * np.cos(current_vertex_angle) + center_x
        vertices[i, 1] = side_length * np.sin(current_vertex_angle) + center_y
        
    return vertices

def create_hexagon_polygon(center_x, center_y, angle_degrees, side_length=HEX_SIDE_LENGTH):
    """
    Creates a shapely Polygon object for a regular hexagon.
    angle_degrees=0 means a vertex is on the positive x-axis (pointy-top).
    Leverages JIT-compiled helper for vertex calculation.
    """
    # Use the JIT-compiled function to get vertices
    translated_vertices = _get_hexagon_vertices_jit(center_x, center_y, angle_degrees, side_length)
    return Polygon(translated_vertices)

@jit(nopython=True, cache=True)
def _find_minimal_enclosing_hexagon_jit(all_vertices_np):
    """
    JIT-compiled function to find the side length R, center (x,y), and rotation (degrees) of the
    smallest regular hexagon that encloses a given set of vertices.
    This function implements an adaptive search strategy for the outer hexagon's orientation.
    Returns: (min_R, best_outer_hex_center_np_array, best_outer_hex_angle_deg)
    """
    if all_vertices_np.shape[0] == 0:
        return 0.0, np.array([0.0, 0.0]), 0.0

    min_R = np.inf
    best_outer_hex_center = np.array([0.0, 0.0])
    best_outer_hex_angle = 0.0

    # Adaptive search for angle: coarse then fine
    # First pass: 1.0 degree steps, Second pass: 0.1 degree steps around the best angle
    angle_steps = [1.0, 0.1] 
    search_ranges = [(0.0, 60.0)] # Initial full range
    
    for step_idx, angle_step in enumerate(angle_steps):
        current_search_angles = []
        for start_angle, end_angle in search_ranges:
            # +step to ensure upper bound is included, then filter for [0, 60)
            current_search_angles.extend(np.arange(start_angle, end_angle + angle_step, angle_step)) 

        # Filter angles to be within [0, 60) for symmetry
        current_search_angles = [a for a in current_search_angles if 0 <= a < 60]
        
        # Ensure unique angles and sort
        current_search_angles = np.unique(np.array(current_search_angles))

        local_min_R = np.inf
        local_best_angle = 0.0

        for angle_deg in current_search_angles:
            # Define three primary axes for projection (vertex-to-vertex and side-to-side directions)
            # These are for a "pointy-top" hexagon (vertex at angle_deg)
            axis1_rad = np.radians(angle_deg)
            axis2_rad = np.radians(angle_deg + 60)
            axis3_rad = np.radians(angle_deg + 30) # Normal to a side, for side-to-side extent

            axis1_vec = np.array([np.cos(axis1_rad), np.sin(axis1_rad)])
            axis2_vec = np.array([np.cos(axis2_rad), np.sin(axis2_rad)])
            axis3_vec = np.array([np.cos(axis3_rad), np.sin(axis3_rad)])

            projs1 = np.dot(all_vertices_np, axis1_vec)
            projs2 = np.dot(all_vertices_np, axis2_vec)
            projs3 = np.dot(all_vertices_np, axis3_vec)

            # R_from_axis1: half the extent along vertex-to-vertex axis
            R_from_axis1 = (np.max(projs1) - np.min(projs1)) / 2.0
            
            # R_from_axis2: half the extent along another vertex-to-vertex axis
            R_from_axis2 = (np.max(projs2) - np.min(projs2)) / 2.0
            
            # R_from_axis3_side: extent along side-to-side axis, converted to R
            # The distance between parallel sides is R * SQRT3.
            R_from_axis3_side = (np.max(projs3) - np.min(projs3)) / SQRT3
            
            current_R = max(R_from_axis1, R_from_axis2, R_from_axis3_side)
            
            if current_R < local_min_R:
                local_min_R = current_R
                local_best_angle = angle_deg
        
        if local_min_R < min_R:
            min_R = local_min_R
            best_outer_hex_angle = local_best_angle
            
            # Recalculate center for the globally best angle found so far
            normal_angles_rad = [np.radians(best_outer_hex_angle + 30), np.radians(best_outer_hex_angle + 90)]
            
            n1_vec = np.array([np.cos(normal_angles_rad[0]), np.sin(normal_angles_rad[0])])
            n2_vec = np.array([np.cos(normal_angles_rad[1]), np.sin(normal_angles_rad[1])])
            
            proj_n1 = np.dot(all_vertices_np, n1_vec)
            proj_n2 = np.dot(all_vertices_np, n2_vec)

            A = np.array([[n1_vec[0], n1_vec[1]], [n2_vec[0], n2_vec[1]]])
            b = np.array([(np.min(proj_n1) + np.max(proj_n1)) / 2.0, (np.min(proj_n2) + np.max(proj_n2)) / 2.0])
            
            # Manual 2x2 matrix inversion for Numba nopython mode
            det = A[0,0] * A[1,1] - A[0,1] * A[1,0]
            if det != 0:
                best_outer_hex_center_x = (b[0] * A[1,1] - b[1] * A[0,1]) / det
                best_outer_hex_center_y = (A[0,0] * b[1] - A[1,0] * b[0]) / det
                best_outer_hex_center = np.array([best_outer_hex_center_x, best_outer_hex_center_y])
            else:
                best_outer_hex_center = np.array([0.0, 0.0]) # Fallback
        
        # For the next (finer) step, narrow down the search range
        if step_idx < len(angle_steps) - 1 and min_R != np.inf:
            search_ranges = [(max(0.0, best_outer_hex_angle - 2.0), min(60.0, best_outer_hex_angle + 2.0))]

    return min_R, best_outer_hex_center, best_outer_hex_angle

def objective_function_full(params):
    """
    Objective function for 12 hexagons without symmetry constraints.
    params: A 36-element array (12 * [x, y, angle_degrees]).
    """
    inner_hex_data = params.reshape(12, 3)
    
    # Generate vertices and centers
    all_vertices_list = []
    centers = np.empty((12, 2), dtype=np.float64)
    for i, (x, y, theta) in enumerate(inner_hex_data):
        all_vertices_list.append(_get_hexagon_vertices_jit(x, y, theta, HEX_SIDE_LENGTH))
        centers[i, 0] = x
        centers[i, 1] = y

    # 1. Hybrid non-overlap check (inspired by IP1)
    # Fast distance pre-check to find pairs that *might* overlap.
    dist_matrix = cdist(centers, centers)
    # Potential overlap if distance < 2 * R_circumradius. Use a small tolerance.
    potential_overlap_indices = np.where((dist_matrix > 1e-9) & (dist_matrix < 2.0 * HEX_CIRCUMRADIUS - 1e-7))
    
    if potential_overlap_indices[0].size > 0:
        # Create polygons lazily only for pairs that are close enough.
        inner_polygons = [None] * 12
        for i, j in zip(potential_overlap_indices[0], potential_overlap_indices[1]):
            if i >= j: # Avoid duplicates and self-checks
                continue

            if inner_polygons[i] is None:
                inner_polygons[i] = Polygon(all_vertices_list[i])
            if inner_polygons[j] is None:
                inner_polygons[j] = Polygon(all_vertices_list[j])
            
            # Use buffer for robustness against floating point errors for touching hexagons (inspired by IP2/IP3)
            if inner_polygons[i].buffer(-1e-9).intersects(inner_polygons[j].buffer(-1e-9)):
                return 1e9 # Penalty for overlaps

    # 2. Collect all vertices into a single NumPy array for JIT function
    all_vertices_np = np.vstack(all_vertices_list)

    # 3. Calculate smallest enclosing hexagon side length R_outer using JIT function
    R_outer, _, _ = _find_minimal_enclosing_hexagon_jit(all_vertices_np)
    return R_outer


def objective_function_symmetric(params):
    """
    Objective function for optimization, enforcing 6-fold rotational symmetry for 12 hexagons.
    
    params: A 6-element array: [r1, alpha1_deg, theta1_rel, r2, alpha2_deg, theta2_rel]
        r_i: Radial distance from the origin for the i-th unique hexagon's center.
        alpha_i_deg: Angular position (in degrees) for the i-th unique hexagon's center
                     within a 60-degree fundamental wedge (e.g., 0 to 60 degrees).
        theta_i_rel: Relative rotation (in degrees) of the i-th hexagon. This rotation
                     is applied *relative* to the angular position of the hexagon's center.
                     (e.g., 0 to 60 degrees).
    
    The function generates 12 inner hexagons: 6 from (r1, alpha1_deg, theta1_rel)
    and 6 from (r2, alpha2_deg, theta2_rel) by applying 60-degree rotations.
    """
    
    r1, alpha1_deg, theta1_rel, r2, alpha2_deg, theta2_rel = params
    
    all_vertices_list = []
    centers = np.empty((12, 2), dtype=np.float64)
    
    # Generate 6 hexagons from the first set of parameters
    for i in range(6):
        rotation_angle = i * 60.0
        current_alpha_rad = np.radians(alpha1_deg + rotation_angle)
        cx = r1 * np.cos(current_alpha_rad)
        cy = r1 * np.sin(current_alpha_rad)
        abs_theta = theta1_rel + rotation_angle
        
        centers[i] = [cx, cy]
        all_vertices_list.append(_get_hexagon_vertices_jit(cx, cy, abs_theta, HEX_SIDE_LENGTH))

    # Generate 6 hexagons from the second set of parameters
    for i in range(6):
        rotation_angle = i * 60.0
        current_alpha_rad = np.radians(alpha2_deg + rotation_angle)
        cx = r2 * np.cos(current_alpha_rad)
        cy = r2 * np.sin(current_alpha_rad)
        abs_theta = theta2_rel + rotation_angle
        
        centers[i + 6] = [cx, cy]
        all_vertices_list.append(_get_hexagon_vertices_jit(cx, cy, abs_theta, HEX_SIDE_LENGTH))

    # 1. Hybrid non-overlap check (inspired by IP1)
    dist_matrix = cdist(centers, centers)
    potential_overlap_indices = np.where((dist_matrix > 1e-9) & (dist_matrix < 2.0 * HEX_CIRCUMRADIUS - 1e-7))

    if potential_overlap_indices[0].size > 0:
        inner_polygons = [None] * 12
        for i, j in zip(potential_overlap_indices[0], potential_overlap_indices[1]):
            if i >= j:
                continue
            if inner_polygons[i] is None:
                inner_polygons[i] = Polygon(all_vertices_list[i])
            if inner_polygons[j] is None:
                inner_polygons[j] = Polygon(all_vertices_list[j])
            
            if inner_polygons[i].buffer(-1e-9).intersects(inner_polygons[j].buffer(-1e-9)):
                return 1e9 # Penalty for overlaps
    
    # 2. Collect all vertices into a single NumPy array for JIT function
    all_vertices_np = np.vstack(all_vertices_list)
    
    # 3. Calculate smallest enclosing hexagon side length R_outer using JIT function
    R_outer, _, _ = _find_minimal_enclosing_hexagon_jit(all_vertices_np)
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
    
    # Define tighter bounds for the 6 symmetric parameters, inspired by IP2/IP3.
    # [r1, alpha1, theta1, r2, alpha2, theta2]
    bounds_symmetric = [
        (HEX_INRADIUS * 1.0, HEX_INRADIUS * 2.8), # r1 (inner ring radius)
        (0.0, 60.0),                               # alpha1_deg
        (0.0, 60.0),                               # theta1_rel
        (HEX_INRADIUS * 2.8, HEX_INRADIUS * 4.8), # r2 (outer ring radius)
        (0.0, 60.0),                               # alpha2_deg
        (0.0, 60.0)                                # theta2_rel
    ]
    
    np.random.seed(42) # For reproducibility
    
    # Create an initial population and inject a good heuristic guess (inspired by IP2/IP3).
    popsize = 60
    good_guess_symmetric = np.array([SQRT3, 0.0, 30.0, 2.0 * SQRT3, 30.0, 0.0])
    
    rng = np.random.default_rng(42)
    bounds_array = np.array(bounds_symmetric)
    initial_population = rng.uniform(bounds_array[:, 0], bounds_array[:, 1], size=(popsize, len(bounds_symmetric)))
    initial_population[0] = good_guess_symmetric # Inject the guess

    # Global optimization using Differential Evolution with enhanced parameters
    result_de = differential_evolution(
        objective_function_symmetric,
        bounds_symmetric,
        strategy='best1bin',
        maxiter=8000,       # Increased iterations for a more thorough global search
        popsize=popsize,
        mutation=(0.8, 1.2),
        recombination=0.9,
        tol=1e-6,           # Tighter tolerance for DE to converge closer
        disp=False,
        workers=-1,
        seed=42,            # Add seed for full reproducibility
        init=initial_population # Provide the custom initial population
    )
    
    optimized_params_sym = result_de.x
    r_outer_sym_from_de_obj = result_de.fun # Store the objective function value from DE

    # Local refinement on the symmetric parameters
    if r_outer_sym_from_de_obj < 1e8: # Check if DE found a valid solution
        result_min_sym = minimize(
            objective_function_symmetric,
            optimized_params_sym,
            method='L-BFGS-B',
            bounds=bounds_symmetric,
            tol=1e-8 # Tight tolerance for symmetric refinement
        )
        if result_min_sym.success and result_min_sym.fun < r_outer_sym_from_de_obj:
            optimized_params_sym = result_min_sym.x
            r_outer_sym_from_de_obj = result_min_sym.fun

    if r_outer_sym_from_de_obj > 1e8:
        print("Error: Global symmetric search failed to find a valid packing.")
        return np.zeros((12, 3)), np.array([0, 0, 0]), 1e9

    # --- Reconstruct the full 12-hexagon configuration from the symmetric solution ---
    r1, alpha1_deg, theta1_rel, r2, alpha2_deg, theta2_rel = optimized_params_sym
    initial_inner_hex_data_list = []
    initial_all_vertices_list = [] # Use this to calculate initial_all_vertices_np
    
    for i in range(6): # First orbit
        rot_angle = i * 60.0
        alpha_rad = np.radians(alpha1_deg + rot_angle)
        cx = r1 * np.cos(alpha_rad)
        cy = r1 * np.sin(alpha_rad)
        abs_theta = theta1_rel + rot_angle
        
        initial_inner_hex_data_list.append([cx, cy, abs_theta])
        hex_vertices = _get_hexagon_vertices_jit(cx, cy, abs_theta, HEX_SIDE_LENGTH)
        initial_all_vertices_list.append(hex_vertices)

    for i in range(6): # Second orbit
        rot_angle = i * 60.0
        alpha_rad = np.radians(alpha2_deg + rot_angle)
        cx = r2 * np.cos(alpha_rad)
        cy = r2 * np.sin(alpha_rad)
        abs_theta = theta2_rel + rot_angle
        
        initial_inner_hex_data_list.append([cx, cy, abs_theta])
        hex_vertices = _get_hexagon_vertices_jit(cx, cy, abs_theta, HEX_SIDE_LENGTH)
        initial_all_vertices_list.append(hex_vertices)
    
    initial_guess_full = np.array(initial_inner_hex_data_list)
    
    # Calculate initial outer hex properties from symmetric solution for comparison and bounds
    initial_all_vertices_np = np.vstack(initial_all_vertices_list)
    r_outer_sym_actual, outer_center_sym, outer_angle_sym = _find_minimal_enclosing_hexagon_jit(initial_all_vertices_np)
    # Ensure r_outer_sym is the *actual* one, not just the DE's objective function value (which might be slightly off)
    r_outer_sym = r_outer_sym_actual 

    # --- Stage 2: Local Refinement without Symmetry Constraints ---
    # Use the symmetric solution as a high-quality starting point for a full local search.
    # This allows the packing to relax into a potentially non-symmetric, denser configuration.
    
    # Define bounds for the 36 parameters based on the symmetric solution's extent
    max_abs_coord = r_outer_sym + HEX_CIRCUMRADIUS * 2.0 # Use the symmetric outer radius
    bounds_full = []
    for i in range(12):
        bounds_full.extend([
            (-max_abs_coord, max_abs_coord), # x-coordinate
            (-max_abs_coord, max_abs_coord), # y-coordinate
            (-360.0, 360.0)                  # angle
        ])

    result_min_full = minimize(
        objective_function_full,
        initial_guess_full.flatten(),
        method='L-BFGS-B', 
        bounds=bounds_full,
        tol=1e-9, # Very tight tolerance for the final refinement
        options={'maxiter': 5000, 'ftol': 1e-12, 'gtol': 1e-8} # Generous limits for fine-tuning
    )
    
    # Final solution is the best of the symmetric and full refinement stages
    if result_min_full.success and result_min_full.fun < r_outer_sym:
        final_outer_side_length = result_min_full.fun
        inner_hex_data = result_min_full.x.reshape(12, 3)
        
        # Recalculate precise outer hex data for the final (potentially asymmetric) configuration
        # Need to re-collect vertices from the final inner_hex_data
        final_all_vertices_list = []
        for x, y, theta in inner_hex_data:
            final_all_vertices_list.append(_get_hexagon_vertices_jit(x, y, theta, HEX_SIDE_LENGTH))
        final_all_vertices_np = np.vstack(final_all_vertices_list)
        _, final_outer_center, final_outer_angle = _find_minimal_enclosing_hexagon_jit(final_all_vertices_np)

    else:
        final_outer_side_length = r_outer_sym
        inner_hex_data = initial_guess_full
        final_outer_center = outer_center_sym
        final_outer_angle = outer_angle_sym

    outer_hex_data = np.array([final_outer_center[0], final_outer_center[1], final_outer_angle])
    
    _eval_time = time.time() - start_time
    
    return inner_hex_data, outer_hex_data, final_outer_side_length
# EVOLVE-BLOCK-END