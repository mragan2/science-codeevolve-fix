# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import differential_evolution, minimize # Added minimize
from shapely.geometry import Polygon
# from shapely.affinity import rotate, translate # No longer needed for create_hexagon directly
import warnings # Added for warning suppression
from numba import jit # Added for JIT compilation
import math # Added for math.radians, math.cos, math.sin in JIT-compiled functions
import time # Moved inside function to track eval_time

# Suppress potential Shapely warnings about GEOS being unstable
warnings.filterwarnings("ignore", category=UserWarning)

# --- Constants for Hexagon Geometry ---
SQRT3 = np.sqrt(3.0)
HEX_SIDE_LENGTH = 1.0
HEX_INRADIUS = HEX_SIDE_LENGTH * SQRT3 / 2.0  # Distance from center to midpoint of side (apothem)
HEX_CIRCUMRADIUS = HEX_SIDE_LENGTH          # Distance from center to vertex (radius)

# JIT-compiled helper to get hexagon vertices (from Inspiration 1/3)
@jit(nopython=True, cache=True)
def _get_hexagon_vertices_jit(center_x, center_y, angle_degrees, side_length):
    """
    JIT-compiled function to calculate the vertices of a regular hexagon.
    angle_degrees=0 means a vertex is on the positive x-axis (pointy-top).
    Returns a (6, 2) numpy array of vertices.
    """
    vertices = np.empty((6, 2), dtype=np.float64)
    angle_rad = np.radians(angle_degrees)
    
    for i in range(6):
        current_vertex_angle = np.radians(i * 60.0) + angle_rad
        vertices[i, 0] = side_length * np.cos(current_vertex_angle) + center_x
        vertices[i, 1] = side_length * np.sin(current_vertex_angle) + center_y
        
    return vertices

# --- Helper function to create a hexagon as a shapely Polygon (adapted from TARGET and Inspiration 1/3) ---
# Moved to module level to be pickleable for multiprocessing
def create_hexagon(center_x, center_y, angle_degrees, side_length=HEX_SIDE_LENGTH):
    """
    Creates a shapely Polygon object for a regular hexagon.
    angle_degrees=0 means a vertex is on the positive x-axis (pointy-top).
    Leverages JIT-compiled helper for vertex calculation.
    """
    translated_vertices = _get_hexagon_vertices_jit(center_x, center_y, angle_degrees, side_length)
    return Polygon(translated_vertices)

# --- JIT-compiled function to find the minimal enclosing hexagon (from Inspiration 1/3) ---
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
            current_search_angles.extend(np.arange(start_angle, end_angle + angle_step, angle_step)) # +step to ensure upper bound is included

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

# --- Objective function for the optimizer (Symmetric) ---
# Renamed from 'objective' to 'objective_function_symmetric'
def objective_function_symmetric(params):
    """
    Objective function for optimization, enforcing 6-fold rotational symmetry for 12 hexagons.
    params: [x1, y1, angle1, x2, y2, angle2] for 2 base hexagons.
    Returns the side length of the smallest enclosing outer hexagon, with penalties for overlaps.
    """
    x1, y1, angle1, x2, y2, angle2 = params
    
    inner_polygons = []
    all_vertices_list = []
    base_hex_data = [(x1, y1, angle1), (x2, y2, angle2)]
    
    # Generate 12 hexagons using 6-fold rotational symmetry
    for i in range(6):
        rot_deg = 60.0 * i
        rot_rad = np.deg2rad(rot_deg)
        cos_t, sin_t = np.cos(rot_rad), np.sin(rot_rad)
        
        for (bx, by, bangle) in base_hex_data:
            # Rotate center and add hexagon's own rotation
            nx = bx * cos_t - by * sin_t
            ny = bx * sin_t + by * cos_t
            nangle = (bangle + rot_deg) % 360.0 # Keep angle within [0, 360)
            
            hex_vertices = _get_hexagon_vertices_jit(nx, ny, nangle, HEX_SIDE_LENGTH)
            inner_polygons.append(Polygon(hex_vertices))
            all_vertices_list.append(hex_vertices)

    # 1. Non-overlap check with penalty (hard penalty for DE and symmetric local refinement)
    for i in range(len(inner_polygons)):
        for j in range(i + 1, len(inner_polygons)):
            if inner_polygons[i].buffer(-1e-9).intersects(inner_polygons[j].buffer(-1e-9)):
                return 1e9  # Large penalty

    # 2. Collect all vertices into a single NumPy array for JIT function
    all_vertices_np = np.vstack(all_vertices_list)

    # 3. Calculate smallest enclosing hexagon side length R_outer using JIT function
    R_outer, _, _ = _find_minimal_enclosing_hexagon_jit(all_vertices_np)
    return R_outer

# --- Objective function for full (non-symmetric) optimization (for local refinement) ---
def objective_function_full(params):
    """
    Objective function for 12 hexagons without symmetry constraints, using area-based penalty.
    params: A 36-element array (12 * [x, y, angle_degrees]).
    """
    inner_hex_data = params.reshape(12, 3)
    
    inner_polygons = []
    all_vertices_list = []
    for x, y, theta in inner_hex_data:
        hex_vertices = _get_hexagon_vertices_jit(x, y, theta, HEX_SIDE_LENGTH)
        inner_polygons.append(Polygon(hex_vertices))
        all_vertices_list.append(hex_vertices)

    # 1. Non-overlap check with area-based penalty for local optimizers
    overlap_penalty = 0.0
    for i in range(len(inner_polygons)):
        for j in range(i + 1, len(inner_polygons)):
            # Use buffer(0) to get a more accurate intersection area for just touching polygons
            if inner_polygons[i].intersects(inner_polygons[j]):
                intersection_area = inner_polygons[i].intersection(inner_polygons[j]).area
                overlap_penalty += intersection_area * 10000.0 # Large penalty factor

    # If there's any overlap, return a large value + penalty.
    # Otherwise, R_outer should be minimized.
    if overlap_penalty > 0:
        # A very large base value to ensure overlap solutions are always worse
        return 1e12 + overlap_penalty

    # 2. Collect all vertices into a single NumPy array for JIT function
    all_vertices_np = np.vstack(all_vertices_list)

    # 3. Calculate smallest enclosing hexagon side length R_outer using JIT function
    R_outer, _, _ = _find_minimal_enclosing_hexagon_jit(all_vertices_np)
    return R_outer # No penalty if no overlap

def hexagon_packing_12():
    """
    Constructs an optimal packing of 12 unit regular hexagons by using
    differential evolution to find a D6-symmetric arrangement that minimizes
    the side length of the enclosing regular hexagon, followed by a local
    refinement without symmetry constraints.
    """
    start_time = time.time() # Moved inside function to track eval_time

    # --- Stage 1: Global Search with D6 Symmetry ---
    # Bounds for the 6 variables of the 2 base hexagons.
    bounds_symmetric = [
        (0.0, 3.5),  # x1: x-coordinate for the first base hexagon
        (-2.5, 2.5), # y1: y-coordinate for the first base hexagon
        (0.0, 60.0), # angle1: Rotation angle for the first base hexagon (0 to <60 degrees)
        (0.0, 3.5),  # x2: x-coordinate for the second base hexagon
        (-2.5, 2.5), # y2: y-coordinate for the second base hexagon
        (0.0, 60.0)  # angle2: Rotation angle for the second base hexagon (0 to <60 degrees)
    ]

    initial_population_size = 50
    bounds_array = np.array(bounds_symmetric)
    
    rng = np.random.default_rng(42) 
    initial_population = rng.uniform(bounds_array[:, 0], bounds_array[:, 1], size=(initial_population_size, len(bounds_symmetric)))

    # Inject a good heuristic guess for a two-ring configuration (from TARGET/Inspiration 2)
    good_guess = np.array([2.0, 0.0, 0.0, 2.8, 0.0, 30.0])
    initial_population[0] = good_guess 

    # Global optimization using differential_evolution
    result_de = differential_evolution(
        objective_function_symmetric, # Use the symmetric objective
        bounds_symmetric,
        maxiter=3000, 
        popsize=50,   
        tol=1e-5,
        seed=42, 
        workers=-1, 
        init=initial_population 
    )

    optimized_params_sym = result_de.x
    r_outer_sym = result_de.fun

    # Local refinement on the symmetric parameters (added for fine-tuning)
    if r_outer_sym < 1e8: # Check against hard penalty threshold
        result_min_sym = minimize(
            objective_function_symmetric,
            optimized_params_sym,
            method='L-BFGS-B',
            bounds=bounds_symmetric,
            tol=1e-8 # Tight tolerance for symmetric refinement
        )
        if result_min_sym.success and result_min_sym.fun < r_outer_sym:
            optimized_params_sym = result_min_sym.x
            r_outer_sym = result_min_sym.fun

    if r_outer_sym >= 1e8: # Use 1e8 as the threshold for 'failed' due to hard penalty
        print("Error: Global symmetric search failed to find a valid packing. Falling back to default.")
        # Fallback to a default arrangement if optimization fails to find a valid solution.
        inner_hex_data = np.array([
            [0, 0, 0], [-2.5, 0, 0], [2.5, 0, 0], [-1.25, 2.17, 0],
            [1.25, 2.17, 0], [-1.25, -2.17, 0], [1.25, -2.17, 0],
            [-3.75, 2.17, 0], [3.75, 2.17, 0], [-3.75, -2.17, 0],
            [3.75, -2.17, 0], [0, -4, 0]
        ])
        outer_hex_side_length = 8.0
        outer_hex_data = np.array([0, 0, 0])
        _eval_time = time.time() - start_time
        return inner_hex_data, outer_hex_data, outer_hex_side_length


    # --- Reconstruct the full 12-hexagon configuration from the symmetric solution ---
    x1_opt, y1_opt, angle1_opt, x2_opt, y2_opt, angle2_opt = optimized_params_sym
    initial_inner_hex_data_list = []
    
    base_hex_data = [(x1_opt, y1_opt, angle1_opt), (x2_opt, y2_opt, angle2_opt)]

    for i in range(6):
        rot_deg = 60.0 * i
        rot_rad = np.deg2rad(rot_deg)
        cos_t, sin_t = np.cos(rot_rad), np.sin(rot_rad)
        
        for (bx, by, bangle) in base_hex_data:
            nx = bx * cos_t - by * sin_t
            ny = bx * sin_t + by * cos_t
            nangle = (bangle + rot_deg) % 360.0
            initial_inner_hex_data_list.append([nx, ny, nangle])

    initial_guess_full = np.array(initial_inner_hex_data_list)
    
    # Calculate initial outer hex properties from symmetric solution for comparison and bounds
    initial_all_vertices_list = []
    for x, y, theta in initial_guess_full:
        initial_all_vertices_list.append(_get_hexagon_vertices_jit(x, y, theta, HEX_SIDE_LENGTH))
    initial_all_vertices_np = np.vstack(initial_all_vertices_list)
    r_outer_sym_actual, outer_center_sym, outer_angle_sym = _find_minimal_enclosing_hexagon_jit(initial_all_vertices_np)
    # Ensure r_outer_sym is the *actual* one, not just the DE's objective function value (which might be slightly off)
    r_outer_sym = r_outer_sym_actual 

    # --- Stage 2: Local Refinement without Symmetry Constraints ---
    # Define bounds for the 36 parameters based on the symmetric solution's extent
    max_radial_extent = np.max(np.linalg.norm(initial_guess_full[:, :2], axis=1)) + HEX_CIRCUMRADIUS * 2.0 # Allow some room
    bounds_full = []
    for i in range(12):
        bounds_full.extend([
            (-max_radial_extent, max_radial_extent), # x-coordinate
            (-max_radial_extent, max_radial_extent), # y-coordinate
            (-360.0, 360.0)                          # angle (allow full rotation for local search)
        ])

    result_min_full = minimize(
        objective_function_full, # Use the full objective with area penalty
        initial_guess_full.flatten(),
        method='L-BFGS-B',
        bounds=bounds_full,
        tol=1e-9, 
        options={'maxiter': 7000, 'ftol': 1e-12, 'gtol': 1e-8} # Increased maxiter for full refinement
    )
    
    # Final solution is the best of the symmetric and full refinement stages
    # Check if the full refinement improved the R and didn't result in overlaps
    if result_min_full.success and result_min_full.fun < r_outer_sym and result_min_full.fun < 1e11: # Check against area penalty threshold
        final_outer_side_length = result_min_full.fun
        inner_hex_data = result_min_full.x.reshape(12, 3)
        
        # Recalculate precise outer hex data for the final (potentially asymmetric) configuration
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