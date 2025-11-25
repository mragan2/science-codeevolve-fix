# EVOLVE-BLOCK-START
import numpy as np
import time
from shapely.geometry import Polygon
from itertools import combinations
from scipy.optimize import differential_evolution, minimize
import warnings
from numba import jit # Added Numba import
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
    angle_degrees=0 means a vertex is on the positive x-axis (pointy-top).
    Returns a (6, 2) numpy array of vertices.
    """
    vertices = np.empty((6, 2), dtype=np.float64)
    angle_rad_offset = math.radians(angle_degrees) # Use math.radians for scalar conversion
    
    for i in range(6):
        current_vertex_angle = math.radians(i * 60.0) + angle_rad_offset # Use math.radians
        vertices[i, 0] = side_length * math.cos(current_vertex_angle) + center_x # Use math.cos
        vertices[i, 1] = side_length * math.sin(current_vertex_angle) + center_y # Use math.sin
        
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
        return 1e9, np.array([0.0, 0.0]), 0.0 # Return penalty for empty input

    min_R = 1e9
    best_center = np.array([0.0, 0.0])
    best_angle = 0.0

    # Search for the optimal angle of the outer hexagon (0 to 30 degrees is sufficient due to symmetry)
    # 0.1 degree steps offer a good balance between precision and performance.
    for angle_deg in np.arange(0.0, 30.1, 0.1): 
        angle_rad = math.radians(angle_deg)
        c, s = math.cos(angle_rad), math.sin(angle_rad)
        
        # Rotate all points to align the candidate outer hexagon's axes with the coordinate system.
        # This simplifies the calculation of extents along the hexagon's principal directions.
        rot_x = all_vertices_np[:, 0] * c + all_vertices_np[:, 1] * s
        rot_y = -all_vertices_np[:, 0] * s + all_vertices_np[:, 1] * c
        
        sqrt3_div_2 = SQRT3 / 2.0
        
        # Calculate projections onto the three primary normal vectors of a flat-top hexagon.
        # These correspond to the "extents" that determine the outer hexagon's size.
        p1 = rot_y
        p2 = sqrt3_div_2 * rot_x + 0.5 * rot_y
        p3 = -sqrt3_div_2 * rot_x + 0.5 * rot_y
        
        # The side length R of the enclosing hexagon is derived from the maximum extent.
        # Distance between parallel sides is R * SQRT3.
        current_R = max(np.max(p1) - np.min(p1), np.max(p2) - np.min(p2), np.max(p3) - np.min(p3)) / SQRT3
        
        if current_R < min_R:
            min_R = current_R
            best_angle = angle_deg
            
            # Calculate the center of the minimal hexagon in the rotated frame
            c1 = (np.max(p1) + np.min(p1)) / 2.0
            c2 = (np.max(p2) + np.min(p2)) / 2.0
            c3 = (np.max(p3) + np.min(p3)) / 2.0
            
            center_rot_x = (c2 - c3) / SQRT3
            center_rot_y = c1
            
            # Rotate the center back to the original coordinate system
            best_center_x = center_rot_x * c - center_rot_y * s
            best_center_y = center_rot_x * s + center_rot_y * c
            best_center = np.array([best_center_x, best_center_y])

    return min_R, best_center, best_angle

def objective_function_full(params):
    """
    Objective function for 12 hexagons without symmetry constraints (36 params).
    """
    inner_hex_data = params.reshape(12, 3)
    
    # Generate polygons and collect all vertices
    inner_polygons = []
    all_vertices_list = []
    for x, y, theta in inner_hex_data:
        # Generate vertices using JIT-compiled function
        hex_vertices = _get_hexagon_vertices_jit(x, y, theta, HEX_SIDE_LENGTH)
        inner_polygons.append(Polygon(hex_vertices))
        all_vertices_list.append(hex_vertices)

    # 1. Non-overlap check with penalty
    for h1, h2 in combinations(inner_polygons, 2):
        if h1.intersects(h2):
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
    
    inner_polygons = []
    all_vertices_list = [] # Collect raw vertices for JIT function
    
    # Generate 6 hexagons from the first set of parameters (r1, alpha1, theta1_rel)
    for i in range(6):
        rotation_angle = i * 60.0 # Apply 0, 60, 120, ..., 300 degrees rotation
        
        # Calculate (x,y) center for the current hexagon
        current_alpha_rad = math.radians(alpha1_deg + rotation_angle) # Use math.radians
        cx1 = r1 * math.cos(current_alpha_rad) # Use math.cos
        cy1 = r1 * math.sin(current_alpha_rad) # Use math.sin
        
        # Calculate the hexagon's absolute orientation
        abs_theta1 = theta1_rel + rotation_angle
        
        # Generate vertices using JIT-compiled function
        hex_vertices = _get_hexagon_vertices_jit(cx1, cy1, abs_theta1, HEX_SIDE_LENGTH)
        inner_polygons.append(Polygon(hex_vertices))
        all_vertices_list.append(hex_vertices)

    # Generate 6 hexagons from the second set of parameters (r2, alpha2, theta2_rel)
    for i in range(6):
        rotation_angle = i * 60.0
        
        current_alpha_rad = math.radians(alpha2_deg + rotation_angle) # Use math.radians
        cx2 = r2 * math.cos(current_alpha_rad) # Use math.cos
        cy2 = r2 * math.sin(current_alpha_rad) # Use math.sin
        
        abs_theta2 = theta2_rel + rotation_angle
        
        # Generate vertices using JIT-compiled function
        hex_vertices = _get_hexagon_vertices_jit(cx2, cy2, abs_theta2, HEX_SIDE_LENGTH)
        inner_polygons.append(Polygon(hex_vertices))
        all_vertices_list.append(hex_vertices)

    # 1. Non-overlap check with penalty
    for h1, h2 in combinations(inner_polygons, 2):
        if h1.intersects(h2):
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
    2. A multi-start local refinement on all 36 parameters (breaking symmetry) to fine-tune the solution.
    """
    start_time = time.time()
    
    # --- Stage 1: Global Search with D6 Symmetry ---
    
    # Define bounds for the 6 symmetric parameters: [r1, alpha1, theta1, r2, alpha2, theta2]
    bounds_symmetric = [(0.0, 4.0), (0.0, 60.0), (0.0, 60.0),
                        (0.0, 4.0), (0.0, 60.0), (0.0, 60.0)]
    
    np.random.seed(42)
    
    # Global optimization using Differential Evolution with polishing (from Inspiration 2)
    result_de = differential_evolution(
        objective_function_symmetric,
        bounds_symmetric,
        strategy='best1bin',
        maxiter=7500,
        popsize=60,
        mutation=(0.8, 1.2),
        recombination=0.9,
        tol=1e-5,
        disp=False,
        workers=-1,
        polish=True # Use DE's built-in polishing for efficiency and better symmetric result
    )
    
    # Global optimization using Differential Evolution with enhanced parameters and polish
    result_de = differential_evolution(
        objective_function_symmetric,
        bounds_symmetric,
        strategy='best1bin',
        maxiter=7500,       # Increased iterations for thorough global search
        popsize=60,         # Increased population size for better exploration
        mutation=(0.8, 1.2), # Increased mutation range for broader exploration
        recombination=0.9,  # Increased recombination for more mixing
        tol=1e-5,           # Tighter tolerance for DE to converge closer
        disp=False,
        workers=-1,         # Use all available CPU cores
        polish=True         # Added polish=True for local refinement within DE (from Inspiration 2)
    )
    
    optimized_params_sym = result_de.x
    r_outer_sym_polished = result_de.fun # Objective function value from DE
        
    if r_outer_sym_polished > 1e8: # Check against penalty value
        print("Error: Global symmetric search failed to find a valid packing.")
        return np.zeros((12, 3)), np.array([0, 0, 0]), 1e9

    # --- Reconstruct the full 12-hexagon configuration from the FINAL symmetric solution ---
    # This reconstruction uses the best parameters found after DE (including its internal polish).
    r1, alpha1_deg, theta1_rel, r2, alpha2_deg, theta2_rel = optimized_params_sym
    initial_inner_hex_data_list = []
    initial_all_vertices_list = [] # Collect raw vertices for JIT function
    
    for i in range(6): # First orbit of 6 hexagons
        rotation_angle = i * 60.0
        alpha_rad = math.radians(alpha1_deg + rotation_angle) # Use math.radians
        cx = r1 * math.cos(alpha_rad) # Use math.cos
        cy = r1 * math.sin(alpha_rad) # Use math.sin
        abs_theta = theta1_rel + rotation_angle
        
        initial_inner_hex_data_list.append([cx, cy, abs_theta])
        hex_vertices = _get_hexagon_vertices_jit(cx, cy, abs_theta, HEX_SIDE_LENGTH)
        initial_all_vertices_list.append(hex_vertices)

    for i in range(6): # Second orbit of 6 hexagons
        rotation_angle = i * 60.0
        alpha_rad = math.radians(alpha2_deg + rotation_angle) # Use math.radians
        cx = r2 * math.cos(alpha_rad) # Use math.cos
        cy = r2 * math.sin(alpha_rad) # Use math.sin
        abs_theta = theta2_rel + rotation_angle
        
        initial_inner_hex_data_list.append([cx, cy, abs_theta])
        hex_vertices = _get_hexagon_vertices_jit(cx, cy, abs_theta, HEX_SIDE_LENGTH)
        initial_all_vertices_list.append(hex_vertices)
    
    initial_guess_full = np.array(initial_inner_hex_data_list)
    
    # Calculate precise outer hex properties from the symmetric solution for comparison and bounds
    initial_all_vertices_np = np.vstack(initial_all_vertices_list)
    r_outer_sym_actual, outer_center_sym, outer_angle_sym = _find_minimal_enclosing_hexagon_jit(initial_all_vertices_np)
    
    # Initialize best results with the symmetric solution's actual values
    best_full_result_fun = r_outer_sym_actual
    best_full_result_x = initial_guess_full.flatten()
    best_outer_center = outer_center_sym
    best_outer_angle = outer_angle_sym

    # --- Stage 2: Multi-Start Local Refinement without Symmetry Constraints ---
    # This stage attempts to find a better, non-symmetric packing by running a local
    # optimizer from multiple perturbed starting points derived from the best symmetric solution.
    
    # Define bounds for the 36 parameters (12 hexagons * [x, y, angle])
    # Bounds are based on the extent of the symmetric solution plus a margin
    max_abs_coord = r_outer_sym_actual + HEX_CIRCUMRADIUS * 1.5 # Using 1.5 as a reasonable margin
    bounds_full = []
    for _ in range(12): # Use _ for unused loop variable
        bounds_full.extend([
            (-max_abs_coord, max_abs_coord), # x-coordinate
            (-max_abs_coord, max_abs_coord), # y-coordinate
            (-360.0, 360.0)                  # angle (allow full range)
        ])
                
    num_restarts = 10 # Number of refinement attempts from different perturbations (aligned with Insp1/3 strategy)
    perturbation_xy_scale = 0.025 # Small perturbation for x, y coordinates (from Insp3)
    perturbation_angle_scale = 5.0 # Small perturbation for angles (increased for more exploration)
        
    for i in range(num_restarts):
        np.random.seed(43 + i) # Use a different seed for each perturbation for reproducibility
                
        perturbed_guess = initial_guess_full.copy()
        # Apply perturbation to break symmetry and explore local landscape
        perturbed_guess[:, 0] += np.random.uniform(-perturbation_xy_scale, perturbation_xy_scale, 12)
        perturbed_guess[:, 1] += np.random.uniform(-perturbation_xy_scale, perturbation_xy_scale, 12)
        perturbed_guess[:, 2] += np.random.uniform(-perturbation_angle_scale, perturbation_angle_scale, 12)
        
        result_min_full = minimize(
            objective_function_full,
            perturbed_guess.flatten(), # Use the perturbed guess for the local search
            method='L-BFGS-B', # L-BFGS-B is generally robust for local, bounded optimization
            bounds=bounds_full,
            tol=1e-9, # Very tight tolerance for the final refinement
            options={'maxiter': 5000, 'ftol': 1e-12, 'gtol': 1e-8} # Generous limits for fine-tuning
        )
                
        # If this run found a better solution, update the best one found so far
        if result_min_full.success and result_min_full.fun < best_full_result_fun:
            best_full_result_fun = result_min_full.fun
            best_full_result_x = result_min_full.x
            # The actual outer_center and outer_angle will be recalculated for the final best_full_result_x at the end.
        
    # --- Finalize Solution ---
    # The final solution is the best one found across the symmetric stage and all asymmetric refinements.
    final_outer_side_length = best_full_result_fun
    inner_hex_data = best_full_result_x.reshape(12, 3)
            
    # Recalculate precise outer hex data (center and angle) for the final best configuration
    final_all_vertices_list = []
    for x, y, theta in inner_hex_data:
        final_all_vertices_list.append(_get_hexagon_vertices_jit(x, y, theta, HEX_SIDE_LENGTH))
    final_all_vertices_np = np.vstack(final_all_vertices_list)
    _, final_outer_center, final_outer_angle = _find_minimal_enclosing_hexagon_jit(final_all_vertices_np)
        
    outer_hex_data = np.array([final_outer_center[0], final_outer_center[1], final_outer_angle])
            
    _eval_time = time.time() - start_time
            
    return inner_hex_data, outer_hex_data, final_outer_side_length
# EVOLVE-BLOCK-END