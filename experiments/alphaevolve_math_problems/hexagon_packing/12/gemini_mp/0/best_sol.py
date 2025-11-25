# EVOLVE-BLOCK-START
import numpy as np
from shapely.geometry import Polygon
from scipy.optimize import differential_evolution
from numba import njit

# Constants for unit regular hexagons
UNIT_HEX_SIDE_LENGTH = 1.0
SQRT3 = np.sqrt(3.0) # Use 3.0 for float to ensure numba infers float type
UNIT_HEX_INRADIUS = UNIT_HEX_SIDE_LENGTH * SQRT3 / 2.0 # Distance from center to midpoint of a side
UNIT_HEX_AREA = (3 * SQRT3 / 2) * UNIT_HEX_SIDE_LENGTH**2 # Area of a single unit hexagon (approx 2.598)

# Numba-optimized function to get hexagon vertices
@njit
def get_hexagon_vertices(cx, cy, side_length, angle_degrees):
    """
    Generates vertices for a regular hexagon.
    Angle 0 degrees for this function means a pointy-top hexagon (first vertex on positive x-axis).
    """
    vertices = np.zeros((6, 2), dtype=np.float64)
    angle_radians = np.deg2rad(angle_degrees)
    
    # Vertices for a pointy-top hexagon centered at (0,0)
    # The first vertex is at (side_length, 0) if angle_degrees=0.
    for i in range(6):
        current_angle = angle_radians + np.deg2rad(i * 60)
        vertices[i, 0] = cx + side_length * np.cos(current_angle)
        vertices[i, 1] = cy + side_length * np.sin(current_angle)
    return vertices

# --- High-performance Numba-based geometric functions ---

@njit
def project_vertices_onto_axis(vertices, axis):
    """Projects hexagon vertices onto an axis and returns the min/max projection."""
    min_proj = np.dot(vertices[0], axis)
    max_proj = min_proj
    for i in range(1, 6):
        proj = np.dot(vertices[i], axis)
        if proj < min_proj:
            min_proj = proj
        elif proj > max_proj:
            max_proj = proj
    return min_proj, max_proj

@njit
def get_hexagon_normals(angle_degrees):
    """Returns the 3 unique normal vectors for a hexagon's sides."""
    normals = np.zeros((3, 2), dtype=np.float64)
    # Normals for a hexagon rotated by angle_degrees are at angle_degrees + 30 + k*60
    base_angle_rad = np.deg2rad(angle_degrees + 30.0)
    for i in range(3):
        angle = base_angle_rad + np.deg2rad(i * 60)
        normals[i, 0] = np.cos(angle)
        normals[i, 1] = np.sin(angle)
    return normals

@njit
def check_hexagons_overlap_sat(c1, verts1, angle1_deg, c2, verts2, angle2_deg):
    """
    Checks for overlap between two hexagons using the Separating Axis Theorem (SAT).
    This is a numba-jitted, high-performance replacement for Shapely's intersection check.
    """
    # Broad phase: Quick distance check between centers.
    # If distance is >= 2*circumradius (2*side_length), they cannot overlap.
    # Circumradius is UNIT_HEX_SIDE_LENGTH = 1.0. So distance is 2.0. dist_sq is 4.0.
    dist_sq = (c1[0] - c2[0])**2 + (c1[1] - c2[1])**2
    if dist_sq >= 4.0:
        return False

    # Narrow phase: Separating Axis Theorem. Test normals of both hexagons.
    normals1 = get_hexagon_normals(angle1_deg)
    normals2 = get_hexagon_normals(angle2_deg)
    
    # Test axes from the first hexagon
    for i in range(3):
        axis = normals1[i]
        min1, max1 = project_vertices_onto_axis(verts1, axis)
        min2, max2 = project_vertices_onto_axis(verts2, axis)
        if max1 < min2 - 1e-9 or max2 < min1 - 1e-9: # Use tolerance for float precision
            return False  # Found a separating axis

    # Test axes from the second hexagon
    for i in range(3):
        axis = normals2[i]
        min1, max1 = project_vertices_onto_axis(verts1, axis)
        min2, max2 = project_vertices_onto_axis(verts2, axis)
        if max1 < min2 - 1e-9 or max2 < min1 - 1e-9: # Use tolerance for float precision
            return False  # Found a separating axis
            
    return True  # No separating axis found, hexagons overlap

# Numba-optimized function to calculate the side length of the smallest enclosing hexagon
@njit
def calculate_min_enclosing_hexagon_side_length_numba(all_vertices, outer_hex_angle_degrees):
    """
    Calculates the side length of the smallest regular hexagon (centered at origin,
    with a specified rotation) that encloses all given points.
    
    The side length R_outer for a hexagon centered at (0,0) with rotation `outer_hex_angle_degrees`
    is given by R_outer = max_apothem_extent / (SQRT3 / 2).
    The apothem is the distance from the center to the midpoint of a side.
    The normals to the sides of the outer hexagon are offset by 30 degrees from its orientation angle.
    For a flat-top hexagon (0 degrees rotation), its sides are at 0, 60, 120. Normals are at 30, 90, 150.
    For a pointy-top hexagon (30 degrees rotation), its sides are at 30, 90, 150. Normals are at 0, 60, 120.
    """
    if len(all_vertices) == 0:
        return 0.0

    max_proj = 0.0
    
    # Angles for normals of the outer hexagon, relative to x-axis
    # These are the angles perpendicular to the sides of the outer hexagon.
    # For a hexagon with `outer_hex_angle_degrees` rotation, its sides are at
    # `outer_hex_angle_degrees + k*60`. Its normals are at `outer_hex_angle_degrees + 30 + k*60`.
    # We only need 3 unique angles due to symmetry.
    base_normal_angle = np.deg2rad(outer_hex_angle_degrees + 30.0)
    angles = np.array([base_normal_angle, base_normal_angle + np.deg2rad(60.0), base_normal_angle + np.deg2rad(120.0)], dtype=np.float64)
    
    for i in range(len(all_vertices)):
        x, y = all_vertices[i]
        for j in range(len(angles)):
            angle = angles[j]
            proj = x * np.cos(angle) + y * np.sin(angle)
            if np.abs(proj) > max_proj:
                max_proj = np.abs(proj)
    
    # The apothem of the outer hexagon is `R_outer * SQRT3 / 2`.
    # So, `max_proj = R_outer * SQRT3 / 2`.
    # `R_outer = max_proj / (SQRT3 / 2) = max_proj * 2 / SQRT3`.
    return max_proj * 2.0 / SQRT3

@njit
def evaluate_packing_numba(params):
    """
    Numba-jitted core of the objective function for high-performance evaluation.
    This function handles geometry generation, overlap checks, and outer hexagon calculation.
    """
    r1, hex_rot1_deg, r2, hex_rot2_deg = params
    
    # Store hexagon data: centers, vertices, angles
    hex_centers = np.zeros((12, 2), dtype=np.float64)
    hex_verts = np.zeros((12, 6, 2), dtype=np.float64)
    hex_angles = np.zeros(12, dtype=np.float64)
    
    # --- Generate Hexagon Positions and Geometries ---
    # Ring 1: 6 hexagons
    for k in range(6):
        angle_rad = np.deg2rad(k * 60.0)
        cx = r1 * np.cos(angle_rad)
        cy = r1 * np.sin(angle_rad)
        hex_centers[k, 0] = cx
        hex_centers[k, 1] = cy
        hex_angles[k] = hex_rot1_deg
        hex_verts[k] = get_hexagon_vertices(cx, cy, UNIT_HEX_SIDE_LENGTH, hex_rot1_deg)

    # Ring 2: 6 hexagons, staggered
    for k in range(6):
        idx = k + 6
        angle_rad = np.deg2rad(k * 60.0 + 30.0)
        cx = r2 * np.cos(angle_rad)
        cy = r2 * np.sin(angle_rad)
        hex_centers[idx, 0] = cx
        hex_centers[idx, 1] = cy
        hex_angles[idx] = hex_rot2_deg
        hex_verts[idx] = get_hexagon_vertices(cx, cy, UNIT_HEX_SIDE_LENGTH, hex_rot2_deg)
        
    # --- Overlap Penalty Calculation (using Numba-based SAT) ---
    overlap_penalty = 0.0
    C_OVERLAP_PENALTY = 100.0  # Large fixed penalty for any overlap
    
    for i in range(12):
        for j in range(i + 1, 12):
            if check_hexagons_overlap_sat(
                hex_centers[i], hex_verts[i], hex_angles[i],
                hex_centers[j], hex_verts[j], hex_angles[j]
            ):
                overlap_penalty += C_OVERLAP_PENALTY
    
    # --- Radius Placement Penalty ---
    # Penalize if rings are too small, causing self-intersection within a ring.
    min_r_for_ring_non_overlap = SQRT3 * UNIT_HEX_SIDE_LENGTH
    radius_placement_penalty = 0.0
    if r1 < min_r_for_ring_non_overlap:
        radius_placement_penalty += (min_r_for_ring_non_overlap - r1) * 50000.0
    if r2 < min_r_for_ring_non_overlap:
        radius_placement_penalty += (min_r_for_ring_non_overlap - r2) * 50000.0
        
    # Combine penalties. If any exist, return a high value to discard the solution.
    total_penalty = overlap_penalty + radius_placement_penalty
    if total_penalty > 0:
        # Adding params helps create a gradient for the optimizer in invalid regions
        return total_penalty + r1 + r2 

    # --- Outer Hexagon Calculation ---
    all_inner_hex_vertices_np = hex_verts.reshape(72, 2)
    
    # Find the optimal orientation for the outer hexagon by checking a range of angles
    min_R = 1e9
    # Search over the unique range of orientations [0, 60) degrees
    for angle_deg in np.arange(0.0, 60.0, 2.0): # 2-degree step for performance
        R = calculate_min_enclosing_hexagon_side_length_numba(all_inner_hex_vertices_np, angle_deg)
        if R < min_R:
            min_R = R
            
    outer_hex_side_length = min_R
    
    return outer_hex_side_length

def evaluate_packing(params):
    """
    Wrapper for the Numba-jitted objective function.
    """
    return evaluate_packing_numba(params)

def hexagon_packing_12():
    """ 
    Constructs a packing of 12 disjoint unit regular hexagons inside a larger regular hexagon, maximizing 1/outer_hex_side_length. 
    Returns
        inner_hex_data: np.ndarray of shape (12,3), where each row is of the form (x, y, angle_degrees) containing the (x,y) coordinates and angle_degree of the respective inner hexagon.
        outer_hex_data: np.ndarray of shape (3,) of form (x,y,angle_degree) containing the (x,y) coordinates and angle_degree of the outer hexagon.
        outer_hex_side_length: float representing the side length of the outer hexagon.
    """
    
    # Define bounds for the optimization variables: [r1, hex_rot1_deg, r2, hex_rot2_deg]
    
    # The lower bound for radii is set to SQRT3, the theoretical minimum for non-overlap within a ring of 6 hexagons.
    min_radius_bound = SQRT3 # approx 1.732 
    max_radius_bound = 4.2 # Reduced from 5.0 to 4.2 to tighten search space, target is ~3.93

    # Due to the 60-degree rotational symmetry of a regular hexagon, all unique orientations
    # can be represented within a 60-degree range. We use [-30, 30] to cover this space
    # efficiently, focusing the search.
    min_rot_bound = -30.0
    max_rot_bound = 30.0

    bounds = [(min_radius_bound, max_radius_bound), (min_rot_bound, max_rot_bound), 
              (min_radius_bound, max_radius_bound), (min_rot_bound, max_rot_bound)]
    
    # Use Differential Evolution for robust global optimization
    # Set a random seed for reproducibility
    np.random.seed(42) 
    
    # `maxiter` and `popsize` are increased for a more exhaustive search.
    # `polish=True` applies a local search (L-BFGS-B) to the best candidate solution
    # found by the global search, which helps to refine the final result.
    result = differential_evolution(
        evaluate_packing, 
        bounds, 
        maxiter=15000, # Increased iterations due to faster evaluation function
        popsize=120,   # Increased population size for broader search
        tol=1e-5,      # Tighter tolerance for higher precision
        atol=1e-5,     # Tighter absolute tolerance
        seed=42,
        workers=-1,
        polish=True    # Refine the result with a local optimizer
    )
    
    optimal_params = result.x
    
    # Reconstruct the optimal packing based on the found parameters
    r1_opt, hex_rot1_deg_opt, r2_opt, hex_rot2_deg_opt = optimal_params
    
    final_inner_hex_data = []
    final_all_inner_hex_vertices = []

    # Ring 1
    for k in range(6):
        angle_rad = np.deg2rad(k * 60)
        cx = r1_opt * np.cos(angle_rad)
        cy = r1_opt * np.sin(angle_rad)
        final_inner_hex_data.append([cx, cy, hex_rot1_deg_opt])
        final_all_inner_hex_vertices.extend(get_hexagon_vertices(cx, cy, UNIT_HEX_SIDE_LENGTH, hex_rot1_deg_opt))

    # Ring 2
    for k in range(6):
        angle_rad = np.deg2rad(k * 60 + 30)
        cx = r2_opt * np.cos(angle_rad)
        cy = r2_opt * np.sin(angle_rad)
        final_inner_hex_data.append([cx, cy, hex_rot2_deg_opt])
        final_all_inner_hex_vertices.extend(get_hexagon_vertices(cx, cy, UNIT_HEX_SIDE_LENGTH, hex_rot2_deg_opt))

    final_inner_hex_data_np = np.array(final_inner_hex_data)
    final_all_inner_hex_vertices_np = np.array(final_all_inner_hex_vertices)
    
    # Determine the final outer hexagon side length and its optimal orientation
    # by searching across all possible orientations for the minimum side length.
    min_R = 1e9
    best_angle_deg = 0.0
    # A finer search (0.1 degree) for the final result to maximize precision.
    for angle_deg in np.arange(0.0, 60.0, 0.1):
        R = calculate_min_enclosing_hexagon_side_length_numba(final_all_inner_hex_vertices_np, angle_deg)
        if R < min_R:
            min_R = R
            best_angle_deg = angle_deg
            
    final_outer_hex_side_length = min_R
    final_outer_hex_orientation_deg = best_angle_deg
    
    # The outer hexagon is centered at (0,0) with the determined optimal orientation
    final_outer_hex_data = np.array([0, 0, final_outer_hex_orientation_deg]) 

    return final_inner_hex_data_np, final_outer_hex_data, final_outer_hex_side_length
# EVOLVE-BLOCK-END