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
    Angle 0 degrees for this function means a flat-top hexagon (sides parallel to x-axis).
    """
    vertices = np.zeros((6, 2), dtype=np.float64)
    angle_radians = np.deg2rad(angle_degrees)
    
    # Vertices for a flat-top hexagon centered at (0,0)
    # The first vertex is at (side_length, 0) if angle_degrees=0.
    for i in range(6):
        current_angle = angle_radians + np.deg2rad(i * 60) # Start from 0 deg for flat-top
        vertices[i, 0] = cx + side_length * np.cos(current_angle)
        vertices[i, 1] = cy + side_length * np.sin(current_angle)
    return vertices

# Not Numba-optimized due to Shapely dependency
def create_hexagon_polygon(cx, cy, side_length, angle_degrees):
    """Creates a shapely Polygon object for a hexagon."""
    vertices = get_hexagon_vertices(cx, cy, side_length, angle_degrees)
    return Polygon(vertices)

# Numba-optimized function to calculate the side length of the smallest enclosing hexagon
@njit
def calculate_min_enclosing_hexagon_side_length_numba(all_vertices, outer_hex_angle_degrees):
    """
    Calculates the side length of the smallest regular hexagon (centered at origin,
    with a given rotation) that encloses all given points.
    
    The side length R_outer for a hexagon centered at (0,0) with rotation `outer_hex_angle_degrees`
    is given by R_outer = max_apothem_extent / (SQRT3 / 2), where max_apothem_extent is the maximum
    absolute projection of any point onto the normals of the hexagon's sides.
    The normals for a flat-top hexagon are at 30, 90, 150 degrees.
    If the hexagon is rotated by `outer_hex_angle_degrees`, these normal angles are also rotated.
    """
    if len(all_vertices) == 0:
        return 0.0

    max_proj = 0.0
    
    # Base angles for normals of a flat-top hexagon (relative to x-axis)
    base_normal_angles = np.array([np.deg2rad(30), np.deg2rad(90), np.deg2rad(150)], dtype=np.float64)
    
    # Adjust normal angles by the outer hexagon's rotation
    outer_hex_angle_radians = np.deg2rad(outer_hex_angle_degrees)
    adjusted_normal_angles = base_normal_angles + outer_hex_angle_radians
    
    for i in range(len(all_vertices)):
        x, y = all_vertices[i]
        for j in range(len(adjusted_normal_angles)):
            angle = adjusted_normal_angles[j]
            proj = x * np.cos(angle) + y * np.sin(angle)
            if np.abs(proj) > max_proj:
                max_proj = np.abs(proj)
    
    # The apothem of the outer hexagon is `R_outer * SQRT3 / 2`.
    # So, `max_proj = R_outer * SQRT3 / 2`.
    # `R_outer = max_proj / (SQRT3 / 2) = max_proj * 2 / SQRT3`.
    return max_proj * 2.0 / SQRT3

def evaluate_packing(params):
    """
    Objective function for the optimization.
    Takes optimization parameters and returns the penalized outer hexagon side length.
    
    Parameters:
        params (list or np.ndarray): [r1, hex_rot1_deg, r2, hex_rot2_deg, outer_hex_angle_deg]
            r1: radius for the centers of hexagons in the first ring
            hex_rot1_deg: rotation angle for hexagons in the first ring
            r2: radius for the centers of hexagons in the second ring
            hex_rot2_deg: rotation angle for hexagons in the second ring
            outer_hex_angle_deg: rotation angle for the outer enclosing hexagon
            
    Returns:
        float: The outer hexagon side length plus penalties for overlaps.
    """
    r1, hex_rot1_deg, r2, hex_rot2_deg, outer_hex_angle_deg = params

    inner_hex_polygons = []
    all_inner_hex_vertices = []
    
    # Hexagon 0: Central hexagon (1 total)
    cx_0, cy_0, rot_0 = 0.0, 0.0, 0.0 # Fixed at origin, 0 degrees rotation
    poly_0 = create_hexagon_polygon(cx_0, cy_0, UNIT_HEX_SIDE_LENGTH, rot_0)
    inner_hex_polygons.append(poly_0)
    all_inner_hex_vertices.extend(get_hexagon_vertices(cx_0, cy_0, UNIT_HEX_SIDE_LENGTH, rot_0))

    # Ring 1: 6 hexagons, centers evenly spaced around r1
    for k in range(6):
        angle_rad = np.deg2rad(k * 60)
        cx = r1 * np.cos(angle_rad)
        cy = r1 * np.sin(angle_rad)
        poly = create_hexagon_polygon(cx, cy, UNIT_HEX_SIDE_LENGTH, hex_rot1_deg)
        inner_hex_polygons.append(poly)
        all_inner_hex_vertices.extend(get_hexagon_vertices(cx, cy, UNIT_HEX_SIDE_LENGTH, hex_rot1_deg))

    # Ring 2: 5 hexagons, centers evenly spaced around r2, staggered by 30 degrees, (one position left empty)
    # This creates a 1 + 6 + 5 = 12 configuration
    for k in range(5): # Only 5 hexagons in this ring (k=0 to 4)
        angle_rad = np.deg2rad(k * 60 + 30) # Staggered by 30 degrees for denser packing
        cx = r2 * np.cos(angle_rad)
        cy = r2 * np.sin(angle_rad)
        poly = create_hexagon_polygon(cx, cy, UNIT_HEX_SIDE_LENGTH, hex_rot2_deg)
        inner_hex_polygons.append(poly)
        all_inner_hex_vertices.extend(get_hexagon_vertices(cx, cy, UNIT_HEX_SIDE_LENGTH, hex_rot2_deg))

    # Overlap Penalty Calculation
    overlap_penalty = 0.0
    
    # Use Shapely's robust intersection check for overlap penalty.
    # A very large penalty is applied for any overlap, consisting of a fixed base
    # penalty plus a term proportional to the intersection area. This strongly
    # discourages invalid configurations.
    C_overlap_area_multiplier = 1000000.0 # Multiplier for intersection area
    C_overlap_fixed_base_penalty = 1000.0 # Fixed penalty for any overlap
    
    for i in range(len(inner_hex_polygons)):
        for j in range(i + 1, len(inner_hex_polygons)):
            if inner_hex_polygons[i].intersects(inner_hex_polygons[j]):
                overlap_penalty += C_overlap_fixed_base_penalty
                intersection_area = inner_hex_polygons[i].intersection(inner_hex_polygons[j]).area
                overlap_penalty += intersection_area * C_overlap_area_multiplier
                
    # Calculate outer hexagon side length, now considering its rotation
    all_inner_hex_vertices_np = np.array(all_inner_hex_vertices)
    outer_hex_side_length = calculate_min_enclosing_hexagon_side_length_numba(all_inner_hex_vertices_np, outer_hex_angle_deg)
    
    # Removed radius_placement_penalty. Overlap penalty should be sufficient to manage
    # minimum distances between hexagons, including the central one.

    # Total objective to minimize: outer_hex_side_length + overlap_penalty
    return outer_hex_side_length + overlap_penalty

def hexagon_packing_12():
    """ 
    Constructs a packing of 12 disjoint unit regular hexagons inside a larger regular hexagon, maximizing 1/outer_hex_side_length. 
    Returns
        inner_hex_data: np.ndarray of shape (12,3), where each row is of the form (x, y, angle_degrees) containing the (x,y) coordinates and angle_degree of the respective inner hexagon.
        outer_hex_data: np.ndarray of shape (3,) of form (x,y,angle_degree) containing the (x,y) coordinates and angle_degree of the outer hexagon.
        outer_hex_side_length: float representing the side length of the outer hexagon.
    """
    
    # Define bounds for the optimization variables: [r1, hex_rot1_deg, r2, hex_rot2_deg, outer_hex_angle_deg]
    
    # For a central hexagon and a ring of 6:
    # Minimum r1 for non-overlap with central hex (circumradius = 1) if touching: 2 * UNIT_HEX_SIDE_LENGTH = 2.0
    # However, allowing r1 to be smaller (e.g., 1.0) lets the overlap penalty guide tight packing,
    # potentially allowing for interlocking configurations.
    min_r1_bound = 1.0 
    max_r1_bound = 4.0 

    # For the second ring, it must be further out than the first.
    min_r2_bound = 1.0 # Can be small if the packing is not strictly nested rings.
    max_r2_bound = 5.0 

    # Due to the 60-degree rotational symmetry of a regular hexagon, all unique orientations
    # for inner hexagons can be represented within a 60-degree range. We use [-30, 30]
    min_inner_rot_bound = -30.0
    max_inner_rot_bound = 30.0

    # For the outer hexagon, we need to search a 60-degree range due to its symmetry.
    # We can use [0, 60) for its angle.
    min_outer_rot_bound = 0.0
    max_outer_rot_bound = 60.0 # Search full 60 degrees for outer hex rotation

    bounds = [(min_r1_bound, max_r1_bound), (min_inner_rot_bound, max_inner_rot_bound), 
              (min_r2_bound, max_r2_bound), (min_inner_rot_bound, max_inner_rot_bound),
              (min_outer_rot_bound, max_outer_rot_bound)]
    
    # Use Differential Evolution for robust global optimization
    # Set a random seed for reproducibility
    np.random.seed(42) 
    
    # `maxiter` and `popsize` are increased for a more exhaustive search,
    # especially with the added optimization variable (outer_hex_angle_deg).
    # `polish=True` applies a local search (L-BFGS-B) to the best candidate solution
    # found by the global search, which helps to refine the final result.
    result = differential_evolution(
        evaluate_packing, 
        bounds, 
        maxiter=30000, # Increased maxiter for more thorough search with increased parameter space
        popsize=200,   # Increased popsize for better exploration
        tol=0.00001,   # Setting tolerance to a very low value for maximum precision
        atol=0.00001,  # Setting tolerance to a very low value for maximum precision
        seed=42,
        workers=-1,
        polish=True    # Refine the result with a local optimizer
    )
    
    optimal_params = result.x
    
    # Reconstruct the optimal packing based on the found parameters
    r1_opt, hex_rot1_deg_opt, r2_opt, hex_rot2_deg_opt, outer_hex_angle_deg_opt = optimal_params
    
    final_inner_hex_data = []
    final_all_inner_hex_vertices = []

    # Hexagon 0: Central hexagon
    cx_0, cy_0, rot_0 = 0.0, 0.0, 0.0 # Fixed at origin, 0 degrees rotation
    final_inner_hex_data.append([cx_0, cy_0, rot_0])
    final_all_inner_hex_vertices.extend(get_hexagon_vertices(cx_0, cy_0, UNIT_HEX_SIDE_LENGTH, rot_0))

    # Ring 1 (6 hexagons)
    for k in range(6):
        angle_rad = np.deg2rad(k * 60)
        cx = r1_opt * np.cos(angle_rad)
        cy = r1_opt * np.sin(angle_rad)
        final_inner_hex_data.append([cx, cy, hex_rot1_deg_opt])
        final_all_inner_hex_vertices.extend(get_hexagon_vertices(cx, cy, UNIT_HEX_SIDE_LENGTH, hex_rot1_deg_opt))

    # Ring 2 (5 hexagons)
    for k in range(5): # Only 5 hexagons in this ring (k=0 to 4)
        angle_rad = np.deg2rad(k * 60 + 30)
        cx = r2_opt * np.cos(angle_rad)
        cy = r2_opt * np.sin(angle_rad)
        final_inner_hex_data.append([cx, cy, hex_rot2_deg_opt])
        final_all_inner_hex_vertices.extend(get_hexagon_vertices(cx, cy, UNIT_HEX_SIDE_LENGTH, hex_rot2_deg_opt))

    final_inner_hex_data_np = np.array(final_inner_hex_data)
    final_all_inner_hex_vertices_np = np.array(final_all_inner_hex_vertices)
    
    final_outer_hex_side_length = calculate_min_enclosing_hexagon_side_length_numba(final_all_inner_hex_vertices_np, outer_hex_angle_deg_opt)
    
    # The outer hexagon is centered at (0,0) with optimized rotation
    final_outer_hex_data = np.array([0.0, 0.0, outer_hex_angle_deg_opt]) 

    return final_inner_hex_data_np, final_outer_hex_data, final_outer_hex_side_length
# EVOLVE-BLOCK-END