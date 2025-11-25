# EVOLVE-BLOCK-START
import numpy as np
from shapely.geometry import Polygon, Point
from scipy.optimize import differential_evolution
from numba import jit, float64 # New import for Numba JIT compilation

# Constants for unit regular hexagon (side_length = 1)
HEX_SIDE_LENGTH = 1.0
HEX_APOTHEM = HEX_SIDE_LENGTH * np.sqrt(3) / 2
HEX_CIRCUMRADIUS = HEX_SIDE_LENGTH # For a unit hexagon, circumradius (distance from center to vertex) is 1.

# Helper function to get the 6 vertices of a unit hexagon, JIT-compiled for performance
@jit(float64[:,:](float64, float64, float64), nopython=True, cache=True)
def get_hexagon_vertices(center_x: float, center_y: float, angle_degrees: float) -> np.ndarray:
    """
    Calculates the 6 vertices of a unit regular hexagon.
    Args:
        center_x, center_y: Coordinates of the hexagon's center.
        angle_degrees: Rotation angle of the hexagon in degrees (0 degrees means flat top/bottom).
    Returns:
        np.ndarray: A (6,2) array of vertex coordinates.
    """
    angle_radians = np.deg2rad(angle_degrees)
    vertices = np.empty((6, 2), dtype=np.float64) # Preallocate array for Numba
    for i in range(6):
        # Vertices are at 0, 60, 120... degrees relative to the hexagon's own orientation.
        # For a unit hexagon, circumradius is 1.
        current_angle = angle_radians + i * np.pi / 3 
        vertices[i, 0] = center_x + HEX_CIRCUMRADIUS * np.cos(current_angle)
        vertices[i, 1] = center_y + HEX_CIRCUMRADIUS * np.sin(current_angle)
    return vertices

# Helper function to create a shapely Polygon object for a hexagon
# Cannot be JIT-compiled due to reliance on shapely
def create_hexagon_polygon(center_x: float, center_y: float, angle_degrees: float) -> Polygon:
    """
    Creates a shapely Polygon object for a unit regular hexagon.
    Args:
        center_x, center_y: Coordinates of the hexagon's center.
        angle_degrees: Rotation angle of the hexagon in degrees.
    Returns:
        shapely.geometry.Polygon: The hexagon polygon.
    """
    vertices = get_hexagon_vertices(center_x, center_y, angle_degrees)
    return Polygon(vertices)

# JIT-compiled helper to calculate the minimal outer hexagon side length
@jit(float64(float64[:,:]), nopython=True, cache=True)
def calculate_min_R_outer(all_vertices_array: np.ndarray) -> float:
    """
    Calculates the side length of the smallest enclosing regular hexagon
    (centered at origin, unrotated) for a given set of vertices.
    Args:
        all_vertices_array: A (N,2) numpy array of vertex coordinates.
    Returns:
        float: The side length of the outer hexagon.
    """
    min_R_outer = 0.0
    sqrt3_over_2 = np.sqrt(3) / 2.0
    sqrt3 = np.sqrt(3)

    for i in range(all_vertices_array.shape[0]):
        vx, vy = all_vertices_array[i, 0], all_vertices_array[i, 1]
        R_candidate_y = np.abs(vy) / sqrt3_over_2
        R_candidate_slope1 = np.abs(sqrt3 * vx - vy) / sqrt3
        R_candidate_slope2 = np.abs(sqrt3 * vx + vy) / sqrt3
        
        min_R_outer = max(min_R_outer, R_candidate_y, R_candidate_slope1, R_candidate_slope2)
    
    return min_R_outer

# Objective function to minimize the outer hexagon's side length
def objective_function(params: np.ndarray) -> float:
    """
    Evaluates a packing configuration defined by `params` and returns the minimum
    side length of an outer hexagon required to contain all inner hexagons without
    interior overlaps.
    Args:
        params: A 1D numpy array of 6 parameters: 
                [x_A, y_A, theta_A, x_B, y_B, theta_B] for the two seed hexagons.
    Returns:
        float: The side length of the outer hexagon, or np.inf if overlaps occur.
    """
    # Unpack parameters for the two seed hexagons
    x_A, y_A, theta_A = params[0], params[1], params[2]
    x_B, y_B, theta_B = params[3], params[4], params[5]

    hex_polygons = []
    all_vertices_list = [] # Collect vertices as list of numpy arrays for later vstack

    # Strategy: Two rings of 6 hexagons each, generated with D6 (6-fold rotational) symmetry.
    # This ensures an overall D6 symmetric packing of 12 hexagons.

    # First Ring (H0-H5) - 6 hexagons generated from H_A_seed by 60-degree rotations
    for i in range(6):
        rot_angle_deg = i * 60
        
        # Rotate H_A_seed's center (x_A, y_A) around the origin by rot_angle_deg
        rot_angle_rad = np.deg2rad(rot_angle_deg)
        cos_a = np.cos(rot_angle_rad)
        sin_a = np.sin(rot_angle_rad)
        rotated_x_A = x_A * cos_a - y_A * sin_a
        rotated_y_A = x_A * sin_a + y_A * cos_a
        
        # The individual hexagon's orientation (theta_A) is relative to its position vector from the origin.
        # So, the absolute angle of the rotated hex is (theta_A + rot_angle_deg).
        current_hex_angle_A = theta_A + rot_angle_deg
        # Calculate vertices once and use for both Polygon creation and all_vertices_list
        vertices_A = get_hexagon_vertices(rotated_x_A, rotated_y_A, current_hex_angle_A)
        hex_polygons.append(Polygon(vertices_A))
        all_vertices_list.append(vertices_A) # Append numpy array

    # Second Ring (H6-H11) - 6 hexagons generated from H_B_seed by 60-degree rotations
    for i in range(6):
        rot_angle_deg = i * 60
        
        # Rotate H_B_seed's center (x_B, y_B) around the origin by rot_angle_deg
        rot_angle_rad = np.deg2rad(rot_angle_deg)
        cos_a = np.cos(rot_angle_rad)
        sin_a = np.sin(rot_angle_rad)
        rotated_x_B = x_B * cos_a - y_B * sin_a
        rotated_y_B = x_B * sin_a + y_B * cos_a

        # The individual hexagon's orientation (theta_B) is relative to its position vector from the origin.
        current_hex_angle_B = theta_B + rot_angle_deg
        # Calculate vertices once and use for both Polygon creation and all_vertices_list
        vertices_B = get_hexagon_vertices(rotated_x_B, rotated_y_B, current_hex_angle_B)
        hex_polygons.append(Polygon(vertices_B))
        all_vertices_list.append(vertices_B) # Append numpy array

    # Check for interior overlaps (touching is allowed, but interiors must not overlap)
    # shapely.Polygon.overlaps() checks if the intersection of the interiors is non-empty
    # and the intersection of the objects themselves is not equal to either object.
    for i in range(len(hex_polygons)):
        for j in range(i + 1, len(hex_polygons)):
            if hex_polygons[i].overlaps(hex_polygons[j]):
                return np.inf # Penalty for interior overlap

    # Concatenate all vertex arrays into a single (N,2) array for the JITted function
    all_vertices_combined = np.vstack(all_vertices_list)
    
    # Calculate the minimum outer hexagon side length using the JITted helper
    min_R_outer = calculate_min_R_outer(all_vertices_combined)
    
    return min_R_outer

def hexagon_packing_12():
    """ 
    Constructs an optimal packing of 12 disjoint unit regular hexagons inside a larger regular hexagon,
    maximizing 1/outer_hex_side_length (minimizing outer_hex_side_length).
    Returns:
        inner_hex_data: np.ndarray of shape (12,3), where each row is (x, y, angle_degrees).
        outer_hex_data: np.ndarray of shape (3,) of form (x,y,angle_degree) (fixed at [0,0,0]).
        outer_hex_side_length: float representing the side length of the outer hexagon.
    """
    # Define bounds for the 6 parameters: [x_A, y_A, theta_A, x_B, y_B, theta_B]
    # Define bounds for the 6 parameters: [x_A, y_A, theta_A, x_B, y_B, theta_B]
    # Based on an estimated R_outer around 3.93, a hexagon center can be up to R_outer - HEX_APOTHEM
    # (approx 3.93 - 0.866 = 3.064). Expanded bounds to 3.5 to ensure the optimal solution is not cut off.
    # Theta bounds [0, 60) degrees are sufficient due to 60-degree rotational symmetry of a hexagon.
    bounds = [(-3.5, 3.5),   # x_A (expanded bounds for wider search)
              (-3.5, 3.5),   # y_A (expanded bounds for wider search)
              (0.0, 59.999), # theta_A (degrees, restricted to 0-60 for unique orientation)
              (-3.5, 3.5),   # x_B (expanded bounds for wider search)
              (-3.5, 3.5),   # y_B (expanded bounds for wider search)
              (0.0, 59.999)] # theta_B (degrees, restricted to 0-60 for unique orientation)

    # Run Differential Evolution for global optimization
    # Parameters are chosen for a balance of exploration and convergence.
    # `seed` for reproducibility.
    result = differential_evolution(objective_function, bounds, 
                                    strategy='best1bin', # A robust strategy for global search
                                    maxiter=20000,        # Increased iterations for a deeper search
                                    popsize=30,           # Population size multiplier
                                    tol=1e-5,             # Tighter tolerance for better convergence
                                    mutation=(0.5, 1),    # Mutation factor range
                                    recombination=0.7,    # Crossover probability
                                    seed=42,              # For reproducibility
                                    disp=False,           # Set to True for verbose output during development
                                    workers=-1            # Use all available CPU cores for parallel evaluation
                                   )

    optimal_params = result.x
    optimal_R = result.fun

    # Construct the final configuration from the optimal parameters
    x_A, y_A, theta_A = optimal_params[0], optimal_params[1], optimal_params[2]
    x_B, y_B, theta_B = optimal_params[3], optimal_params[4], optimal_params[5]

    inner_hex_data_list = []

    # Strategy: Two rings of 6 hexagons each, generated with D6 (6-fold rotational) symmetry.
    # This ensures an overall D6 symmetric packing of 12 hexagons.

    # H0-H5: First Ring
    for i in range(6):
        rot_angle_deg = i * 60
        rot_angle_rad = np.deg2rad(rot_angle_deg)
        cos_a = np.cos(rot_angle_rad)
        sin_a = np.sin(rot_angle_rad)
        rotated_x_A = x_A * cos_a - y_A * sin_a
        rotated_y_A = x_A * sin_a + y_A * cos_a
        current_hex_angle_A = theta_A + rot_angle_deg
        inner_hex_data_list.append([rotated_x_A, rotated_y_A, current_hex_angle_A])

    # H6-H11: Second Ring
    for i in range(6): # Changed from 5 hexagons with 72-degree rotations to 6 hexagons with 60-degree rotations
        rot_angle_deg = i * 60 # Changed from 72 to 60
        rot_angle_rad = np.deg2rad(rot_angle_deg)
        cos_a = np.cos(rot_angle_rad)
        sin_a = np.sin(rot_angle_rad)
        rotated_x_B = x_B * cos_a - y_B * sin_a
        rotated_y_B = x_B * sin_a + y_B * cos_a
        current_hex_angle_B = theta_B + rot_angle_deg
        inner_hex_data_list.append([rotated_x_B, rotated_y_B, current_hex_angle_B])

    inner_hex_data = np.array(inner_hex_data_list)
    outer_hex_data = np.array([0, 0, 0]) # Outer hexagon is fixed at origin, unrotated
    outer_hex_side_length = optimal_R

    return inner_hex_data, outer_hex_data, outer_hex_side_length
# EVOLVE-BLOCK-END