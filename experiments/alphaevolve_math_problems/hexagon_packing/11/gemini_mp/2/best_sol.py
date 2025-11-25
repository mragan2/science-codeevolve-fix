# EVOLVE-BLOCK-START
import numpy as np
# from shapely.geometry import Polygon # Shapely is no longer needed for collision detection in the objective function
from scipy.optimize import differential_evolution
import time
from numba import jit # Added numba for JIT compilation

# --- JIT-compiled function for hexagon vertex calculation ---
@jit(nopython=True, cache=True)
def _calculate_vertices_jit(center_x, center_y, rotation_rad, side_length):
    """
    Calculates the 6 vertices of a hexagon in a Numba-optimized way.
    """
    s = side_length
    vertices = np.empty((6, 2), dtype=np.float64)
    for i in range(6):
        angle = np.pi / 3 * i + rotation_rad
        vx = center_x + s * np.cos(angle)
        vy = center_y + s * np.sin(angle)
        vertices[i, 0] = vx
        vertices[i, 1] = vy
    return vertices

# --- JIT-compiled SAT (Separating Axis Theorem) for Hexagons ---
@jit(nopython=True, cache=True)
def _get_sat_axes_jit(rotation_rad):
    """
    Calculates the 3 unique normalized normal vectors for a regular hexagon's edges
    based on its rotation. These are the potential separating axes.
    """
    axes = np.empty((3, 2), dtype=np.float64)
    
    # For a flat-top hexagon (rotation_rad = 0):
    # The normals to its edges are at angles pi/6, pi/2, 5*pi/6 relative to the x-axis.
    # These angles correspond to the unique orientations of the hexagon's edges.
    
    # First axis (normal to the 'top-right' edge for unrotated hex)
    angle_offset_1 = np.pi / 6.0
    axes[0, 0] = np.cos(rotation_rad + angle_offset_1)
    axes[0, 1] = np.sin(rotation_rad + angle_offset_1)

    # Second axis (normal to the 'top' horizontal edge for unrotated hex)
    angle_offset_2 = np.pi / 2.0
    axes[1, 0] = np.cos(rotation_rad + angle_offset_2)
    axes[1, 1] = np.sin(rotation_rad + angle_offset_2)

    # Third axis (normal to the 'top-left' edge for unrotated hex)
    angle_offset_3 = 5.0 * np.pi / 6.0
    axes[2, 0] = np.cos(rotation_rad + angle_offset_3)
    axes[2, 1] = np.sin(rotation_rad + angle_offset_3)

    return axes

@jit(nopython=True, cache=True)
def _project_polygon_jit(vertices, axis):
    """
    Projects all vertices of a polygon onto an axis and returns the min/max projection.
    """
    min_proj = np.dot(vertices[0], axis)
    max_proj = min_proj
    for i in range(1, len(vertices)):
        proj = np.dot(vertices[i], axis)
        if proj < min_proj:
            min_proj = proj
        elif proj > max_proj:
            max_proj = proj
    return min_proj, max_proj

@jit(nopython=True, cache=True)
def _hex_overlap_sat_jit(vertices1, rotation1_rad, vertices2, rotation2_rad, tolerance=1e-9):
    """
    Checks for overlap between two hexagons using the Separating Axis Theorem (SAT).
    Returns the minimum overlap distance (a positive value if overlapping, 0.0 otherwise).
    This value can be used as a penalty metric.
    """
    axes1 = _get_sat_axes_jit(rotation1_rad)
    axes2 = _get_sat_axes_jit(rotation2_rad)
    
    # Store the minimum overlap found across all axes.
    # If it remains positive after checking all axes, polygons overlap.
    min_overlap_distance = np.inf 

    # Concatenate all potential separating axes for efficiency in the loop
    # Numba doesn't support np.vstack directly with jit(nopython=True) for arbitrary shapes,
    # so we will iterate over axes1 and axes2 separately.
    
    # Check axes from the first hexagon
    for i in range(axes1.shape[0]):
        axis = axes1[i]
        min1, max1 = _project_polygon_jit(vertices1, axis)
        min2, max2 = _project_polygon_jit(vertices2, axis)
        
        # Calculate overlap of the projected intervals
        # A positive value means overlap, a non-positive value means separation or touching.
        current_overlap = min(max1, max2) - max(min1, min2)
        
        if current_overlap <= tolerance:
            # If there's no significant overlap on this axis, they are separated or just touching.
            # This is a separating axis (or an axis where they touch), so no actual overlap for penalty.
            return 0.0 # No overlap, return 0 penalty

        # If we are here, there is overlap on this axis. Keep track of the minimum overlap.
        # The smallest overlap across all axes is a measure of "penetration depth".
        if current_overlap < min_overlap_distance:
            min_overlap_distance = current_overlap
            
    # Check axes from the second hexagon
    for i in range(axes2.shape[0]):
        axis = axes2[i]
        min1, max1 = _project_polygon_jit(vertices1, axis)
        min2, max2 = _project_polygon_jit(vertices2, axis)
        
        current_overlap = min(max1, max2) - max(min1, min2)
        
        if current_overlap <= tolerance:
            # If there's no significant overlap on this axis, they are separated or just touching.
            return 0.0 # No overlap, return 0 penalty

        if current_overlap < min_overlap_distance:
            min_overlap_distance = current_overlap
            
    # If we reached here, no separating axis was found (all current_overlap > tolerance).
    # Thus, the polygons are overlapping. Return the smallest overlap distance found.
    return min_overlap_distance

# --- Hexagon Class (kept for conceptual clarity, but not used in evaluate_packing) ---
# It's good practice to keep the class definition even if not directly used in the
# hot loop, as it might be useful for reconstruction, visualization, or external tooling.
class Hexagon:
    """
    Represents a unit regular hexagon with a given center and rotation.
    Side length is fixed at 1.0.
    """
    def __init__(self, center_x, center_y, rotation_rad, side_length=1.0):
        self.center_x = center_x
        self.center_y = center_y
        self.rotation_rad = rotation_rad
        self.side_length = side_length
        self._vertices = _calculate_vertices_jit(center_x, center_y, rotation_rad, side_length)
        # Note: self.polygon is removed as shapely is not used for collision detection anymore
        # If visualization or shapely-based operations are needed, a separate function
        # would create shapely Polygon objects from the vertices.

    def get_vertices(self):
        """
        Returns the pre-calculated vertices (NumPy array).
        """
        return self._vertices

# --- Fully JIT-compiled Objective Function ---
@jit(nopython=True, cache=True)
def evaluate_packing_jit(params):
    """
    A fully JIT-compiled version of the objective function for maximum performance.
    This function avoids all Python overhead during its execution.
    """
    num_hexagons = 11
    side_length = 1.0
    tolerance = 1e-9
    sqrt3 = np.sqrt(3.0)

    # Pre-allocate arrays to avoid list creation and other Python overhead
    all_hex_vertices = np.empty((num_hexagons, 6, 2), dtype=np.float64)
    all_hex_rotations = np.empty(num_hexagons, dtype=np.float64)

    # Step 1: Calculate vertices and rotations for all hexagons
    for i in range(num_hexagons):
        center_x = params[i * 3]
        center_y = params[i * 3 + 1]
        rotation_rad = params[i * 3 + 2]
        all_hex_rotations[i] = rotation_rad

        # Inlined vertex calculation for potential micro-optimization
        s = side_length
        for k in range(6):
            angle = np.pi / 3 * k + rotation_rad
            vx = center_x + s * np.cos(angle)
            vy = center_y + s * np.sin(angle)
            all_hex_vertices[i, k, 0] = vx
            all_hex_vertices[i, k, 1] = vy

    # Step 2: Calculate overlap penalty using the optimized SAT penetration depth
    overlap_penalty = 0.0
    # Penalty weight for overlaps. This value might need tuning.
    # A higher weight makes the optimizer prioritize avoiding overlaps more strongly.
    penalty_weight = 50000.0 # Significantly increased penalty weight to strongly discourage overlaps and ensure feasibility

    for i in range(num_hexagons):
        for j in range(i + 1, num_hexagons):
            # _hex_overlap_sat_jit returns 0.0 if no overlap, or a positive penetration depth if overlapping.
            overlap_depth = _hex_overlap_sat_jit(all_hex_vertices[i, :, :], all_hex_rotations[i],
                                                 all_hex_vertices[j, :, :], all_hex_rotations[j], tolerance)
            
            if overlap_depth > 0.0: # Add penalty only if there's actual penetration
                overlap_penalty += overlap_depth * penalty_weight

    # Step 3: Calculate outer hexagon side length R
    # Reshape is a zero-copy, highly efficient operation in Numba
    all_vertices_flat = all_hex_vertices.reshape(num_hexagons * 6, 2)
    
    if all_vertices_flat.shape[0] == 0:
        # This case should ideally not happen with 11 hexagons.
        # Returning a very large value to penalize empty configurations.
        return np.inf

    # Explicit loop to find the maximum R_candidate, avoiding unsupported np.vstack/np.max(axis=...)
    # The outer hexagon is assumed to be centered at (0,0) and unrotated.
    # Its side length R is determined by the maximum extent of any inner hexagon vertex
    # along the three principal axes of a regular hexagon.
    max_r = 0.0
    for i in range(all_vertices_flat.shape[0]):
        vx = all_vertices_flat[i, 0]
        vy = all_vertices_flat[i, 1]
        
        # Projections onto axes parallel to hex edges
        # Axis 1 (vertical): y-coordinate. Max extent is R*sqrt(3)/2. So R = abs(y) * 2/sqrt(3)
        r_candidate_1 = np.abs(vy) * 2.0 / sqrt3
        # Axis 2 (30 degrees from vertical, y = sqrt(3)x): project onto normal (sqrt(3)/2, 1/2)
        # For a vertex (x,y), projection is x*cos(30) + y*sin(30) + y*sin(30)
        # This simplifies to R = abs(x + y/sqrt(3)) for the flat-top hexagonal grid axes.
        r_candidate_2 = np.abs(vx + vy / sqrt3) 
        # Axis 3 (-30 degrees from vertical, y = -sqrt(3)x): project onto normal (-sqrt(3)/2, 1/2)
        # This simplifies to R = abs(x - y/sqrt(3)) for the flat-top hexagonal grid axes.
        r_candidate_3 = np.abs(vx - vy / sqrt3) 
        
        current_max_r_from_vertex = r_candidate_1
        if r_candidate_2 > current_max_r_from_vertex:
            current_max_r_from_vertex = r_candidate_2
        if r_candidate_3 > current_max_r_from_vertex:
            current_max_r_from_vertex = r_candidate_3
            
        if current_max_r_from_vertex > max_r:
            max_r = current_max_r_from_vertex
            
    # Objective is to minimize R plus any penalties.
    return max_r + overlap_penalty

# --- Python Wrapper for Scipy Optimizer ---
def evaluate_packing(params):
    """
    Thin Python wrapper that calls the high-performance JIT-compiled objective function.
    """
    return evaluate_packing_jit(params)

@jit(nopython=True, cache=True)
def get_geometric_side_length_jit(params):
    """
    Calculates the minimal outer hexagon side length for a given configuration
    of inner hexagons, without any overlap checks or penalties. This provides the
    pure geometric result from a set of parameters.
    """
    num_hexagons = 11
    side_length = 1.0
    sqrt3 = np.sqrt(3.0)

    all_hex_vertices = np.empty((num_hexagons, 6, 2), dtype=np.float64)

    # Step 1: Calculate vertices for all hexagons
    for i in range(num_hexagons):
        center_x = params[i * 3]
        center_y = params[i * 3 + 1]
        rotation_rad = params[i * 3 + 2]
        
        s = side_length
        for k in range(6):
            angle = np.pi / 3 * k + rotation_rad
            vx = center_x + s * np.cos(angle)
            vy = center_y + s * np.sin(angle)
            all_hex_vertices[i, k, 0] = vx
            all_hex_vertices[i, k, 1] = vy

    # Step 2: Calculate outer hexagon side length R
    all_vertices_flat = all_hex_vertices.reshape(num_hexagons * 6, 2)
    
    if all_vertices_flat.shape[0] == 0:
        return np.inf

    max_r = 0.0
    for i in range(all_vertices_flat.shape[0]):
        vx = all_vertices_flat[i, 0]
        vy = all_vertices_flat[i, 1]
        
        r_candidate_1 = np.abs(vy) * 2.0 / sqrt3
        r_candidate_2 = np.abs(vx + vy / sqrt3) 
        r_candidate_3 = np.abs(vx - vy / sqrt3) 
        
        current_max_r_from_vertex = r_candidate_1
        if r_candidate_2 > current_max_r_from_vertex:
            current_max_r_from_vertex = r_candidate_2
        if r_candidate_3 > current_max_r_from_vertex:
            current_max_r_from_vertex = r_candidate_3
            
        if current_max_r_from_vertex > max_r:
            max_r = current_max_r_from_vertex
            
    return max_r

# --- Main Packing Function ---
def hexagon_packing_11():
    """ 
    Constructs a packing of 11 disjoint unit regular hexagons inside a larger regular hexagon, maximizing 1/outer_hex_side_length. 
    Returns
        inner_hex_data: np.ndarray of shape (11,3), where each row is of the form (x, y, angle_degrees) containing the (x,y) coordinates and angle_degree of the respective inner hexagon.
        outer_hex_data: np.ndarray of shape (3,) of form (x,y,angle_degree) containing the (x,y) coordinates and angle_degree of the outer hexagon.
        outer_hex_side_length: float representing the side length of the outer hexagon.
    """
    n = 11
    
    # Define bounds for x, y, theta for each hexagon
    # x, y bounds: An optimal packing for N=11 has R around 3.931.
    # The max extent of an inner hexagon from its center is its circumradius (s=1).
    # So, the outer boundary of the packing could be R + s.
    # For R=3.931, this means up to ~4.931. So, [-5, 5] is a more appropriate range.
    xy_bounds = (-5.0, 5.0)
    # theta bounds: Due to 6-fold rotational symmetry of a regular hexagon,
    # rotations only need to be explored within [0, pi/3) (0 to 60 degrees exclusive).
    theta_bounds = (0.0, np.pi / 3.0)

    # Create a list of bounds for all 33 parameters (11 hexagons * 3 params each)
    # Relaxing the constraint on the first hexagon to allow for asymmetric optimal packings,
    # as the optimal N=11 packing is known not to be centrally symmetric.
    # The outer hexagon is still assumed to be centered at (0,0) and unrotated.
    bounds = [xy_bounds, xy_bounds, theta_bounds] * n

    # --- Initial Population Generation ---
    def _generate_initial_population(n_hexagons, bounds, popsize, seed):
        """
        Generates an initial population based on a compact, non-overlapping arrangement
        for 11 hexagons, with small random perturbations.
        Strategy 2: Refined initial configuration for the last 4 hexagons.
        """
        np.random.seed(seed)
        initial_pop_list = []

        # Base configuration for 11 hexagons: 1 central, 6 in first layer, 4 in second layer
        # Side length s=1. Distance between centers of touching hexagons is 2s=2.
        sqrt3 = np.sqrt(3.0)

        # Hexagon 0: Center (already fixed by bounds)
        base_coords = [(0.0, 0.0, 0.0)] # x, y, theta

        # Hexagons 1-6: First layer (6 hexagons)
        for i in range(6):
            angle = i * np.pi / 3.0
            x = 2.0 * np.cos(angle)
            y = 2.0 * np.sin(angle)
            base_coords.append((x, y, 0.0))

        # Hexagons 7-10: Second layer (4 hexagons) - more compact arrangement
        # This configuration is often found in optimal or near-optimal packings for N=11.
        # Original base_coords had all rotations at 0.0. We will introduce diversity below.
        base_coords.append((4.0, 0.0, 0.0))       # Extends the line from (0,0) -> (2,0)
        base_coords.append((3.0, sqrt3, 0.0))     # Fills a gap in the first layer
        base_coords.append((3.0, -sqrt3, 0.0))    # Fills another gap
        base_coords.append((-3.0, sqrt3, 0.0))    # A more balanced position

        assert len(base_coords) == n_hexagons, f"Initial base coords count mismatch: {len(base_coords)} vs {n_hexagons}"

        # Add small random perturbations and diverse rotations to the base configuration for each individual
        perturbation_scale_xy = 0.2 # Perturbation scale for x,y coordinates
        perturbation_scale_theta = np.pi / 12.0 # ~15 degrees, larger perturbation for theta to encourage finding optimal rotations

        for _ in range(popsize):
            individual = []
            for h_idx in range(n_hexagons):
                x_base, y_base, _ = base_coords[h_idx] # Ignore base_coords theta, generate dynamically

                # Apply perturbation and clip to bounds for x,y
                x_param_idx = h_idx * 3
                y_param_idx = h_idx * 3 + 1
                theta_param_idx = h_idx * 3 + 2

                x_perturbed = np.clip(x_base + np.random.uniform(-perturbation_scale_xy, perturbation_scale_xy), bounds[x_param_idx][0], bounds[x_param_idx][1])
                y_perturbed = np.clip(y_base + np.random.uniform(-perturbation_scale_xy, perturbation_scale_xy), bounds[y_param_idx][0], bounds[y_param_idx][1])
                
                # Introduce rotational diversity in the initial population
                # Randomly choose between 0 and pi/6 (30 degrees) as a base rotation for each hexagon
                # This helps explore optimal solutions known to have 30-degree rotations for N=11.
                theta_base_for_perturb = 0.0
                if np.random.rand() < 0.5: # 50% chance to start with a pi/6 base rotation
                    theta_base_for_perturb = np.pi / 6.0
                
                theta_perturbed = np.clip(theta_base_for_perturb + np.random.uniform(-perturbation_scale_theta, perturbation_scale_theta), bounds[theta_param_idx][0], bounds[theta_param_idx][1])

                individual.extend([x_perturbed, y_perturbed, theta_perturbed])
            initial_pop_list.append(individual)

        return np.array(initial_pop_list)

    # --- Differential Evolution optimization ---
    # This algorithm is suitable for global optimization of non-convex problems.
    # Parameters are chosen to balance exploration and convergence time.
    start_time = time.time()

    # Generate custom initial population
    # Strategy 3: Tune Differential Evolution parameters for a more thorough search.
    pop_size = 200 # Increased population size for better exploration (from 150)
    initial_population = _generate_initial_population(n, bounds, popsize=pop_size, seed=42)

    result = differential_evolution(
        evaluate_packing,
        bounds,
        init=initial_population, # Use custom initial population for faster convergence
        popsize=pop_size,        # Population size for exploration
        maxiter=150000,          # Max generations for a more exhaustive search (increased from 100k)
        mutation=0.8,            # Mutation rate for exploration
        recombination=0.9,       # Recombination rate for convergence
        strategy='randtobest1bin', # Use a strategy that balances exploration and exploitation
        workers=-1,              # Use all available CPU cores for parallel evaluation
        seed=42,                 # Fixed random seed for reproducibility
        disp=True,               # Display progress during optimization
        polish=True,             # Apply local optimization to the best solution found
        tol=1e-6,                # Further tighten tolerance for higher precision results
        # Added atol for absolute tolerance on objective function value for convergence.
        # This helps ensure the optimizer continues until the objective value
        # changes by less than this amount, leading to finer convergence.
        atol=1e-7                
    )
    eval_time = time.time() - start_time

    # Extract the best parameters found by the optimizer
    optimized_params = result.x
    # Re-calculate the outer hex side length from the best parameters *without* penalty.
    # result.fun includes the penalty term, so it's not the pure geometric result.
    outer_hex_side_length = get_geometric_side_length_jit(optimized_params)

    # Reconstruct inner_hex_data from optimized_params
    inner_hex_data = np.zeros((n, 3))
    for i in range(n):
        inner_hex_data[i, 0] = optimized_params[i * 3]     # x-coordinate
        inner_hex_data[i, 1] = optimized_params[i * 3 + 1] # y-coordinate
        # Convert rotation from radians (used in optimization) to degrees (for output)
        inner_hex_data[i, 2] = np.degrees(optimized_params[i * 3 + 2]) 

    # The outer hexagon is assumed to be centered at origin and unrotated for the objective function.
    outer_hex_data = np.array([0, 0, 0]) 

    # Final verification: re-evaluate the objective function on the best solution
    # to check for any remaining overlap penalty.
    final_objective_value = evaluate_packing(optimized_params)
    final_penalty = final_objective_value - outer_hex_side_length
    if final_penalty > 1e-6: # Allow for tiny floating point differences
        print(f"Warning: Final solution has a non-zero overlap penalty of {final_penalty:.2e}.")
        # This indicates the solution might have a very small, residual overlap.
        # The optimizer may terminate here if further reduction in overlap
        # would lead to a worse overall objective value (larger R).

    print(f"Optimization finished in {eval_time:.2f} seconds.")
    print(f"Optimized outer_hex_side_length: {outer_hex_side_length:.4f}")
    print(f"1/outer_hex_side_length: {1/outer_hex_side_length:.4f}")
    
    return inner_hex_data, outer_hex_data, outer_hex_side_length
# EVOLVE-BLOCK-END