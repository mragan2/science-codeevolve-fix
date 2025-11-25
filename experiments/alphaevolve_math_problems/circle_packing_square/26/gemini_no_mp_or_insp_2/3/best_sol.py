# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
from scipy.spatial.distance import pdist, squareform


def circle_packing26() -> np.ndarray:
    """
    Places 26 non-overlapping circles in the unit square in order to maximize the sum of radii.
    This implementation uses a Sequential Least Squares Programming (SLSQP) optimizer
    to find a local optimum for the circle positions and radii.
    """
    n_circles = 26

    # The state is a flat vector X = [x0, y0, r0, x1, y1, r1, ...].
    # Total variables: n_circles * 3 = 78.

    def objective(X: np.ndarray) -> float:
        """Objective function: maximize sum of radii with small penalties for constraint violations."""
        circles = X.reshape((n_circles, 3))
        coords = circles[:, :2]
        radii = circles[:, 2]
        
        # Primary objective: maximize sum of radii
        primary_objective = -np.sum(radii)
        
        # Add small penalties to encourage staying away from constraint boundaries
        # These penalties guide the optimizer but are not the hard constraints, which are handled by `constraints`.
        penalty = 0.0
        # Keeping penalty_weight at 1e-4, as it's a good balance.
        penalty_weight = 1e-4 
        
        # Boundary penalties: violation is when r > x or r > 1-x.
        # We use np.maximum(0, ...) to only penalize violations.
        boundary_violations = np.concatenate([
            np.maximum(0, radii - coords[:, 0]),      # x - r < 0 implies violation
            np.maximum(0, radii - (1 - coords[:, 0])), # 1 - x - r < 0 implies violation
            np.maximum(0, radii - coords[:, 1]),      # y - r < 0 implies violation
            np.maximum(0, radii - (1 - coords[:, 1]))  # 1 - y - r < 0 implies violation
        ])
        penalty += penalty_weight * np.sum(boundary_violations**2)
        
        # Overlap penalties: violation is when dist < sum_radii.
        if n_circles > 1:
            distances = pdist(coords)
            # Vectorized calculation of all pairwise sum of radii for efficiency
            radii_matrix = radii[:, np.newaxis] + radii[np.newaxis, :]
            # Extract upper triangle (excluding diagonal) to match pdist output format
            radii_sums_flat = radii_matrix[np.triu_indices(n_circles, k=1)]
            
            overlap_violations = np.maximum(0, radii_sums_flat - distances)
            penalty += penalty_weight * np.sum(overlap_violations**2)
        
        return primary_objective + penalty

    def constraints(X: np.ndarray) -> np.ndarray:
        """
        Constraint function: all returned values must be >= 0 for a feasible solution.
        """
        circles = X.reshape((n_circles, 3))
        coords = circles[:, :2]
        radii = circles[:, 2]

        # 1. Boundary constraints: circle must be inside the unit square.
        # ri <= xi, ri <= 1-xi  =>  xi - ri >= 0, 1 - xi - ri >= 0
        # ri <= yi, ri <= 1-yi  =>  yi - ri >= 0, 1 - yi - r >= 0
        boundary_constraints = np.concatenate([
            coords[:, 0] - radii,      # x - r >= 0
            1 - coords[:, 0] - radii,  # 1 - x - r >= 0
            coords[:, 1] - radii,      # y - r >= 0
            1 - coords[:, 1] - radii   # 1 - y - r >= 0
        ])

        # 2. Non-overlap constraints: distance between centers >= sum of radii.
        # sqrt((xi-xj)^2 + (yi-yj)^2) >= ri+rj
        # To avoid sqrt, we use: (xi-xj)^2 + (yi-yj)^2 >= (ri+rj)^2
        # Constraint form: dist_sq - (sum_radii)^2 >= 0
        if n_circles > 1:
            # Pairwise squared Euclidean distances between centers
            sq_dist_matrix = squareform(pdist(coords, 'sqeuclidean'))
            
            # Pairwise sum of radii, squared
            sum_radii = radii[:, np.newaxis] + radii
            sq_sum_radii = sum_radii**2
            
            # Get the upper triangle of the difference matrix (to avoid redundant constraints)
            # The diagonal is zero, so we use k=1.
            overlap_constraints = (sq_dist_matrix - sq_sum_radii)[np.triu_indices(n_circles, k=1)]
        else:
            overlap_constraints = np.array([])
            
        return np.concatenate([boundary_constraints, overlap_constraints])

    # --- Optimization Setup ---
    # Bounds for each variable [xi, yi, ri]
    # 0 <= x, y <= 1
    # 0 <= r <= 0.5 (theoretical max for one circle)
    bounds = []
    for _ in range(n_circles):
        bounds.extend([(0, 1), (0, 1), (0, 0.5)])
        
    # Constraints dictionary for SLSQP
    cons = {'type': 'ineq', 'fun': constraints}

    # --- Hybrid Multi-start Optimization ---
    best_sum_radii = -np.inf
    best_circles_result = None
    
    # Set a fixed initial seed for the overall process, then use default_rng for isolated streams
    initial_global_seed = 42

    def generate_structured_init(strategy: str, n_circles: int, rng: np.random.Generator) -> np.ndarray:
        """Generate structured initial configurations based on different strategies."""
        x0_base = np.zeros(n_circles * 3)
        
        positions = []
        radii = []

        if strategy == "corner_focused":
            # Place larger circles in corners, smaller ones elsewhere
            corners = [(0.15, 0.15), (0.85, 0.15), (0.15, 0.85), (0.85, 0.85)]
            for i, (cx, cy) in enumerate(corners[:min(4, n_circles)]):
                positions.append([cx, cy])
                radii.append(0.15)  # Increased radius from 0.14 to 0.15
            
            # Fill remaining positions randomly with slightly larger radii
            for i in range(len(positions), n_circles):
                x = rng.uniform(0.08, 0.92)
                y = rng.uniform(0.08, 0.92)
                positions.append([x, y])
                radii.append(0.06)  # Increased filler radius from 0.05 to 0.06
                
        elif strategy == "edge_focused":
            # Place circles along edges first
            edge_positions = [
                (0.1, 0.5), (0.9, 0.5), (0.5, 0.1), (0.5, 0.9),  # Center of edges
                (0.3, 0.1), (0.7, 0.1), (0.3, 0.9), (0.7, 0.9),  # Along top/bottom
                (0.1, 0.3), (0.1, 0.7), (0.9, 0.3), (0.9, 0.7)   # Along left/right
            ]
            
            for i, (ex, ey) in enumerate(edge_positions[:min(len(edge_positions), n_circles)]):
                positions.append([ex, ey])
                radii.append(0.1) # Increased edge radius from 0.095 to 0.1
            
            # Fill remaining positions with slightly larger radii
            for i in range(len(positions), n_circles):
                x = rng.uniform(0.2, 0.8)
                y = rng.uniform(0.2, 0.8)
                positions.append([x, y])
                radii.append(0.06) # Increased filler radius from 0.05 to 0.06
                
        elif strategy == "hexagonal_inspired":
            # Enhanced hexagonal pattern for N=26 with better radius calculation
            hex_configs = [
                [4, 5, 6, 5, 4, 2],  # 26 circles total
                [5, 6, 5, 6, 4],     # 26 circles total  
                [4, 6, 6, 6, 4],     # 26 circles total
                [5, 5, 6, 5, 5],     # 26 circles total
                [6, 5, 4, 5, 6],     # 26 circles total
                [5, 6, 6, 5, 4],     # Added new configuration
                [4, 6, 5, 6, 5],     # Added new configuration
            ]
            
            best_r = 0
            best_config = None
            
            for config in hex_configs:
                if sum(config) != n_circles:
                    continue
                    
                num_rows = len(config)
                max_cols = max(config)
                
                # Calculate the minimum radius needed for this configuration to fit
                # assuming ideal hexagonal packing.
                # Max width required: 2 * r + (max_cols - 1) * 2 * r = 2 * max_cols * r
                # Max height required: 2 * r + (num_rows - 1) * np.sqrt(3) * r
                
                r_from_width = 1.0 / (2 * max_cols)
                r_from_height = 1.0 / (2 + (num_rows - 1) * np.sqrt(3))
                
                config_r = min(r_from_width, r_from_height)
                
                if config_r > best_r:
                    best_r = config_r
                    best_config = config
            
            # Use the best configuration found
            if best_config is not None:
                # Use a slightly aggressive factor for initial radius, let optimizer adjust
                r_effective = best_r * 0.99 # Increased from 0.98 to 0.99 for a tighter initial packing
                row_counts = best_config
                num_rows = len(row_counts)
                
                # Calculate total height of the pattern
                total_height = (num_rows - 1) * np.sqrt(3) * r_effective + 2 * r_effective
                y_start = (1 - total_height) / 2 + r_effective # Center vertically
                
                # Calculate max pattern width for horizontal centering
                max_pattern_width = 2 * max_cols * r_effective
                x_pattern_start = (1 - max_pattern_width) / 2 # Center horizontally
                
                current_circle_idx = 0
                for row_idx, num_in_row in enumerate(row_counts):
                    if current_circle_idx >= n_circles:
                        break
                    
                    x_offset_row = (row_idx % 2) * r_effective # Staggering offset
                    
                    for col_idx in range(num_in_row):
                        if current_circle_idx >= n_circles:
                            break
                        
                        cx = x_pattern_start + x_offset_row + r_effective + col_idx * 2 * r_effective
                        cy = y_start + row_idx * np.sqrt(3) * r_effective
                        
                        positions.append([cx, cy])
                        radii.append(r_effective)
                        current_circle_idx += 1
            else:
                for i in range(n_circles):
                    positions.append([rng.uniform(0.05, 0.95), rng.uniform(0.05, 0.95)])
                    radii.append(0.06) # Consistent with other filler radii

        elif strategy == "grid_staggered_inspired":
            # Modified for 26 circles by starting with a 5x6 grid and removing 4 corners.
            rows = 5
            cols = 6
            
            # Calculate base radius to fit a 5x6 grid. Max dimension dictates r.
            # Max width for 6 circles (staggered or not) needs 2r * cols space.
            # Max height for 5 circles needs 2r * rows space.
            r_base = 1.0 / (2 * max(rows, cols))
            r_grid = r_base * 0.98 # Increased radius from 0.97 to 0.98 for tighter packing
            
            all_generated_positions = []
            all_generated_radii = []

            for row in range(rows):
                for col in range(cols):
                    # Initial center calculation for a grid that starts at 0,0
                    cx = r_grid + col * 2 * r_grid 
                    cy = r_grid + row * 2 * r_grid
                    
                    if row % 2 == 1: # Staggering
                        cx += r_grid 
                    
                    all_generated_positions.append([cx, cy])
                    all_generated_radii.append(r_grid)
            
            # Select 26 circles from the 30 generated by removing 4 corners
            # Indices for a 5x6 grid (row-major): (0,0), (0,5), (4,0), (4,5)
            indices_to_remove = {0, 5, 24, 29} 
            
            positions = []
            radii = []
            for i in range(len(all_generated_positions)):
                if i not in indices_to_remove:
                    positions.append(all_generated_positions[i])
                    radii.append(all_generated_radii[i])
            
            # This should result in exactly 26 circles for n_circles=26.
            if len(positions) != n_circles:
                print(f"WARNING: grid_staggered_inspired generated {len(positions)} circles, expected {n_circles}. Falling back to random selection.")
                positions = []
                radii = []
                for i in range(n_circles):
                    positions.append([rng.uniform(0.05, 0.95), rng.uniform(0.05, 0.95)])
                    radii.append(0.06) # Consistent with other filler radii

            # Scale to unit square, ensuring a minimal margin
            all_x_coords = np.array([p[0] for p in positions])
            all_y_coords = np.array([p[1] for p in positions])
            all_radii = np.array(radii)

            min_x_packed = np.min(all_x_coords - all_radii)
            max_x_packed = np.max(all_x_coords + all_radii)
            min_y_packed = np.min(all_y_coords - all_radii)
            max_y_packed = np.max(all_y_coords + all_radii)
            
            current_span_x = max_x_packed - min_x_packed
            current_span_y = max_y_packed - min_y_packed
            
            target_span = 0.99 # Increased from 0.95 to 0.99 for a tighter initial fit

            scale_factor = target_span / max(current_span_x, current_span_y) if max(current_span_x, current_span_y) > 0 else 1.0
            
            for i in range(len(positions)):
                positions[i][0] = (positions[i][0] - min_x_packed) * scale_factor
                positions[i][1] = (positions[i][1] - min_y_packed) * scale_factor
                radii[i] = radii[i] * scale_factor

            offset_x = (1 - target_span) / 2
            offset_y = (1 - target_span) / 2

            for i in range(len(positions)):
                positions[i][0] += offset_x
                positions[i][1] += offset_y
            
            radii = [np.clip(r, 1e-5, 0.5) for r in radii]


        elif strategy == "central_anchor_fill":
            if n_circles >= 1:
                positions.append([0.5, 0.5])
                radii.append(0.17) # Increased central circle radius from 0.16 to 0.17
            
            num_remaining = n_circles - len(positions)
            if num_remaining > 0:
                angle_step = 2 * np.pi / max(1, num_remaining)
                current_spiral_radius = 0.3 # Increased start distance from center from 0.28 to 0.3
                radius_increment = 0.04 # Slightly increased increment from 0.035 to 0.04
                
                for i in range(num_remaining):
                    angle = i * angle_step
                    x = 0.5 + current_spiral_radius * np.cos(angle)
                    y = 0.5 + current_spiral_radius * np.sin(angle)
                    
                    positions.append([x, y])
                    radii.append(0.06) # Increased filler radii from 0.055 to 0.06
                    
                    if (i + 1) % 5 == 0:
                        current_spiral_radius += radius_increment
                        
            for i in range(len(positions)):
                r_val = radii[i]
                x_val = positions[i][0]
                y_val = positions[i][1]
                
                positions[i][0] = np.clip(x_val, r_val, 1 - r_val)
                positions[i][1] = np.clip(y_val, r_val, 1 - r_val)
                radii[i] = np.clip(r_val, 1e-5, 0.5)


        elif strategy == "size_hierarchical":
            large_positions = [(0.25, 0.25), (0.75, 0.75), (0.25, 0.75), (0.75, 0.25)]
            for i, (lx, ly) in enumerate(large_positions[:min(4, n_circles)]):
                positions.append([lx, ly])
                radii.append(0.17)  # Increased large radius from 0.16 to 0.17
            for i in range(len(positions), n_circles):
                x = rng.uniform(0.05, 0.95)
                y = rng.uniform(0.05, 0.95)
                positions.append([x, y])
                radii.append(0.05)  # Increased very small radius from 0.045 to 0.05
                
        else:  # "random" fallback
            positions = [[rng.uniform(0.05, 0.95), rng.uniform(0.05, 0.95)] for _ in range(n_circles)]
            radii = [0.055] * n_circles # Increased random initial radius from 0.05 to 0.055
        
        for i in range(n_circles):
            x0_base[3*i] = positions[i][0]
            x0_base[3*i + 1] = positions[i][1]
            x0_base[3*i + 2] = radii[i]
            
        return x0_base

    # Define initialization strategies
    strategies = [
        "corner_focused", "edge_focused", "hexagonal_inspired", 
        "size_hierarchical", "grid_staggered_inspired", "central_anchor_fill"
    ] * 12  # Increased repetitions from 10 to 12 for more exploration
    
    # Add more random initializations to increase diversity
    strategies.extend(["random"] * 40) # Increased random restarts from 35 to 40
    
    total_restarts = len(strategies)
    
    for i, strategy in enumerate(strategies):
        rng = np.random.default_rng(seed=initial_global_seed + i) 
        
        x0 = generate_structured_init(strategy, n_circles, rng)
        
        # Apply small perturbation to initial guess for better exploration
        perturbation_scale_xy = 0.02 # Increased perturbation for coordinates from 0.015 to 0.02
        perturbation_scale_r = 0.007  # Increased perturbation for radii from 0.005 to 0.007

        x0_perturbed = x0.copy()
        x0_perturbed[0::3] += rng.uniform(-perturbation_scale_xy, perturbation_scale_xy, n_circles)
        x0_perturbed[1::3] += rng.uniform(-perturbation_scale_xy, perturbation_scale_xy, n_circles)
        x0_perturbed[2::3] += rng.uniform(-perturbation_scale_r, perturbation_scale_r, n_circles)

        x0_perturbed[2::3] = np.clip(x0_perturbed[2::3], 1e-5, 0.5)
        for k in range(n_circles):
            r_k = x0_perturbed[3*k + 2]
            x0_perturbed[3*k] = np.clip(x0_perturbed[3*k], r_k, 1 - r_k)
            x0_perturbed[3*k + 1] = np.clip(x0_perturbed[3*k + 1], r_k, 1 - r_k)
        
        # --- Run the Optimizer ---
        options = {'maxiter': 5000, 'ftol': 1e-13, 'disp': False} # Increased maxiter from 4000 to 5000, tightened ftol from 1e-12 to 1e-13
            
        result = minimize(
            objective, 
            x0_perturbed,
            method='SLSQP', 
            bounds=bounds, 
            constraints=cons, 
            options=options
        )

        if result.success:
            result_circles = result.x.reshape((n_circles, 3))
            current_sum_radii = np.sum(result_circles[:, 2])
            if current_sum_radii > best_sum_radii:
                best_sum_radii = current_sum_radii
                best_circles_result = result_circles

    if best_circles_result is not None:
        x0_refine = best_circles_result.flatten()
        
        # The objective function already has minimal penalties, so we can use a truly pure one
        # for refinement to ensure sum of radii is maximized without any penalty influence.
        def objective_pure(X: np.ndarray) -> float:
            radii = X[2::3]
            return -np.sum(radii)
        
        options_refine = {'maxiter': 2500, 'ftol': 1e-13, 'disp': False} # Increased maxiter from 2000 to 2500, tightened ftol from 1e-12 to 1e-13
        
        result_refine = minimize(
            objective_pure,
            x0_refine,
            method='SLSQP',
            bounds=bounds,
            constraints=cons,
            options=options_refine
        )
        
        if result_refine.success:
            refined_circles = result_refine.x.reshape((n_circles, 3))
            refined_sum_radii = np.sum(refined_circles[:, 2])
            if refined_sum_radii > best_sum_radii:
                best_sum_radii = refined_sum_radii
                best_circles_result = refined_circles

    if best_circles_result is None:
        # Fallback if all optimization restarts fail
        print("WARNING: All optimization restarts failed. Returning a default initial guess.")
        # Generate a final, deterministic initial guess as a fallback
        rng_fallback = np.random.default_rng(seed=initial_global_seed) # Ensure fallback is also deterministic
        initial_radius_start = 0.055 # Consistent with new random initial radius from "random" strategy
        initial_coords_x = rng_fallback.uniform(initial_radius_start, 1 - initial_radius_start, n_circles)
        initial_coords_y = rng_fallback.uniform(initial_radius_start, 1 - initial_radius_start, n_circles)
        initial_radii = np.full(n_circles, initial_radius_start)
        x0_fallback = np.zeros(n_circles * 3)
        x0_fallback[0::3] = initial_coords_x
        x0_fallback[1::3] = initial_coords_y
        x0_fallback[2::3] = initial_radii
        return x0_fallback.reshape((n_circles, 3))
    
    return best_circles_result


# EVOLVE-BLOCK-END
