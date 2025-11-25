# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize, differential_evolution, NonlinearConstraint
from scipy.spatial.distance import pdist
from numba import njit # For performance critical parts

# Define constants
N_CIRCLES = 26
RANDOM_SEED = 42

# --- Optimization Helper Functions ---

# Numba-jitted function for evaluating constraints
# This will significantly speed up constraint checks, especially the overlap part.
@njit
def _evaluate_constraints_numba(x_flat, n_circles):
    circles = x_flat.reshape(n_circles, 3)
    x = circles[:, 0]
    y = circles[:, 1]
    r = circles[:, 2]

    # Boundary Containment Constraints (x-r >= 0, 1-x-r >= 0, y-r >= 0, 1-y-r >= 0)
    boundary_constraints_list = []
    for i in range(n_circles):
        boundary_constraints_list.append(x[i] - r[i])
        boundary_constraints_list.append(1.0 - x[i] - r[i]) # Use 1.0 for float literal
        boundary_constraints_list.append(y[i] - r[i])
        boundary_constraints_list.append(1.0 - y[i] - r[i])
    
    # Non-overlap Constraints ((xi-xj)^2 + (yi-yj)^2 - (ri+rj)^2 >= 0)
    overlap_constraints_list = []
    for i in range(n_circles):
        for j in range(i + 1, n_circles):
            dist_sq = (x[i] - x[j])**2 + (y[i] - y[j])**2
            min_dist_sq = (r[i] + r[j])**2
            overlap_constraints_list.append(dist_sq - min_dist_sq)

    return np.array(boundary_constraints_list + overlap_constraints_list)

def objective_function(x_flat):
    """
    Objective: maximize sum of radii.
    Minimize -sum(radii)
    x_flat is [x1, y1, r1, ..., xN, yN, rN]
    """
    radii = x_flat[2::3]
    return -np.sum(radii)

def penalty_objective(x_flat, penalty_factor=1e6):
    """
    Objective function for global optimization (e.g., differential_evolution)
    incorporating penalties for constraint violations.
    """
    obj = objective_function(x_flat) # -sum(radii)

    # Get constraint values (all should be >= 0)
    constraints_values = _evaluate_constraints_numba(x_flat, N_CIRCLES)

    # Calculate violations: max(0, -value) for inequalities
    violations = np.maximum(0, -constraints_values)
    total_penalty = penalty_factor * np.sum(violations)

    return obj + total_penalty

def initial_grid_layout(n, square_dim=1.0):
    """
    Generates an initial grid-like layout for n circles.
    For N=26, uses a 5x6 grid and fills the first 26 positions.
    """
    num_rows = 5
    num_cols = 6

    # Calculate radius based on the grid dimensions
    # This radius ensures that circles placed in a num_rows x num_cols grid
    # will fit and touch if they were all of this radius.
    radius = min(square_dim / (2 * num_cols), square_dim / (2 * num_rows))

    circles = []
    k = 0
    # Iterate through the grid cells to place N circles
    for i in range(num_rows):
        for j in range(num_cols):
            if k < n: # Only place N circles
                x_center = (2 * j + 1) * radius
                y_center = (2 * i + 1) * radius
                circles.append([x_center, y_center, radius])
                k += 1
            else:
                break
        if k >= n:
            break
            
    return np.array(circles).flatten()

# New function for a hexagonal-like initial layout
def initial_hexagonal_layout(n, square_dim=1.0):
    """
    Generates a hexagonal-like initial layout for N_CIRCLES (specifically 26).
    Uses a 5-row pattern: 5, 6, 5, 6, 4 circles.
    """
    if n != N_CIRCLES:
        # Fallback if N_CIRCLES is not 26
        return initial_grid_layout(n, square_dim)

    row_counts = [5, 6, 5, 6, 4] # Total 26 circles
    
    # Determine initial radius based on the tightest packing constraint for a shifted row
    # Max circles in a shifted row is 6.
    # For a shifted row of `m` circles, `r <= square_dim / (2 * m + 2)`.
    # So for m=6, `r <= 1 / (2*6 + 2) = 1/14 = 0.071428...`
    # Use a specific value known to work well for this pattern to ensure initial fit:
    initial_r = 0.071 

    circles = []
    y_current = initial_r # Start y position, accounting for radius

    # Vertical step for hexagonal packing (center-to-center)
    y_step = initial_r * np.sqrt(3) 

    # Horizontal step (center-to-center)
    x_step = 2 * initial_r 

    for row_idx, num_in_row in enumerate(row_counts):
        # Calculate the horizontal span covered by the centers in this row
        row_span_width = x_step * (num_in_row - 1)
        
        # Calculate the base x-offset to center this row
        x_offset_center = (square_dim - row_span_width) / 2
        
        # Apply hexagonal staggering: shift odd rows by half a horizontal step (initial_r)
        if row_idx % 2 == 1: # Odd rows (1, 3) are typically shifted
            x_current = x_offset_center + initial_r
        else: # Even rows (0, 2, 4) are not shifted relative to the square's center
            x_current = x_offset_center

        # Add circles for the current row
        for _ in range(num_in_row):
            circles.append([x_current, y_current, initial_r])
            x_current += x_step
        
        y_current += y_step
        
    return np.array(circles).flatten()


def circle_packing26() -> np.ndarray:
    """
    Places 26 non-overlapping circles in the unit square in order to maximize the sum of radii.

    Returns:
        circles: np.array of shape (26,3), where the i-th row (x,y,r) stores the (x,y) coordinates of the i-th circle of radius r.
    """
    n = N_CIRCLES
    
    # 1. Define bounds for all parameters (x, y, r)
    # x and y coordinates must be within [0, 1]
    # radii must be positive and cannot exceed 0.5 (for a single circle in a unit square)
    bounds = [(0, 1), (0, 1), (0, 0.5)] * n

    # 2. Generate an initial guess (hexagonal layout for better starting point)
    x0_initial = initial_hexagonal_layout(n)

    # 3. Global Optimization using Differential Evolution
    # This step tries to find a good starting point in the complex search space,
    # handling constraints with a penalty function.
    print("Starting Differential Evolution for global search...")
    de_result = differential_evolution(
        func=penalty_objective,
        bounds=bounds,
        popsize=15,          # Population size multiplier (N_params * popsize)
        maxiter=2000,        # Increased max iterations for global search
        mutation=(0.5, 1.0), # Mutation factor range
        recombination=0.7,   # Crossover probability
        strategy='best1bin', # DE strategy
        seed=RANDOM_SEED,
        tol=0.001,           # Tightened tolerance for DE convergence
        disp=True,           # Display progress
        workers=-1,          # Use all available cores
        x0=x0_initial        # Use the hexagonal layout as initial population seed
    )
    print(f"Differential Evolution finished. Best value: {-de_result.fun:.6f}")
    
    # Extract the best solution from DE
    x_de_best = de_result.x

    # 4. Local Optimization using SLSQP with NonlinearConstraint
    # This step refines the solution from DE to strictly satisfy constraints
    # and find a local optimum with higher precision.
    print("Starting SLSQP for local refinement...")
    
    # NonlinearConstraint requires the fun to return an array of constraint values.
    # lb=0, ub=np.inf means all constraints must be >= 0.
    nonlinear_constraint = NonlinearConstraint(
        fun=lambda x: _evaluate_constraints_numba(x, N_CIRCLES),
        lb=0,
        ub=np.inf,
        jac='2-point' # Use finite difference approximation for Jacobian
    )

    slsqp_result = minimize(
        fun=objective_function,
        x0=x_de_best, # Start from the DE result
        method='SLSQP',
        bounds=bounds,
        constraints=[nonlinear_constraint],
        options={'maxiter': 3000, 'ftol': 1e-9, 'disp': True} # Increased maxiter, tightened ftol
    )
    print(f"SLSQP finished. Best value: {-slsqp_result.fun:.6f}")

    # Check for successful optimization
    if not slsqp_result.success:
        print(f"Warning: SLSQP optimization did not converge successfully. Status: {slsqp_result.status}, Message: {slsqp_result.message}")
        # In case of failure, differential_evolution's result is still a good fallback,
        # though it might have minor constraint violations.
        final_x = de_result.x # Fallback to DE result if SLSQP refinement fails
    else:
        final_x = slsqp_result.x

    # Reshape the result into (N, 3) format
    circles = final_x.reshape(n, 3)
    
    # Post-processing: ensure radii are not negative due to numerical issues
    circles[:, 2] = np.maximum(0, circles[:, 2])

    return circles


# EVOLVE-BLOCK-END
