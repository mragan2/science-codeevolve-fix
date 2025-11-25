# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize, Bounds, NonlinearConstraint, basinhopping
from scipy.spatial.distance import pdist, squareform
from numba import jit # Added for performance

# Helper function for numba-accelerated overlap constraint calculation
@jit(nopython=True, cache=True)
def _calculate_overlap_cons_numba(circles_array):
    n_circles = circles_array.shape[0]
    num_overlap_constraints = n_circles * (n_circles - 1) // 2
    overlap_cons = np.empty(num_overlap_constraints, dtype=circles_array.dtype)
    
    idx = 0
    for i in range(n_circles):
        x_i, y_i, r_i = circles_array[i, 0], circles_array[i, 1], circles_array[i, 2]
        for j in range(i + 1, n_circles):
            x_j, y_j, r_j = circles_array[j, 0], circles_array[j, 1], circles_array[j, 2]

            dist_sq = (x_i - x_j)**2 + (y_i - y_j)**2
            radii_sum = r_i + r_j
            # Constraint: dist_sq - (radii_sum)^2 >= 0
            overlap_cons[idx] = dist_sq - radii_sum**2
            idx += 1
    return overlap_cons


def circle_packing32() -> np.ndarray:
    """
    Places 32 non-overlapping circles in the unit square in order to maximize the sum of radii.
    This implementation uses a two-phase hybrid optimization strategy:
    1. Global Search: scipy's basinhopping explores the landscape to find a promising solution basin.
    2. Local Refinement: A final high-precision SLSQP run "polishes" the result from basinhopping.

    Returns:
        circles: np.array of shape (32,3), where the i-th row (x,y,r) stores the (x,y) coordinates of the i-th circle of radius r.
    """
    n_circles = 32

    # For reproducibility of the initial guess
    np.random.seed(42)

    # 1. INITIALIZATION: Create a perturbed grid.
    # A grid layout helps to spread circles out initially.
    # A 6x6 grid provides 36 points, we take the first 32.
    grid_size = int(np.ceil(np.sqrt(n_circles)))  # This will be 6
    x_coords, y_coords = np.meshgrid(
        np.linspace(0.1, 0.9, grid_size),
        np.linspace(0.1, 0.9, grid_size)
    )
    initial_centers = np.vstack([x_coords.ravel(), y_coords.ravel()]).T
    initial_centers = initial_centers[:n_circles]

    # Add small random perturbations to break symmetry and avoid degenerate cases.
    initial_centers += np.random.uniform(-0.02, 0.02, initial_centers.shape)

    # Start with varied radii to promote diversity, within a reasonable range.
    # Average radius for 32 circles might be around 0.1, so 0.03-0.09 is a good starting range.
    initial_radii = np.random.uniform(0.03, 0.09, n_circles)

    # The optimizer works with a flat 1D array of variables (x0, y0, r0, x1, y1, r1, ...).
    initial_guess = np.hstack([initial_centers, initial_radii.reshape(-1, 1)]).ravel()

    # 2. OBJECTIVE FUNCTION: Maximize sum of radii -> Minimize negative sum of radii.
    def objective_func(variables):
        # Radii are every third element starting from index 2.
        radii = variables[2::3]
        return -np.sum(radii)

    # 3. BOUNDS: Define simple bounds for each variable.
    # 0 <= x_i, y_i <= 1
    # 0 <= r_i <= 0.5 (a circle cannot have a radius > 0.5 in a unit square)
    lower_bounds = np.zeros_like(initial_guess)
    upper_bounds = np.ones_like(initial_guess)
    # Set a small positive lower bound for radii to prevent degenerate circles.
    lower_bounds[2::3] = 1e-4 
    upper_bounds[2::3] = 0.5  # Set max radius for all r_i
    bounds = Bounds(lower_bounds, upper_bounds)

    # 4. CONSTRAINTS: Define non-linear constraints.
    def constraint_func(variables):
        circles = variables.reshape((n_circles, 3))
        x, y, r = circles[:, 0], circles[:, 1], circles[:, 2]

        # a) Containment constraints: all circles must be inside the square.
        #    g(x) >= 0, so we formulate as:
        #    x_i - r_i >= 0
        #    1 - (x_i + r_i) >= 0  ->  1 - x_i - r_i >= 0
        #    y_i - r_i >= 0
        #    1 - (y_i + r_i) >= 0  ->  1 - y_i - r_i >= 0
        containment_cons = np.concatenate([
            x - r,
            1 - x - r,
            y - r,
            1 - y - r
        ])

        # b) Non-overlap constraints: circles must not overlap.
        #    sqrt((xi-xj)^2 + (yi-yj)^2) >= ri + rj
        #    (xi-xj)^2 + (yi-yj)^2 - (ri+rj)^2 >= 0
        # Using numba-accelerated helper for overlap constraints
        overlap_cons = _calculate_overlap_cons_numba(circles)

        return np.concatenate([containment_cons, overlap_cons])

    # All constraints are of the form g(x) >= 0.
    num_containment = n_circles * 4
    num_overlap = n_circles * (n_circles - 1) // 2
    constraint_lower_bounds = np.zeros(num_containment + num_overlap)
    constraint_upper_bounds = np.inf * np.ones(num_containment + num_overlap)

    nonlinear_constraint = NonlinearConstraint(constraint_func, constraint_lower_bounds, constraint_upper_bounds)

    # 5. OPTIMIZATION PHASE 1: Global search with Basin-hopping.
    # Define local minimization options for basinhopping's steps.
    # These are kept relatively loose to allow the global search to proceed faster.
    # Increased maxiter for local steps during basinhopping to allow deeper exploration of local basins.
    local_minimizer_options = {'maxiter': 200, 'ftol': 1e-7, 'disp': False}
    minimizer_kwargs = {
        "method": "SLSQP",
        "bounds": bounds,
        "constraints": [nonlinear_constraint],
        "options": local_minimizer_options
    }

    # Run basinhopping to find a good solution basin.
    # Increased niter and T for more thorough global search, leveraging available eval_time.
    bh_result = basinhopping(
        objective_func,
        initial_guess,
        minimizer_kwargs=minimizer_kwargs,
        niter=150,  # Increased number of global search iterations
        T=2.0,      # Increased temperature for more uphill moves and broader exploration
        stepsize=0.05,
        disp=False,
        seed=42
    )

    # 6. OPTIMIZATION PHASE 2: High-precision local refinement ("polishing").
    # Use the best result from basin-hopping as the starting point for a final, more precise run.
    best_guess_from_bh = bh_result.x

    # Use stricter tolerance and more iterations for the final polish.
    polish_options = {'maxiter': 2000, 'ftol': 1e-9, 'disp': False}
    final_result = minimize(
        objective_func,
        best_guess_from_bh,
        method='SLSQP',
        bounds=bounds,
        constraints=[nonlinear_constraint],
        options=polish_options
    )

    # 7. RETURN RESULT: Reshape the optimized variables back to (N, 3) format.
    if final_result.success:
        final_circles = final_result.x.reshape((n_circles, 3))
    else:
        # If the final polishing fails, it might still be a valid (though not fully converged) state.
        # We fall back to the result from basinhopping, which is guaranteed to be the best found so far.
        print(f"Final polishing optimization failed: {final_result.message}. Falling back to basinhopping result.")
        final_circles = bh_result.x.reshape((n_circles, 3))

    return final_circles


# EVOLVE-BLOCK-END
