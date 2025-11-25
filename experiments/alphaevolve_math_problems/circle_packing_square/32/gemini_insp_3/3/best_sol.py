# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize, basinhopping # Added basinhopping, removed differential_evolution, NonlinearConstraint
from numba import jit

# --- Numba-jitted functions (moved to module level to avoid pickling issues) ---

@jit(nopython=True, cache=True)
def fast_objective_pure(params, n):
    """Numba-jitted objective function: pure negative sum of radii."""
    s = 0.0
    for i in range(n):
        s += params[i * 3 + 2]  # Add radius of circle i
    return -s

@jit(nopython=True, cache=True)
def fast_constraints_all(params, n):
    """Numba-jitted constraint evaluation function, returns an array of g(x) >= 0."""
    num_overlap_cons = n * (n - 1) // 2
    num_boundary_cons = 4 * n
    total_constraints = num_overlap_cons + num_boundary_cons
    constraints_values = np.empty(total_constraints, dtype=np.float64)

    centers = np.empty((n, 2), dtype=np.float64)
    radii = np.empty(n, dtype=np.float64)
    for i in range(n):
        base_idx = i * 3
        centers[i, 0] = params[base_idx]
        centers[i, 1] = params[base_idx + 1]
        radii[i] = params[base_idx + 2]

    # Non-overlap constraints: dist_sq(i, j) - (ri + rj)^2 >= 0
    cons_idx = 0
    for i in range(n):
        for j in range(i + 1, n):
            dx = centers[i, 0] - centers[j, 0]
            dy = centers[i, 1] - centers[j, 1]
            dist_sq = dx * dx + dy * dy
            sum_r = radii[i] + radii[j]
            constraints_values[cons_idx] = dist_sq - sum_r * sum_r
            cons_idx += 1

    # Boundary constraints: x_i - r_i >= 0, 1 - x_i - r_i >= 0, etc.
    for i in range(n):
        r_i = radii[i]
        constraints_values[cons_idx] = centers[i, 0] - r_i
        cons_idx += 1
        constraints_values[cons_idx] = 1.0 - centers[i, 0] - r_i
        cons_idx += 1
        constraints_values[cons_idx] = centers[i, 1] - r_i
        cons_idx += 1
        constraints_values[cons_idx] = 1.0 - centers[i, 1] - r_i
        cons_idx += 1
        
    return constraints_values

# Removed fast_penalized_objective as it's not used with basinhopping's local minimizer

# --- Main circle packing function ---

def circle_packing32() -> np.ndarray:
    """
    Generates an optimal arrangement of 32 non-overlapping circles within a unit square,
    maximizing the sum of their radii, using a Basinhopping global optimization approach
    with SLSQP as the local minimizer, accelerated with Numba.

    This approach combines:
    - Basinhopping for robust global search to escape local minima.
    - SLSQP for precise local refinement in each basin.
    - Numba for high-performance objective and constraint evaluations.
    """
    n = 32
    np.random.seed(42)  # For reproducibility

    # 1. Initial Guess: A jittered grid provides a strong starting point.
    grid_dim = int(np.ceil(np.sqrt(n)))
    # A slightly larger radius than 1.0 / (2.0 * grid_dim) for the initial guess
    # helps to fill space more aggressively, as seen in some successful inspirations.
    # The actual max radius is clamped by bounds and constraints.
    initial_r_guess = 0.5 / grid_dim * 0.95 # Start slightly smaller to be feasible
    circles0 = np.zeros((n, 3))
    count = 0
    for i in range(grid_dim):
        for j in range(grid_dim):
            if count < n:
                x = (2 * j + 1) * (1.0 / (2 * grid_dim))
                y = (2 * i + 1) * (1.0 / (2 * grid_dim))
                circles0[count] = [x, y, initial_r_guess]
                count += 1
    
    # Add a small random jitter to break symmetry and aid optimization
    circles0[:, :2] += np.random.uniform(-1e-5, 1e-5, circles0[:, :2].shape)
    x0 = circles0.flatten()

    # Define bounds for each variable (0 <= x,y <= 1, min_radius_bound <= r <= 0.5)
    min_radius_bound = 1e-6 # Smallest allowed radius, consistent with Inspiration 1
    bounds = []
    for _ in range(n):
        bounds.extend([(0.0, 1.0), (0.0, 1.0), (min_radius_bound, 0.5)])

    # 2. Objective Function: Minimize negative sum of radii.
    # Wrapped in lambda to pass 'n' to the Numba-jitted function.
    objective_func = lambda p: fast_objective_pure(p, n)

    # 3. Constraints: All constraint functions must be >= 0 for 'ineq'.
    # Wrapped in lambda to pass 'n' to the Numba-jitted function.
    constraints_func = lambda p: fast_constraints_all(p, n)
    cons = {'type': 'ineq', 'fun': constraints_func}

    # 4. Local minimizer options for SLSQP within basinhopping
    slsqp_options = {
        'maxiter': 2000,  # High iterations for thorough local search (from Inspiration 1)
        'ftol': 1e-9,     # Stricter tolerance for higher precision (from Inspiration 1)
        'disp': False,
        'eps': 1e-7       # Step size for numerical approximation of Jacobian
    }

    minimizer_kwargs = {
        'method': 'SLSQP',
        'bounds': bounds,
        'constraints': cons,
        'options': slsqp_options
    }

    # 5. Run Basinhopping global optimization (parameters from Inspiration 1)
    result = basinhopping(
        objective_func,
        x0,
        niter=50,        # Number of hopping steps (from Inspiration 1)
        T=1.0,           # Standard temperature
        stepsize=0.04,   # Step size for perturbations (from Inspiration 1)
        minimizer_kwargs=minimizer_kwargs,
        disp=False,
        seed=42,         # For reproducibility
        # niter_success=5 # Optional: stop after N successful steps without improvement
    )

    final_circles = result.x.reshape((n, 3))
    
    # Basinhopping usually finds a good solution, but ensure radii are positive
    final_circles[:, 2] = np.maximum(final_circles[:, 2], min_radius_bound)

    return final_circles


# EVOLVE-BLOCK-END
