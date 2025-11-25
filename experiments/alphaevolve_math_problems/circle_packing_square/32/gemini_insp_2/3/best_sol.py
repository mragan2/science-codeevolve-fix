# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize, differential_evolution, basinhopping
import numba # Added numba for performance


# Constants
N_CIRCLES = 32
EPSILON = 1e-7 # Smallest allowed radius, adjusted to avoid numerical issues


# Objective function: -sum(radii)
def _objective(params):
    """Calculates the negative sum of radii to be minimized."""
    radii = params[2::3]
    return -np.sum(radii)

# Jacobian of the objective function
def _objective_jacobian(params):
    """Calculates the Jacobian of the objective function."""
    grad = np.zeros_like(params)
    grad[2::3] = -1.0 # Gradient with respect to radii is -1
    return grad


def _generate_hexagonal_initial_guess(n_circles, seed):
    """
    Creates a high-quality initial guess based on a hexagonal grid pattern,
    which is known to be efficient for packing. This is inspired by Inspiration Program 1.
    """
    np.random.seed(seed)
    
    # Estimate spacing 's' for a hexagonal grid. A factor > 1 gives some initial space.
    s = np.sqrt(2.0 / (n_circles * np.sqrt(3))) * 1.05
    y_step = s * np.sqrt(3) / 2
    
    points = []
    # Generate grid points over an expanded area to ensure we can select the most central ones.
    x_range_gen = int(1.4 / s) + 2
    y_range_gen = int(1.4 / y_step) + 2

    for j in range(y_range_gen):
        y = j * y_step - 0.2
        for i in range(x_range_gen):
            x = i * s - 0.2
            if j % 2 == 1: x += s / 2
            
            if 0.0 <= x <= 1.0 and 0.0 <= y <= 1.0:
                points.append([x, y])

    if len(points) < n_circles:
        # Fallback to random positions if hex grid is insufficient (unlikely)
        initial_pos = np.random.rand(n_circles, 2)
    else:
        # Select the n_circles points closest to the center of the square (0.5, 0.5)
        points = np.array(points)
        center = np.array([0.5, 0.5])
        distances_to_center = np.linalg.norm(points - center, axis=1)
        sorted_indices = np.argsort(distances_to_center)
        initial_pos = points[sorted_indices[:n_circles]]
    
    # Add small random noise to break perfect symmetry
    initial_pos += np.random.normal(scale=s*0.02, size=initial_pos.shape)
    np.clip(initial_pos, 0.01, 0.99, out=initial_pos)

    # Start with radii that are slightly non-overlapping
    initial_radii = np.full(n_circles, s * 0.49)
    np.clip(initial_radii, EPSILON, 0.5, out=initial_radii)
    
    return np.hstack([initial_pos, initial_radii.reshape(-1, 1)]).ravel()


# Numba-optimized functions for constraints and their Jacobians
@numba.njit(cache=True)
def _boundary_constraints_func_numba(params):
    """Numba-optimized function for boundary constraints.
    Returns: A 1D array of values for each of the 4*N_CIRCLES boundary constraints.
    Each value must be >= 0 for the constraint to be satisfied.
    """
    n = len(params) // 3
    constraints = np.empty(n * 4, dtype=params.dtype)
    for i in range(n):
        x, y, r = params[i*3], params[i*3+1], params[i*3+2]
        constraints[i*4] = x - r         # r <= x  => x - r >= 0
        constraints[i*4+1] = 1 - x - r   # x <= 1-r => 1 - x - r >= 0
        constraints[i*4+2] = y - r         # r <= y  => y - r >= 0
        constraints[i*4+3] = 1 - y - r   # y <= 1-r => 1 - y - r >= 0
    return constraints

@numba.njit(cache=True)
def _boundary_constraints_jacobian_numba(params):
    """Numba-optimized function for boundary constraints Jacobian.
    Returns: A 2D array where rows correspond to constraints and columns to parameters.
    """
    n = len(params) // 3
    jac = np.zeros((n * 4, n * 3), dtype=params.dtype)
    for i in range(n):
        # Constraint x_i - r_i >= 0
        jac[i*4, i*3] = 1.0     # d(x-r)/dx_i
        jac[i*4, i*3+2] = -1.0  # d(x-r)/dr_i
        # Constraint 1 - x_i - r_i >= 0
        jac[i*4+1, i*3] = -1.0  # d(1-x-r)/dx_i
        jac[i*4+1, i*3+2] = -1.0 # d(1-x-r)/dr_i
        # Constraint y_i - r_i >= 0
        jac[i*4+2, i*3+1] = 1.0   # d(y-r)/dy_i
        jac[i*4+2, i*3+2] = -1.0  # d(y-r)/dr_i
        # Constraint 1 - y_i - r_i >= 0
        jac[i*4+3, i*3+1] = -1.0 # d(1-y-r)/dy_i
        jac[i*4+3, i*3+2] = -1.0 # d(1-y-r)/dr_i
    return jac

@numba.njit(cache=True)
def _overlap_constraints_func_numba(params):
    """Numba-optimized function for overlap constraints.
    Returns: A 1D array of values for each of the N*(N-1)/2 overlap constraints.
    Each value must be >= 0 for the constraint to be satisfied.
    """
    n = len(params) // 3
    num_pairs = n * (n - 1) // 2
    constraints = np.empty(num_pairs, dtype=params.dtype)
    k = 0
    for i in range(n):
        for j in range(i + 1, n): # Iterate over unique pairs
            xi, yi, ri = params[i*3], params[i*3+1], params[i*3+2]
            xj, yj, rj = params[j*3], params[j*3+1], params[j*3+2]
            dist_sq = (xi - xj)**2 + (yi - yj)**2
            radii_sum_sq = (ri + rj)**2
            constraints[k] = dist_sq - radii_sum_sq # (xi-xj)^2 + (yi-yj)^2 - (ri+rj)^2 >= 0
            k += 1
    return constraints

@numba.njit(cache=True)
def _overlap_constraints_jacobian_numba(params):
    """Numba-optimized function for overlap constraints Jacobian.
    Returns: A 2D array where rows correspond to constraints and columns to parameters.
    """
    n = len(params) // 3
    num_pairs = n * (n - 1) // 2
    jac = np.zeros((num_pairs, n * 3), dtype=params.dtype)
    k = 0
    for i in range(n):
        for j in range(i + 1, n):
            xi, yi, ri = params[i*3], params[i*3+1], params[i*3+2]
            xj, yj, rj = params[j*3], params[j*3+1], params[j*3+2]

            # Derivatives for circle i
            jac[k, i*3] = 2 * (xi - xj)       # d/dxi [ (xi-xj)^2 + ... ]
            jac[k, i*3+1] = 2 * (yi - yj)     # d/dyi [ (yi-yj)^2 + ... ]
            jac[k, i*3+2] = -2 * (ri + rj)    # d/dri [ ... - (ri+rj)^2 ]

            # Derivatives for circle j
            jac[k, j*3] = 2 * (xj - xi)       # d/dxj [ (xi-xj)^2 + ... ]
            jac[k, j*3+1] = 2 * (yj - yi)     # d/dyj [ (yi-yj)^2 + ... ]
            jac[k, j*3+2] = -2 * (ri + rj)    # d/drj [ ... - (ri+rj)^2 ]
            k += 1
    return jac

def circle_packing32() -> np.ndarray:
    """
    Places 32 non-overlapping circles in the unit square to maximize the sum of radii.
    Uses a hybrid global optimization approach (Basin Hopping) with a gradient-aware
    local optimizer (SLSQP) and Numba-JIT compiled functions for high performance.
    This version incorporates a superior hexagonal initial guess and tuned hyperparameters
    to push towards the global optimum.

    Returns:
        circles: np.array of shape (32,3), where the i-th row (x,y,r) stores the (x,y)
                 coordinates of the i-th circle of radius r.
    """
    # 1. Generate a high-quality hexagonal initial guess (inspired by Inspiration Program 1)
    # This provides a theoretically denser starting packing than a simple grid.
    initial_guess_flat = _generate_hexagonal_initial_guess(N_CIRCLES, seed=42)

    # 2. Define bounds and constraints (using existing high-performance functions)
    bounds = [(0, 1), (0, 1), (EPSILON, 0.5)] * N_CIRCLES
    constraints = [
        {'type': 'ineq', 'fun': _boundary_constraints_func_numba, 'jac': _boundary_constraints_jacobian_numba},
        {'type': 'ineq', 'fun': _overlap_constraints_func_numba, 'jac': _overlap_constraints_jacobian_numba}
    ]

    # 3. Configure the local optimizer (SLSQP) for Basin Hopping with tuned parameters
    # Increased maxiter for more thorough local searches within each basin.
    slsqp_options = {'maxiter': 300, 'ftol': 1e-8, 'disp': False}
    minimizer_kwargs = {
        'method': 'SLSQP',
        'jac': _objective_jacobian,
        'bounds': bounds,
        'constraints': constraints,
        'options': slsqp_options
    }

    # 4. Run the global optimization using Basin Hopping with tuned hyperparameters
    # Increased niter for more exploration and adjusted T for better acceptance.
    print("Starting global optimization with Basin Hopping...")
    bh_result = basinhopping(
        func=_objective,
        x0=initial_guess_flat,
        minimizer_kwargs=minimizer_kwargs,
        niter=150,           # Increased iterations for more global search
        T=0.75,              # Adjusted temperature for metropolis criterion
        stepsize=0.05,       # Perturbation step size
        seed=42,             # Use a fixed seed for the random number generator
        disp=False
    )
    print(f"Basin Hopping finished. Best objective: {-bh_result.fun:.6f}")

    # 5. Final high-precision refinement using SLSQP
    # Start from the best point found by basinhopping and polish it with tight tolerances.
    print("Starting final high-precision refinement with SLSQP...")
    final_result = minimize(
        fun=_objective,
        x0=bh_result.x,
        method='SLSQP',
        jac=_objective_jacobian,
        constraints=constraints,
        bounds=bounds,
        options={'maxiter': 6000, 'ftol': 1e-11, 'disp': False} # Increased precision
    )
    
    if final_result.success:
        optimized_params = final_result.x
        print(f"Final refinement successful! Final sum of radii: {-final_result.fun:.15f}")
    else:
        print(f"Final refinement failed ({final_result.message}). Using Basin Hopping's best result.")
        optimized_params = bh_result.x

    # 6. Post-processing and formatting the output
    circles = optimized_params.reshape((N_CIRCLES, 3))

    # Ensure radii are positive and clip coordinates to be strictly within [r, 1-r]
    # This robust post-processing step guarantees a valid output.
    final_radii = np.maximum(circles[:, 2], EPSILON)
    final_x = np.clip(circles[:, 0], final_radii, 1.0 - final_radii)
    final_y = np.clip(circles[:, 1], final_radii, 1.0 - final_radii)
    
    circles = np.vstack((final_x, final_y, final_radii)).T

    return circles


# EVOLVE-BLOCK-END
