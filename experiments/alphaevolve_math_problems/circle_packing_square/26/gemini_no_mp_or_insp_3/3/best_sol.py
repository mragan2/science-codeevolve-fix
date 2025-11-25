# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize, NonlinearConstraint, basinhopping
from scipy.spatial.distance import pdist, squareform
from numba import njit

# --- Stage 1: Physics-Based Initial Guess Generation ---

@njit
def _resolve_step(positions: np.ndarray, radii: np.ndarray, learning_rate: float):
    """
    Performs one step of collision and boundary resolution using repulsive forces.
    This function is JIT-compiled with Numba for performance.
    Modifies positions in-place.
    """
    n = positions.shape[0]
    move_vectors = np.zeros_like(positions)

    # Pairwise repulsive forces for overlapping circles
    for i in range(n):
        for j in range(i + 1, n):
            vec = positions[i] - positions[j]
            dist_sq = vec[0]**2 + vec[1]**2
            r_sum = radii[i] + radii[j]
            
            if dist_sq < r_sum**2 and dist_sq > 1e-12:
                dist = np.sqrt(dist_sq)
                overlap = r_sum - dist
                direction = vec / dist
                move = direction * overlap * 0.5 * learning_rate
                move_vectors[i] += move
                move_vectors[j] -= move

    # Boundary repulsive forces
    for i in range(n):
        if positions[i, 0] < radii[i]:
            move_vectors[i, 0] += (radii[i] - positions[i, 0]) * learning_rate
        if positions[i, 0] > 1 - radii[i]:
            move_vectors[i, 0] -= (positions[i, 0] - (1 - radii[i])) * learning_rate
        if positions[i, 1] < radii[i]:
            move_vectors[i, 1] += (radii[i] - positions[i, 1]) * learning_rate
        if positions[i, 1] > 1 - radii[i]:
            move_vectors[i, 1] -= (positions[i, 1] - (1 - radii[i])) * learning_rate
            
    positions += move_vectors
    # Clip to unit square as a safeguard
    np.clip(positions, 0.0, 1.0, out=positions)

def _generate_initial_guess(n: int, seed: int) -> np.ndarray:
    """
    Generates a high-quality initial guess using a physics-based simulation.
    Circles are iteratively inflated and their positions relaxed to avoid overlap.
    """
    rng = np.random.default_rng(seed)
    positions = rng.random((n, 2))
    radii = np.full(n, 0.01)
    
    # Simulation hyperparameters
    inflation_rate = 1.002
    relaxation_steps = 20
    learning_rate = 0.2
    
    for _ in range(1600):  # Main simulation loop
        radii *= inflation_rate
        for _ in range(relaxation_steps):
            _resolve_step(positions, radii, learning_rate)
            
    return np.hstack((positions, radii.reshape(-1, 1)))

# --- Stage 2: SLSQP-Based Refinement ---

def _objective_func(params: np.ndarray) -> float:
    """Objective function: minimize the negative sum of radii."""
    return -np.sum(params[2::3])

def _constraint_func(params: np.ndarray) -> np.ndarray:
    """
    Computes all constraint violations. For a valid solution, all values should be >= 0.
    """
    n = 26
    circles = params.reshape((n, 3))
    pos, radii = circles[:, :2], circles[:, 2]

    # Boundary constraints: ri <= xi <= 1-ri and ri <= yi <= 1-ri
    c_bounds_x1 = pos[:, 0] - radii
    c_bounds_x2 = (1 - radii) - pos[:, 0]
    c_bounds_y1 = pos[:, 1] - radii
    c_bounds_y2 = (1 - radii) - pos[:, 1]

    # Non-overlap constraints: sqrt((xi-xj)^2 + (yi-yj)^2) >= ri + rj
    # Using squared distances for efficiency: (xi-xj)^2 + (yi-yj)^2 - (ri+rj)^2 >= 0
    dist_sq_matrix = squareform(pdist(pos, 'sqeuclidean'))
    r_sum = radii[:, None] + radii
    iu = np.triu_indices(n, k=1)
    c_overlap = dist_sq_matrix[iu] - r_sum[iu]**2

    return np.concatenate([c_bounds_x1, c_bounds_x2, c_bounds_y1, c_bounds_y2, c_overlap])

def circle_packing26() -> np.ndarray:
    """
    Places 26 non-overlapping circles in the unit square in order to maximize the sum of radii.
    This function uses a hybrid two-stage optimization approach:
    1. A physics-based simulation to generate a high-quality initial guess.
    2. A Sequential Least Squares Programming (SLSQP) algorithm to refine the solution
       to high precision, satisfying all constraints.
    
    Returns:
        circles: np.array of shape (26,3), where the i-th row (x,y,r) stores the
                 (x,y) coordinates of the i-th circle of radius r.
    """
    n = 26
    seed = 42  # For deterministic results

    # Stage 1: Generate a good starting point
    initial_circles = _generate_initial_guess(n, seed)
    x0 = initial_circles.flatten()

    # Stage 2: Refine with SLSQP
    bounds = [(0, 1), (0, 1), (0, 0.5)] * n
    
    num_constraints = (4 * n) + (n * (n - 1) // 2)
    nonlinear_constraint = NonlinearConstraint(
        _constraint_func, 
        lb=np.zeros(num_constraints), 
        ub=np.inf * np.ones(num_constraints)
    )

    minimizer_kwargs = {
        "method": 'SLSQP',
        "bounds": bounds,
        "constraints": [nonlinear_constraint],
        "options": {'maxiter': 500, 'ftol': 1e-9, 'disp': False} # Reduced maxiter for SLSQP in basinhopping
    }

    # Stage 2: Refine with basinhopping for global optimization
    # This combines local optimization with stochastic jumps to escape local minima.
    bh_res = basinhopping(
        _objective_func,
        x0,
        minimizer_kwargs=minimizer_kwargs,
        niter=200,      # Number of basin-hopping iterations
        T=1.0,          # Temperature parameter for the Metropolis criterion
        stepsize=0.05,  # Max step size for random perturbations
        seed=seed,      # For deterministic results of basinhopping's random steps
        disp=False      # Suppress basinhopping output
    )
    
    # Return the best found solution from basinhopping, reshaped to (n, 3)
    return bh_res.x.reshape((n, 3))


# EVOLVE-BLOCK-END
