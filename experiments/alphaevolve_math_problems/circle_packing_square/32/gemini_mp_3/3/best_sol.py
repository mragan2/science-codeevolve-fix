# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize, Bounds, NonlinearConstraint, basinhopping
from scipy.spatial.distance import pdist, squareform


def circle_packing32() -> np.ndarray:
    """
    Places 32 non-overlapping circles in the unit square to maximize the sum of radii.
    This implementation uses scipy's `basinhopping` global optimizer with tuned
    parameters for more extensive exploration and refined local optimization.

    Returns:
        circles: np.array of shape (32,3), where the i-th row (x,y,r) stores the
                 (x,y) coordinates of the i-th circle of radius r.
    """
    n_circles = 32

    # Use a dedicated RandomState for reproducible initial guess generation
    rng = np.random.RandomState(42)

    # 1. INITIALIZATION: Create a good starting point (perturbed grid).
    grid_size = int(np.ceil(np.sqrt(n_circles)))
    x_coords, y_coords = np.meshgrid(
        np.linspace(0.1, 0.9, grid_size),
        np.linspace(0.1, 0.9, grid_size)
    )
    initial_centers = np.vstack([x_coords.ravel(), y_coords.ravel()]).T[:n_circles]
    initial_centers += rng.uniform(-0.02, 0.02, initial_centers.shape)
    initial_radii = np.full(n_circles, 0.05)
    initial_guess = np.hstack([initial_centers, initial_radii.reshape(-1, 1)]).ravel()

    # 2. OBJECTIVE FUNCTION: Maximize sum of radii -> Minimize negative sum of radii.
    def objective_func(variables):
        radii = variables[2::3]
        return -np.sum(radii)

    # 3. BOUNDS: Define simple bounds for each variable.
    lower_bounds = np.zeros_like(initial_guess)
    upper_bounds = np.ones_like(initial_guess)
    upper_bounds[2::3] = 0.5  # Max radius for any circle is 0.5
    bounds = Bounds(lower_bounds, upper_bounds)

    # 4. CONSTRAINTS: Define non-linear constraints (containment and non-overlap).
    def constraint_func(variables):
        circles = variables.reshape((n_circles, 3))
        x, y, r = circles[:, 0], circles[:, 1], circles[:, 2]

        # Containment: x_i-r_i >= 0, 1-x_i-r_i >= 0, etc.
        containment_cons = np.concatenate([x - r, 1 - x - r, y - r, 1 - y - r])

        # Non-overlap: (xi-xj)^2 + (yi-yj)^2 - (ri+rj)^2 >= 0
        centers = circles[:, :2]
        sq_dists = squareform(pdist(centers, 'sqeuclidean'))
        radii_sum = r[:, np.newaxis] + r
        radii_sum_sq = radii_sum**2
        iu = np.triu_indices(n_circles, k=1)
        overlap_cons = (sq_dists - radii_sum_sq)[iu]

        return np.concatenate([containment_cons, overlap_cons])

    num_constraints = n_circles * 4 + n_circles * (n_circles - 1) // 2
    constraint_bounds = np.zeros(num_constraints)
    nonlinear_constraint = NonlinearConstraint(constraint_func, constraint_bounds, np.inf)

    # 5. LOCAL MINIMIZER SETUP for basinhopping.
    # Each hop will use SLSQP to find the local minimum.
    minimizer_kwargs = {
        'method': 'SLSQP',
        'bounds': bounds,
        'constraints': [nonlinear_constraint],
        'options': {'maxiter': 1500, 'ftol': 1e-7, 'disp': False} # Increased maxiter
    }

    # 6. CUSTOM STEP-TAKER for basinhopping.
    # This class defines how to "hop" from one minimum to another.
    class RandomDisplacement:
        def __init__(self, stepsize=0.1, random_seed=42):
            self.stepsize = stepsize
            self.rng = np.random.RandomState(random_seed)

        def __call__(self, x):
            # Perturb x and y coordinates
            x[0::3] += self.rng.uniform(-self.stepsize, self.stepsize, n_circles)
            x[1::3] += self.rng.uniform(-self.stepsize, self.stepsize, n_circles)
            # Perturb radii more dynamically
            x[2::3] += self.rng.uniform(-self.stepsize * 0.2, self.stepsize * 0.2, n_circles) # Increased radii perturbation factor
            
            # Enforce simple bounds after perturbation
            x[0::3] = np.clip(x[0::3], 0, 1)
            x[1::3] = np.clip(x[1::3], 0, 1)
            x[2::3] = np.clip(x[2::3], 0, 0.5)
            return x

    # 7. GLOBAL OPTIMIZATION: Run the basinhopping algorithm.
    take_step = RandomDisplacement(stepsize=0.04, random_seed=1337) # Reduced stepsize
    result = basinhopping(
        objective_func,
        initial_guess,
        minimizer_kwargs=minimizer_kwargs,
        niter=100,  # Increased number of hopping iterations
        take_step=take_step,
        seed=42,
        disp=False
    )

    # 8. RETURN RESULT: Reshape the optimized variables back to (N, 3) format.
    final_circles = result.x.reshape((n_circles, 3))
    return final_circles


# EVOLVE-BLOCK-END
