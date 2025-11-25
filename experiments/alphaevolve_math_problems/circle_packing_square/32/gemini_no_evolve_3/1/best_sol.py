# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize, basinhopping
from numba import njit


def circle_packing32() -> np.ndarray:
    """
    Places 32 non-overlapping circles in the unit square to maximize the sum of radii.
    This uses a two-phase optimization strategy:
    1. Basin-Hopping (Global Search): A stochastic global optimization algorithm is used to
       explore the solution space and escape local minima. It performs iterated local searches
       from perturbed starting points. A custom step function perturbs a small subset of
       circles at a time, creating more effective exploration.
    2. SLSQP (Local Refinement): The best solution found by basin-hopping is then refined
       using a high-precision SLSQP optimization to ensure it is a high-quality local optimum.

    This hybrid approach is more robust than a single local search and is more likely to find
    a globally competitive solution.

    - Initial Guess: A slightly perturbed 6x6 grid, which serves as the starting point for
      the basin-hopping algorithm.
    - Objective Function: The negative sum of radii (as optimizers minimize).
    - Constraints: Containment within the unit square and non-overlapping between circles.
    - Performance: The O(n^2) non-overlap constraint calculation is JIT-compiled with `numba`.
    """
    n = 32
    # Use a RandomState object for better control over seeding in complex algorithms
    rng = np.random.RandomState(42)

    @njit
    def non_overlap_constraints(p_flat):
        circles = p_flat.reshape((n, 3))
        num_pairs = n * (n - 1) // 2
        cons = np.empty(num_pairs, dtype=np.float64)
        k = 0
        for i in range(n):
            for j in range(i + 1, n):
                xi, yi, ri = circles[i]
                xj, yj, rj = circles[j]
                d_sq = (xi - xj)**2 + (yi - yj)**2
                r_sum_sq = (ri + rj)**2
                cons[k] = d_sq - r_sum_sq
                k += 1
        return cons

    def objective(p):
        return -np.sum(p[2::3])

    def all_constraints(p):
        circles = p.reshape((n, 3))
        pos = circles[:, :2]
        radii = circles[:, 2]
        cons_boundary_x1 = pos[:, 0] - radii
        cons_boundary_x2 = 1 - pos[:, 0] - radii
        cons_boundary_y1 = pos[:, 1] - radii
        cons_boundary_y2 = 1 - pos[:, 1] - radii
        cons_overlap = non_overlap_constraints(p)
        return np.concatenate([
            cons_boundary_x1, cons_boundary_x2,
            cons_boundary_y1, cons_boundary_y2,
            cons_overlap
        ])

    # Initial Guess: Use a 4x8 grid to perfectly match 32 circles
    n_rows = 4
    n_cols = 8
    initial_circles = np.zeros((n, 3))
    # Calculate radius for a square grid, ensuring it fits within the unit square
    radius = 1 / (2 * max(n_rows, n_cols)) # This is 1 / (2 * 8) = 0.0625

    count = 0
    for i in range(n_rows):
        for j in range(n_cols):
            initial_circles[count] = [(j + 0.5) / n_cols, (i + 0.5) / n_rows, radius]
            count += 1
    
    initial_guess = initial_circles.flatten()
    # Add perturbation to positions and radii to break initial symmetry
    initial_guess[0::3] += rng.normal(0, 1e-4, size=n) # x-coordinates
    initial_guess[1::3] += rng.normal(0, 1e-4, size=n) # y-coordinates
    initial_guess[2::3] += rng.normal(0, 2e-3, size=n) # radii - slightly larger perturbation for radii

    bounds = []
    for _ in range(n):
        bounds.append((0, 1)) # x-coordinate
        bounds.append((0, 1)) # y-coordinate
        bounds.append((0.01, 0.5)) # radius (min 0.01 to avoid degenerate circles, max 0.5)

    bounds_arr = np.array(bounds)
    xmin, xmax = bounds_arr[:, 0], bounds_arr[:, 1]
    
    # Clip initial guess to ensure it respects the bounds
    np.clip(initial_guess, xmin, xmax, out=initial_guess)

    # Custom step function for basin-hopping to make intelligent perturbations
    class CustomStep:
        def __init__(self, stepsize=0.1, num_circles_to_move=None):
            self.stepsize = stepsize
            # Perturb a larger subset of circles for better exploration
            self.num_circles_to_move = num_circles_to_move if num_circles_to_move is not None else n // 3

        def __call__(self, x):
            x_new = np.copy(x)
            circles_to_move_idx = rng.choice(n, self.num_circles_to_move, replace=False)
            
            for i in circles_to_move_idx:
                start_idx = i * 3
                # Perturb x, y, r with scaled random steps
                x_step = rng.uniform(-self.stepsize, self.stepsize) * (xmax[start_idx] - xmin[start_idx])
                y_step = rng.uniform(-self.stepsize, self.stepsize) * (xmax[start_idx+1] - xmin[start_idx+1])
                r_step = rng.uniform(-self.stepsize, self.stepsize) * (xmax[start_idx+2] - xmin[start_idx+2])
                
                x_new[start_idx:start_idx+3] += [x_step, y_step, r_step]

            np.clip(x_new, xmin, xmax, out=x_new)
            return x_new

    # Phase 1: Global search with Basin-Hopping
    cons = {'type': 'ineq', 'fun': all_constraints}
    minimizer_kwargs = {
        "method": "SLSQP",
        "bounds": bounds,
        "constraints": cons,
        "options": {'maxiter': 1000, 'ftol': 1e-7} # Increased maxiter for local minimizer within basin-hopping
    }
    
    # Increased number of circles to move for broader exploration
    take_step = CustomStep(stepsize=0.1, num_circles_to_move=n // 3) 
    
    bh_res = basinhopping(
        objective,
        initial_guess,
        niter=100,  # Significantly increased iterations for more thorough global search
        minimizer_kwargs=minimizer_kwargs,
        take_step=take_step,
        seed=42,
        disp=False
    )

    # Phase 2: Final high-precision refinement
    best_guess = bh_res.x
    final_res = minimize(objective, best_guess, method='SLSQP',
                         bounds=bounds, constraints=cons,
                         options={'maxiter': 5000, 'disp': False, 'ftol': 1e-10}) # Increased maxiter

    if not final_res.success:
        print(f"Final refinement failed: {final_res.message}. Using basin-hopping result.")
        return bh_res.x.reshape((n, 3))
    
    return final_res.x.reshape((n, 3))


# EVOLVE-BLOCK-END
