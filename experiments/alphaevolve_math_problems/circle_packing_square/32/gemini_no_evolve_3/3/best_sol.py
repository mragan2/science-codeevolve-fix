# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
from scipy.spatial.distance import pdist
from itertools import combinations
import numba


def circle_packing32() -> np.ndarray:
    """
    Generates an optimized arrangement of 32 non-overlapping circles within a unit square.

    The method uses a physics-based relaxation to generate a high-quality initial guess,
    which is then refined using Sequential Least Squares Programming (SLSQP) to maximize
    the sum of radii.

    Returns:
        np.ndarray: An array of shape (32, 3) representing the circles (x, y, r).
    """
    N = 32
    seed = 42

    @numba.njit
    def _relax_positions(pos, radii, n_relax_steps, dt, random_kick_seed):
        """Numba-jitted inner loop for physics relaxation."""
        N = pos.shape[0]
        # Numba requires seeding within the jitted function for reproducibility
        np.random.seed(random_kick_seed)

        for _ in range(n_relax_steps):
            forces = np.zeros_like(pos)

            # Repulsive forces between circles
            for i in range(N):
                for j in range(i + 1, N):
                    vec = pos[j] - pos[i]
                    dist_sq = vec[0]**2 + vec[1]**2
                    
                    if dist_sq < 1e-12: # Avoid division by zero
                        vec = (np.random.rand(2) - 0.5) * 1e-6
                        dist_sq = vec[0]**2 + vec[1]**2

                    dist = np.sqrt(dist_sq)
                    sum_r = radii[i] + radii[j]

                    if dist < sum_r:
                        overlap = sum_r - dist
                        direction = vec / dist
                        force_magnitude = overlap
                        forces[i] -= force_magnitude * direction
                        forces[j] += force_magnitude * direction

            # Boundary forces
            for i in range(N):
                if pos[i, 0] < radii[i]: forces[i, 0] += (radii[i] - pos[i, 0])
                if pos[i, 0] > 1 - radii[i]: forces[i, 0] -= (pos[i, 0] - (1 - radii[i]))
                if pos[i, 1] < radii[i]: forces[i, 1] += (radii[i] - pos[i, 1])
                if pos[i, 1] > 1 - radii[i]: forces[i, 1] -= (pos[i, 1] - (1 - radii[i]))

            pos += forces * dt
            
            for i in range(N):
                pos[i, 0] = min(max(pos[i, 0], 1e-6), 1 - 1e-6)
                pos[i, 1] = min(max(pos[i, 1], 1e-6), 1 - 1e-6)
                
        return pos

    def _generate_physics_initial_guess(N, seed):
        """Generates a high-quality initial guess using physics-based relaxation."""
        rng = np.random.default_rng(seed)
        pos = rng.random((N, 2))
        radii = np.zeros(N)

        # Simulation parameters tuned for this problem
        n_steps = 800
        n_relax_steps = 5
        growth_rate = 0.00015
        dt = 0.2

        for i in range(n_steps):
            radii += growth_rate
            pos = _relax_positions(pos, radii, n_relax_steps, dt, random_kick_seed=seed + i)
        
        max_r_x = np.minimum(pos[:, 0], 1 - pos[:, 0])
        max_r_y = np.minimum(pos[:, 1], 1 - pos[:, 1])
        radii = np.minimum(radii, np.minimum(max_r_x, max_r_y))
        
        return np.hstack([pos, radii[:, np.newaxis]]).flatten()

    # Pre-calculate indices for non-overlap constraints for efficiency.
    pair_indices = np.array(list(combinations(range(N), 2)))

    # 2. Objective Function (to be minimized)
    def objective(variables: np.ndarray) -> float:
        """Minimize the negative sum of radii."""
        radii = variables[2::3]
        return -np.sum(radii)

    # 3. Constraint Function (all constraints must be >= 0)
    def constraints(variables: np.ndarray) -> np.ndarray:
        """Return a vector of all constraint violations."""
        circles = variables.reshape((N, 3))
        positions = circles[:, :2]
        radii = circles[:, 2]

        # a) Containment constraints: ri <= xi <= 1-ri and ri <= yi <= 1-ri
        containment_cons = np.concatenate([
            positions[:, 0] - radii,       # xi - ri >= 0
            1 - positions[:, 0] - radii,   # 1 - xi - ri >= 0
            positions[:, 1] - radii,       # yi - ri >= 0
            1 - positions[:, 1] - radii,   # 1 - yi - ri >= 0
        ])

        # b) Non-overlap constraints: (xi-xj)^2 + (yi-yj)^2 >= (ri+rj)^2
        if N > 1:
            # Use pdist for efficient pairwise squared Euclidean distance calculation
            sq_dists = pdist(positions, 'sqeuclidean')
            sum_radii = radii[pair_indices[:, 0]] + radii[pair_indices[:, 1]]
            # Constraint: sq_dist - (ri+rj)^2 >= 0
            overlap_cons = sq_dists - sum_radii**2
            return np.concatenate([containment_cons, overlap_cons])
        else:
            return containment_cons

    # 4. Bounds for each variable (x_i, y_i, r_i)
    bounds = []
    for _ in range(N):
        # x, y coordinates must be within [1e-6, 1 - 1e-6] to ensure numerical stability and small buffer
        # Radius must be positive (1e-6) and at most 0.5 (for a single circle in unit square)
        bounds.extend([(1e-6, 1 - 1e-6), (1e-6, 1 - 1e-6), (1e-6, 0.5)])

    # 5. Run the optimization using SLSQP with multiple initial guesses
    cons = {'type': 'ineq', 'fun': constraints}
    
    best_sum_radii = -np.inf
    best_optimized_circles = None
    
    num_initial_guesses = 5 # Number of different initial guesses to try
    
    print(f"Starting {num_initial_guesses} optimization attempts with varied initial guesses...")

    for i in range(num_initial_guesses):
        # Generate initial guess using a physics-based simulation with a varied seed.
        # Varying the seed ensures different initial placements and relaxation paths,
        # increasing the chance of finding a better local optimum.
        current_seed = seed + i * 100 # Use a larger step for seeds to ensure more distinct initializations
        print(f"  Attempt {i+1}/{num_initial_guesses} with initial guess seed: {current_seed}...")
        x0 = _generate_physics_initial_guess(N, current_seed)

        result = minimize(
            objective,
            x0,
            method='SLSQP',
            bounds=bounds,
            constraints=cons,
            # Increased iterations and tighter tolerance for better refinement
            options={'maxiter': 3000, 'ftol': 1e-11, 'disp': False}
        )
        
        if result.success:
            current_circles = result.x.reshape((N, 3))
            current_sum_radii = np.sum(current_circles[:, 2])
            print(f"    Optimization successful. Sum of radii: {current_sum_radii:.8f}")
            # Keep track of the best result found so far
            if current_sum_radii > best_sum_radii:
                best_sum_radii = current_sum_radii
                best_optimized_circles = current_circles
        else:
            print(f"    Warning: Optimization for seed {current_seed} did not converge. Reason: {result.message}")
            # Failed optimizations are not considered for the best result.

    if best_optimized_circles is None:
        # Fallback if all optimization attempts failed (unlikely, but robust).
        print("Error: All optimization attempts failed. Falling back to the initial guess from the default seed.")
        best_optimized_circles = _generate_physics_initial_guess(N, seed).reshape((N, 3))
    else:
        print(f"\nAll optimization attempts complete. Best sum of radii found: {best_sum_radii:.8f}")
        
    return best_optimized_circles


# EVOLVE-BLOCK-END
