# EVOLVE-BLOCK-START
import numpy as np
import random
from deap import base, creator, tools, algorithms
import numba
from scipy.optimize import minimize


def circle_packing26() -> np.ndarray:
    """
    Places 26 non-overlapping circles in the unit square in order to maximize the sum of radii
    using a hybrid Genetic Algorithm and local optimization approach.

    Returns:
        circles: np.array of shape (26,3), where the i-th row (x,y,r) stores the (x,y) coordinates of the i-th circle of radius r.
    """
    N_CIRCLES = 26
    PENALTY_WEIGHT = 10000.0  # High penalty weight to strongly enforce constraints

    # EA parameters are tuned for a deep search on this specific problem
    POP_SIZE = 400 # Re-increased population size for better diversity and exploration with more generations
    N_GEN = 2500 # Significantly increased generations for even more extensive exploration
    CXPB = 0.9  # Crossover probability
    MUTPB = 0.3  # Mutation probability

    # Set seed for reproducibility
    RANDOM_SEED = 42
    random.seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)

    # Numba-jitted helper for fast violation calculation (critical for performance)
    @numba.jit(nopython=True, fastmath=True)
    def calculate_violations(circles: np.ndarray):
        n = circles.shape[0]
        overlap_penalty = 0.0
        containment_penalty = 0.0

        # 1. Containment and positive radius check
        for i in range(n):
            x, y, r = circles[i]
            if r < 0:
                containment_penalty += -r * 10.0  # Heavy penalty for negative radius
            
            containment_penalty += max(0.0, r - x)
            containment_penalty += max(0.0, x - (1.0 - r))
            containment_penalty += max(0.0, r - y)
            containment_penalty += max(0.0, y - (1.0 - r))

        # 2. Overlap check (using squared distances to avoid expensive sqrt)
        for i in range(n):
            for j in range(i + 1, n):
                xi, yi, ri = circles[i]
                xj, yj, rj = circles[j]
                
                dist_sq = (xi - xj)**2 + (yi - yj)**2
                radii_sum = ri + rj
                
                # Using (ri+rj)^2 as target_dist_sq.
                # If ri or rj are negative, the r < 0 penalty should handle it.
                # Assuming ri, rj are non-negative, then radii_sum is non-negative.
                target_dist_sq = (ri + rj)**2
                overlap = target_dist_sq - dist_sq
                if overlap > 0:
                    # Direct penalty for squared overlap distance.
                    # This avoids division by small numbers (radii_sum_sq)
                    # which can lead to unstable fitness values, making the
                    # optimization landscape smoother for the EA.
                    overlap_penalty += overlap

        return overlap_penalty, containment_penalty

    # Fitness function for DEAP, combining objective and penalties
    def evaluate(individual: list):
        circles = np.array(individual).reshape((N_CIRCLES, 3))
        sum_radii = np.sum(circles[:, 2])
        
        overlap_p, contain_p = calculate_violations(circles)
        total_penalty = PENALTY_WEIGHT * (overlap_p + contain_p)
        
        return (sum_radii - total_penalty,)

    # --- 1. Global Search: Genetic Algorithm (DEAP) ---
    # Use try-except to prevent re-registration error on multiple runs
    try:
        creator.create("FitnessMax", base.Fitness, weights=(1.0,))
        creator.create("Individual", list, fitness=creator.FitnessMax)
    except Exception:
        pass # Types are already created

    toolbox = base.Toolbox()

    # Enhanced initialization with multiple strategies for better diversity
    def create_individual():
        strategy = random.choice(['grid', 'random', 'hexagonal'])
        genes = []
        
        if strategy == 'grid':
            # Grid-based initialization (original approach)
            grid_size = 5  # 5x5 grid for 25 circles
            max_initial_r_val = 1.0 / (2 * grid_size)
            indices = list(range(grid_size * grid_size))
            random.shuffle(indices)
            
            for k in range(N_CIRCLES):
                r = random.uniform(0.01, max_initial_r_val)
                x_min, x_max = r, 1.0 - r
                y_min, y_max = r, 1.0 - r

                if k < 25:
                    i, j = divmod(indices[k], grid_size)
                    base_coords = np.linspace(max_initial_r_val, 1.0 - max_initial_r_val, grid_size)
                    base_x = base_coords[i]
                    base_y = base_coords[j]
                    jitter_amount = 0.05
                    x = np.clip(base_x + random.uniform(-jitter_amount, jitter_amount), x_min, x_max)
                    y = np.clip(base_y + random.uniform(-jitter_amount, jitter_amount), y_min, y_max)
                else:
                    x = random.uniform(x_min, x_max)
                    y = random.uniform(y_min, y_max)
                
                genes.extend([x, y, r])
                
        elif strategy == 'hexagonal':
            # Hexagonal-inspired initialization for denser packing
            max_initial_r_val = 0.08
            hex_spacing = 0.15
            
            for k in range(N_CIRCLES):
                r = random.uniform(0.01, max_initial_r_val)
                x_min, x_max = r, 1.0 - r
                y_min, y_max = r, 1.0 - r
                
                if k < 20:  # Hexagonal pattern for first 20 circles
                    row = k // 5
                    col = k % 5
                    offset = (row % 2) * hex_spacing * 0.5
                    
                    base_x = 0.1 + col * hex_spacing + offset
                    base_y = 0.1 + row * hex_spacing * 0.866  # sqrt(3)/2 for hex geometry
                    
                    x = np.clip(base_x + random.uniform(-0.03, 0.03), x_min, x_max)
                    y = np.clip(base_y + random.uniform(-0.03, 0.03), y_min, y_max)
                else:  # Random placement for remaining circles
                    x = random.uniform(x_min, x_max)
                    y = random.uniform(y_min, y_max)
                
                genes.extend([x, y, r])
        
        else:  # 'random' strategy
            # Purely random initialization
            max_initial_r_val = 0.08 # Increased max initial radius for random strategy
            for k in range(N_CIRCLES):
                r = random.uniform(0.01, max_initial_r_val)
                x = random.uniform(r, 1.0 - r)
                y = random.uniform(r, 1.0 - r)
                genes.extend([x, y, r])
        
        return creator.Individual(genes)

    toolbox.register("individual", create_individual)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)

    # Genetic Operators
    toolbox.register("evaluate", evaluate)
    toolbox.register("mate", tools.cxBlend, alpha=0.5)
    toolbox.register("mutate", tools.mutGaussian, mu=0, sigma=0.03, indpb=0.15) # Slightly increased mutation sigma for broader exploration
    toolbox.register("select", tools.selTournament, tournsize=3)

    # Run the EA with a larger Hall of Fame to capture more diverse good solutions
    pop = toolbox.population(n=POP_SIZE)
    hof = tools.HallOfFame(10)  # Keep top 10 solutions for multi-start local optimization
    algorithms.eaSimple(pop, toolbox, cxpb=CXPB, mutpb=MUTPB, ngen=N_GEN, 
                        halloffame=hof, verbose=False)

    best_individual_from_ea = np.array(hof[0])
    
    # --- 2. Local Refinement: SLSQP Optimization (SciPy) ---
    
    # Objective function and its analytical gradient
    def objective_func_and_grad(x):
        # Objective value: -sum(radii)
        obj_val = -np.sum(x[2::3])

        # Gradient: -1 for radii components, 0 for x, y components
        grad = np.zeros_like(x)
        grad[2::3] = -1.0
        return obj_val, grad

    # Analytical Jacobians for constraints
    def jac_r_i(x, i):
        grad = np.zeros_like(x)
        grad[i*3+2] = 1.0
        return grad

    def jac_x_minus_r_i(x, i):
        grad = np.zeros_like(x)
        grad[i*3] = 1.0
        grad[i*3+2] = -1.0
        return grad

    def jac_1_minus_x_minus_r_i(x, i):
        grad = np.zeros_like(x)
        grad[i*3] = -1.0
        grad[i*3+2] = -1.0
        return grad
    
    def jac_y_minus_r_i(x, i):
        grad = np.zeros_like(x)
        grad[i*3+1] = 1.0
        grad[i*3+2] = -1.0
        return grad

    def jac_1_minus_y_minus_r_i(x, i):
        grad = np.zeros_like(x)
        grad[i*3+1] = -1.0
        grad[i*3+2] = -1.0
        return grad

    def jac_overlap(x, i, j):
        grad = np.zeros_like(x)
        
        xi, yi, ri = x[i*3], x[i*3+1], x[i*3+2]
        xj, yj, rj = x[j*3], x[j*3+1], x[j*3+2]

        # Derivatives with respect to circle i parameters
        grad[i*3] = 2 * (xi - xj)               # d/dx_i
        grad[i*3+1] = 2 * (yi - yj)             # d/dy_i
        grad[i*3+2] = -2 * (ri + rj)            # d/dr_i

        # Derivatives with respect to circle j parameters
        grad[j*3] = -2 * (xi - xj)              # d/dx_j
        grad[j*3+1] = -2 * (yi - yj)            # d/dy_j
        grad[j*3+2] = -2 * (ri + rj)            # d/dr_j
        
        return grad

    # Define hard constraints for the local optimizer with analytical Jacobians
    constraints = []
    # a) Containment: r_i >= 0, x_i-r_i >= 0, 1-x_i-r_i >= 0, etc.
    for i in range(N_CIRCLES):
        constraints.append({'type': 'ineq', 'fun': lambda x, i=i: x[i*3+2], 'jac': lambda x, i=i: jac_r_i(x, i)})
        constraints.append({'type': 'ineq', 'fun': lambda x, i=i: x[i*3] - x[i*3+2], 'jac': lambda x, i=i: jac_x_minus_r_i(x, i)})
        constraints.append({'type': 'ineq', 'fun': lambda x, i=i: 1.0 - x[i*3] - x[i*3+2], 'jac': lambda x, i=i: jac_1_minus_x_minus_r_i(x, i)})
        constraints.append({'type': 'ineq', 'fun': lambda x, i=i: x[i*3+1] - x[i*3+2], 'jac': lambda x, i=i: jac_y_minus_r_i(x, i)})
        constraints.append({'type': 'ineq', 'fun': lambda x, i=i: 1.0 - x[i*3+1] - x[i*3+2], 'jac': lambda x, i=i: jac_1_minus_y_minus_r_i(x, i)})

    # b) Non-overlap: (x_i-x_j)^2 + (y_i-y_j)^2 - (r_i+r_j)^2 >= 0
    for i in range(N_CIRCLES):
        for j in range(i + 1, N_CIRCLES):
            constraints.append({
                'type': 'ineq',
                'fun': lambda x, i=i, j=j: \
                    (x[i*3] - x[j*3])**2 + (x[i*3+1] - x[j*3+1])**2 - (x[i*3+2] + x[j*3+2])**2,
                'jac': lambda x, i=i, j=j: jac_overlap(x, i, j)
            })
    
    # Define bounds for each variable [x, y, r]
    bounds = []
    for _ in range(N_CIRCLES):
        bounds.extend([(0.0, 1.0), (0.0, 1.0), (0.0, 0.5)])

    # Physics-based refinement function for resolving overlaps
    @numba.jit(nopython=True, fastmath=True)
    def physics_refinement(circles, n_iterations=400, dt=0.001, damping=0.95): # Increased iterations
        """Apply repulsive forces to resolve overlaps while preserving structure."""
        n = circles.shape[0]
        velocities = np.zeros_like(circles)
        
        for iteration in range(n_iterations):
            forces = np.zeros_like(circles)
            
            # Calculate repulsive forces between overlapping circles
            for i in range(n):
                for j in range(i + 1, n):
                    xi, yi, ri = circles[i]
                    xj, yj, rj = circles[j]
                    
                    dx = xi - xj
                    dy = yi - yj
                    dist = np.sqrt(dx*dx + dy*dy)
                    min_dist = ri + rj
                    
                    if dist < min_dist and dist > 1e-10:
                        # Repulsive force proportional to overlap
                        overlap = min_dist - dist
                        force_magnitude = overlap * 0.25 # Increased force magnitude for stronger push
                        
                        # Normalize direction
                        fx = (dx / dist) * force_magnitude
                        fy = (dy / dist) * force_magnitude
                        
                        # Apply equal and opposite forces
                        forces[i, 0] += fx
                        forces[i, 1] += fy
                        forces[j, 0] -= fx
                        forces[j, 1] -= fy
                
                # Boundary forces to keep circles inside unit square
                x, y, r = circles[i]
                if x - r < 0:
                    forces[i, 0] += (r - x) * 1.0 # Increased boundary force
                if x + r > 1:
                    forces[i, 0] -= (x + r - 1) * 1.0
                if y - r < 0:
                    forces[i, 1] += (r - y) * 1.0
                if y + r > 1:
                    forces[i, 1] -= (y + r - 1) * 1.0
            
            # Update velocities and positions
            velocities[:, :2] = (velocities[:, :2] + forces[:, :2] * dt) * damping
            circles[:, :2] += velocities[:, :2] * dt
            
            # Clamp positions to valid ranges
            for i in range(n):
                r = circles[i, 2]
                circles[i, 0] = max(r, min(1.0 - r, circles[i, 0]))
                circles[i, 1] = max(r, min(1.0 - r, circles[i, 1]))
        
        return circles

    # Multi-start local optimization on top solutions from Hall of Fame
    best_result = None
    best_sum_radii = -np.inf
    
    # Try local optimization on multiple promising solutions (up to 10 from hof)
    for idx in range(min(10, len(hof))): # Use up to 10 best individuals
        candidate = np.array(hof[idx])
        
        # Run SLSQP optimizer with analytical gradients
        res = minimize(objective_func_and_grad, candidate, method='SLSQP', jac=True, bounds=bounds,
                       constraints=constraints, options={'maxiter': 2500, 'ftol': 1e-10, 'gtol': 1e-6}) # Increased maxiter
        
        if res.success:
            candidate_circles = res.x.reshape((N_CIRCLES, 3))
            candidate_sum = np.sum(candidate_circles[:, 2])
            
            # Verify constraints are satisfied (using a small tolerance)
            overlap_p, contain_p = calculate_violations(candidate_circles)
            if overlap_p < 1e-7 and contain_p < 1e-7:  # Tighter feasibility check
                if candidate_sum > best_sum_radii:
                    best_sum_radii = candidate_sum
                    best_result = candidate_circles
    
    # If no local optimization succeeded or found a feasible solution, use EA result with physics refinement
    if best_result is None:
        best_result = best_individual_from_ea.reshape((N_CIRCLES, 3))
    
    # Apply physics-based refinement to resolve any remaining overlaps and push to boundaries
    final_circles = physics_refinement(best_result.copy())
    
    # Final validation and potential radius adjustment (as a last resort)
    overlap_p, contain_p = calculate_violations(final_circles)
    if overlap_p > 1e-7 or contain_p > 1e-7:  # If still overlapping or out of bounds, slightly shrink radii
        shrink_factor = 0.998 # A smaller shrink factor for more gentle adjustment
        initial_sum_radii = np.sum(final_circles[:, 2])
        attempts = 0
        max_attempts = 10
        while (overlap_p > 1e-7 or contain_p > 1e-7) and attempts < max_attempts:
            final_circles[:, 2] *= shrink_factor
            # Also adjust positions to maintain containment if radii shrink
            for i in range(N_CIRCLES):
                r = final_circles[i, 2]
                final_circles[i, 0] = max(r, min(1.0 - r, final_circles[i, 0]))
                final_circles[i, 1] = max(r, min(1.0 - r, final_circles[i, 1]))
            
            overlap_p, contain_p = calculate_violations(final_circles)
            attempts += 1
            if np.sum(final_circles[:, 2]) < initial_sum_radii * 0.95: # Prevent excessive shrinking
                break

    return final_circles


# EVOLVE-BLOCK-END
