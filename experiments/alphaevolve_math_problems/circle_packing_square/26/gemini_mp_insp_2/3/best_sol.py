# EVOLVE-BLOCK-START
import numpy as np
import random
import math
import time # Added for timing optimization
from deap import base, creator, tools, algorithms
from scipy.optimize import minimize, Bounds, NonlinearConstraint # Added Bounds, NonlinearConstraint
from scipy.stats import qmc
from numba import jit # Added numba for JIT compilation
from joblib import Parallel, delayed # Added joblib for parallel processing

# Ensure reproducibility
RANDOM_SEED = 42
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

N_CIRCLES = 26
MIN_RADIUS_EPSILON = 1e-4 # Consistent minimum allowed radius for numerical stability
MAX_TIME = 170 # Max time for the entire optimization process (from Inspiration 2)

# --- DEAP Setup ---
# Create a custom type for the evolutionary algorithm
# We want to maximize the fitness value (sum of radii - penalties)
try: # Wrap in try-except to prevent re-creation errors in environments like notebooks
    creator.create("FitnessMax", base.Fitness, weights=(1.0,))
    creator.create("Individual", list, fitness=creator.FitnessMax)
except Exception:
    pass

# --- Helper Functions (Numba JIT for performance) ---
@jit(nopython=True)
def _calculate_penalties_numba(circles_data: np.ndarray, n_circles: int, C_overlap: float, C_boundary: float, C_small_r: float, min_radius_epsilon: float) -> float:
    """
    Calculates penalties for constraint violations using Numba for performance.
    circles_data: (n_circles, 3) array of (x, y, r)
    """
    total_overlap_penalty = 0.0
    total_boundary_penalty = 0.0
    total_small_radius_penalty = 0.0 # Dedicated penalty for radii below epsilon

    for i in range(n_circles):
        xi, yi, ri = circles_data[i, 0], circles_data[i, 1], circles_data[i, 2]

        # Minimum radius penalty
        total_small_radius_penalty += max(0.0, min_radius_epsilon - ri)

        # Boundary containment penalties (r <= x <= 1-r and r <= y <= 1-r)
        total_boundary_penalty += max(0.0, ri - xi)
        total_boundary_penalty += max(0.0, xi + ri - 1.0) # x+r > 1
        total_boundary_penalty += max(0.0, ri - yi)
        total_boundary_penalty += max(0.0, yi + ri - 1.0) # y+r > 1

        # Overlap penalties
        for j in range(i + 1, n_circles): # Only compare unique pairs
            xj, yj, rj = circles_data[j, 0], circles_data[j, 1], circles_data[j, 2]
            
            dx = xi - xj
            dy = yi - yj
            dist_sq = dx*dx + dy*dy
            
            min_dist = ri + rj
            min_dist_sq = min_dist * min_dist
            
            # Penalize if circles overlap (dist_sq < min_dist_sq)
            total_overlap_penalty += max(0.0, min_dist_sq - dist_sq)

    return C_overlap * total_overlap_penalty + C_boundary * total_boundary_penalty + C_small_r * total_small_radius_penalty

def _evaluate_configuration(individual, n_circles, C_overlap, C_boundary, C_small_r):
    """
    Evaluates the fitness of a circle packing configuration for DEAP.
    Maximizes sum of radii, penalizes overlaps and out-of-bounds circles.
    Numba-optimized version is called internally.
    """
    circles_data = np.array(individual).reshape(n_circles, 3)
    
    radii = circles_data[:, 2]
    sum_radii = np.sum(radii)

    # Calculate penalties using the Numba-optimized function
    total_penalty = _calculate_penalties_numba(circles_data, n_circles, C_overlap, C_boundary, C_small_r, MIN_RADIUS_EPSILON)
    
    # Fitness function: Maximize sum_radii, minimize penalties
    fitness = sum_radii - total_penalty
    
    return fitness, # DEAP expects a tuple for fitness values

def _get_circle_data(params: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Extracts x, y, r from flattened params array."""
    x = params[0::3]
    y = params[1::3]
    r = params[2::3]
    return x, y, r

def _scipy_objective(params: np.ndarray) -> float:
    """Objective function to minimize: negative sum of radii."""
    radii = params[2::3]
    return -np.sum(radii)

def _scipy_objective_jac(params: np.ndarray) -> np.ndarray:
    """Analytical Jacobian of objective function."""
    n = len(params) // 3
    jac = np.zeros(len(params))
    jac[2::3] = -1.0  # df/dr_i = -1
    return jac

def _containment_constraints_func(params: np.ndarray) -> np.ndarray:
    """Containment constraints: all circles within [0,1]x[0,1]. g(x) >= 0"""
    x, y, r = _get_circle_data(params)
    return np.concatenate([x - r, 1 - x - r, y - r, 1 - y - r])

def _containment_constraints_jac(params: np.ndarray) -> np.ndarray:
    """Analytical Jacobian of containment constraints."""
    n = len(params) // 3
    jac = np.zeros((4 * n, len(params)))
    
    for i in range(n):
        # x_i - r_i >= 0
        jac[i, 3*i] = 1      # dx_i
        jac[i, 3*i + 2] = -1 # dr_i
        
        # 1 - x_i - r_i >= 0  
        jac[n + i, 3*i] = -1     # dx_i
        jac[n + i, 3*i + 2] = -1 # dr_i
        
        # y_i - r_i >= 0
        jac[2*n + i, 3*i + 1] = 1  # dy_i
        jac[2*n + i, 3*i + 2] = -1 # dr_i
        
        # 1 - y_i - r_i >= 0
        jac[3*n + i, 3*i + 1] = -1 # dy_i
        jac[3*n + i, 3*i + 2] = -1 # dr_i
    
    return jac

def _non_overlap_constraints_func(params: np.ndarray) -> np.ndarray:
    """Non-overlap constraints: distance^2 - (sum of radii)^2 >= 0 for all pairs."""
    x, y, r = _get_circle_data(params)
    n = len(x)
    
    i_indices, j_indices = np.triu_indices(n, k=1)
    dx = x[i_indices] - x[j_indices]
    dy = y[i_indices] - y[j_indices]
    dist_sq = dx**2 + dy**2
    r_sum_sq = (r[i_indices] + r[j_indices])**2
    
    return dist_sq - r_sum_sq

def _non_overlap_constraints_jac(params: np.ndarray) -> np.ndarray:
    """Analytical Jacobian of non-overlap constraints."""
    x, y, r = _get_circle_data(params)
    n = len(x)
    num_pairs = n * (n - 1) // 2
    jac = np.zeros((num_pairs, len(params)))
    
    pair_idx = 0
    for i in range(n):
        for j in range(i + 1, n):
            dx = x[i] - x[j]
            dy = y[i] - y[j]
            r_sum = r[i] + r[j]
            
            # Derivative w.r.t. x_i, y_i, r_i
            jac[pair_idx, 3*i] = 2 * dx      # d/dx_i (dx^2)
            jac[pair_idx, 3*i + 1] = 2 * dy  # d/dy_i (dy^2)
            jac[pair_idx, 3*i + 2] = -2 * r_sum  # d/dr_i (r_sum^2)
            
            # Derivative w.r.t. x_j, y_j, r_j
            jac[pair_idx, 3*j] = -2 * dx     # d/dx_j (dx^2)
            jac[pair_idx, 3*j + 1] = -2 * dy # d/dy_j (dy^2)
            jac[pair_idx, 3*j + 2] = -2 * r_sum  # d/dr_j (r_sum^2)
            
            pair_idx += 1
    
    return jac

def _is_valid_configuration(individual, n_circles, tol=1e-6):
    """
    Checks if a configuration is valid (no overlaps, all contained) within a tolerance.
    (Adapted from Inspiration 2 for robustness)
    """
    circles_data = np.array(individual).reshape(n_circles, 3)
    x = circles_data[:, 0]
    y = circles_data[:, 1]
    r = circles_data[:, 2]

    # Check boundary containment
    if not (np.all(x - r >= -tol) and np.all(x + r <= 1 + tol) and
            np.all(y - r >= -tol) and np.all(y + r <= 1 + tol)):
        return False

    # Check non-overlap
    dx = x[:, np.newaxis] - x
    dy = y[:, np.newaxis] - y
    dist_sq = dx**2 + dy**2
    r_sum = r[:, np.newaxis] + r
    r_sum_sq = r_sum**2
    
    upper_tri_indices = np.triu_indices(n_circles, k=1)
    
    if not np.all(dist_sq[upper_tri_indices] >= r_sum_sq[upper_tri_indices] - tol):
        return False
        
    # Check for negative or too small radii
    if not np.all(r >= MIN_RADIUS_EPSILON - tol):
        return False

    return True

def generate_initial_params(n_circles):
    """Generates a flattened array of initial (x, y, r) parameters using Sobol sequence."""
    params = []
    # Use Sobol sequence for a more uniform initial (x,y) distribution
    sampler = qmc.Sobol(d=2, scramble=True, seed=RANDOM_SEED)
    initial_xy = sampler.random(n=n_circles)

    for i in range(n_circles):
        x, y = initial_xy[i]
        # Start with small, random radii to allow them to grow into place
        r = np.random.uniform(MIN_RADIUS_EPSILON, 0.1) # Increased upper bound for initial radii
        params.extend([x, y, r])
    
    # Clipping is a good safeguard, though Sobol points are in [0,1]
    params_np = np.array(params)
    params_np[0::3] = np.clip(params_np[0::3], MIN_RADIUS_EPSILON, 1 - MIN_RADIUS_EPSILON) # x
    params_np[1::3] = np.clip(params_np[1::3], MIN_RADIUS_EPSILON, 1 - MIN_RADIUS_EPSILON) # y
    params_np[2::3] = np.clip(params_np[2::3], MIN_RADIUS_EPSILON, 0.5) # r

    return creator.Individual(params_np.tolist())

def generate_initial_params_physics_based(n_circles: int, n_steps: int = 1200, dt: float = 0.01) -> np.ndarray:
    """
    Generate initial configuration using physics simulation with repulsive forces.
    (Adapted from Inspiration 2 with increased steps)
    """
    # Random initial positions with small radii
    initial_r = 0.03 # Small initial radius for physics simulation
    margin = initial_r
    x = np.random.uniform(margin, 1 - margin, n_circles)
    y = np.random.uniform(margin, 1 - margin, n_circles)
    r = np.full(n_circles, initial_r)
    
    # Velocities
    vx = np.zeros(n_circles)
    vy = np.zeros(n_circles)
    
    for step in range(n_steps):
        # Reset forces
        fx = np.zeros(n_circles)
        fy = np.zeros(n_circles)
        
        # Circle-circle repulsion
        for i in range(n_circles):
            for j in range(i + 1, n_circles):
                dx = x[i] - x[j]
                dy = y[i] - y[j]
                dist = np.sqrt(dx**2 + dy**2)
                min_dist = r[i] + r[j]
                
                # Apply force if too close (or overlapping)
                if dist < min_dist * 1.2: # Apply force if centers are within 1.2 * sum of radii
                    if dist < 1e-6: # Avoid division by zero
                        dist = 1e-6
                    # Force magnitude increases as overlap increases
                    force_mag = (min_dist * 1.2 - dist) / dist * 0.1 # Reduced force for stability
                    fx[i] += force_mag * dx
                    fy[i] += force_mag * dy
                    fx[j] -= force_mag * dx
                    fy[j] -= force_mag * dy
        
        # Boundary repulsion
        boundary_force_strength = 20.0 
        for i in range(n_circles):
            # Left/right walls
            if x[i] - r[i] < MIN_RADIUS_EPSILON:
                fx[i] += (MIN_RADIUS_EPSILON - (x[i] - r[i])) * boundary_force_strength
            if x[i] + r[i] > 1.0 - MIN_RADIUS_EPSILON:
                fx[i] -= ((x[i] + r[i]) - (1.0 - MIN_RADIUS_EPSILON)) * boundary_force_strength
            
            # Top/bottom walls  
            if y[i] - r[i] < MIN_RADIUS_EPSILON:
                fy[i] += (MIN_RADIUS_EPSILON - (y[i] - r[i])) * boundary_force_strength
            if y[i] + r[i] > 1.0 - MIN_RADIUS_EPSILON:
                fy[i] -= ((y[i] + r[i]) - (1.0 - MIN_RADIUS_EPSILON)) * boundary_force_strength
        
        # Update velocities and positions with damping
        vx = 0.9 * vx + dt * fx
        vy = 0.9 * vy + dt * fy
        x += dt * vx
        y += dt * vy
        
        # Clamp positions to ensure they are within the square, considering radii
        x = np.clip(x, MIN_RADIUS_EPSILON, 1 - MIN_RADIUS_EPSILON)
        y = np.clip(y, MIN_RADIUS_EPSILON, 1 - MIN_RADIUS_EPSILON)
    
    # Final clamping for radii to ensure strict positivity and upper bound
    r = np.clip(r, MIN_RADIUS_EPSILON, 0.5)

    # Interleave x, y, r
    params = np.zeros(3 * n_circles)
    params[0::3] = x
    params[1::3] = y
    params[2::3] = r
    
    return params


def custom_mutate(individual, mu, sigma_xy, sigma_r, indpb_xy, indpb_r):
    """
    Custom mutation operator for circle parameters (x, y, r) with specific bounds.
    Applies Gaussian mutation and then clamps values to valid ranges.
    """
    for i in range(len(individual) // 3):
        # Mutate x
        if random.random() < indpb_xy:
            individual[i*3] += random.gauss(mu, sigma_xy)
            individual[i*3] = np.clip(individual[i*3], MIN_RADIUS_EPSILON, 1 - MIN_RADIUS_EPSILON)
        # Mutate y
        if random.random() < indpb_xy:
            individual[i*3+1] += random.gauss(mu, sigma_xy)
            individual[i*3+1] = np.clip(individual[i*3+1], MIN_RADIUS_EPSILON, 1 - MIN_RADIUS_EPSILON)
        # Mutate r
        if random.random() < indpb_r:
            individual[i*3+2] += random.gauss(mu, sigma_r)
            individual[i*3+2] = np.clip(individual[i*3+2], MIN_RADIUS_EPSILON, 0.5)
    return individual,


def circle_packing26() -> np.ndarray:
    """
    Places 26 non-overlapping circles in the unit square in order to maximize the sum of radii.
    Uses a hybrid approach: Evolutionary Algorithm (DEAP) followed by parallel local optimization (SciPy).
    (Enhanced based on Inspiration 2 for better performance)

    Returns:
        circles: np.array of shape (26,3), where the i-th row (x,y,r) stores the (x,y) coordinates of the i-th circle of radius r.
    """
    n = N_CIRCLES
    start_time = time.time()

    # --- DEAP Toolbox Setup ---
    toolbox = base.Toolbox()
    # Use Sobol for initial population (from target), but also consider other initializations for local opt
    toolbox.register("individual", generate_initial_params, n_circles=n)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)

    # Evolutionary Algorithm Parameters (Adjusted for better exploration and Numba speedup)
    POP_SIZE = 500  # Increased population size from target
    GEN_COUNT = 300 # Kept from target, but with time check
    CXPB = 0.7      # Crossover probability (from target)
    MUTPB = 0.4     # Mutation probability (from target)
    
    # Penalty coefficients - these are critical for guiding the search (from Inspiration 2, C_SMALL_R from Insp 1)
    C_OVERLAP = 5000.0  # Significantly increased high penalty for overlaps
    C_BOUNDARY = 2000.0 # Increased high penalty for going out of bounds
    C_SMALL_R = 1000.0 # Reduced penalty for radii below MIN_RADIUS_EPSILON, relying more on clipping

    # Initial mutation strengths for adaptive strategy (from target)
    INITIAL_SIGMA_XY = 0.05
    INITIAL_SIGMA_R = 0.02 # Increased from 0.01 for more exploration in radii
    INDPB_XY = 0.2
    INDPB_R = 0.3

    toolbox.register("evaluate", _evaluate_configuration, n_circles=n, C_overlap=C_OVERLAP, C_boundary=C_BOUNDARY, C_small_r=C_SMALL_R)
    toolbox.register("mate", tools.cxBlend, alpha=0.5) # Blend crossover (from target)
    # The "mutate" operator will be re-registered inside the loop with adaptive sigmas
    toolbox.register("select", tools.selTournament, tournsize=5) # Tournament selection (from target)

    # --- Run Custom Evolutionary Algorithm with Adaptive Mutation and Time Limit ---
    pop = toolbox.population(n=POP_SIZE)
    hof = tools.HallOfFame(1) # Store only the best individual from EA
    
    # Evaluate the initial population
    fitnesses = list(map(toolbox.evaluate, pop))
    for ind, fit in zip(pop, fitnesses):
        ind.fitness.values = fit
    hof.update(pop)

    ea_time_limit = start_time + MAX_TIME * 0.5 # Allocate 50% of time to EA for more exploration

    for g in range(GEN_COUNT):
        if time.time() > ea_time_limit:
            # print(f"EA time limit reached after {g} generations. Elapsed: {time.time() - start_time:.2f}s")
            break
        
        # Adapt mutation strength: linearly decrease sigma over generations
        progress = g / GEN_COUNT
        # Sigmas decrease by 90% from their initial value over the run
        current_sigma_xy = INITIAL_SIGMA_XY * (1 - 0.9 * progress)
        current_sigma_r = INITIAL_SIGMA_R * (1 - 0.9 * progress)
        toolbox.register("mutate", custom_mutate, mu=0, sigma_xy=current_sigma_xy, sigma_r=current_sigma_r, indpb_xy=INDPB_XY, indpb_r=INDPB_R)

        # Standard generational step (from eaSimple)
        offspring = toolbox.select(pop, len(pop))
        offspring = list(map(toolbox.clone, offspring))

        for child1, child2 in zip(offspring[::2], offspring[1::2]):
            if random.random() < CXPB:
                toolbox.mate(child1, child2)
                del child1.fitness.values
                del child2.fitness.values
        
        for mutant in offspring:
            if random.random() < MUTPB:
                toolbox.mutate(mutant)
                del mutant.fitness.values

        invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
        fitnesses = map(toolbox.evaluate, invalid_ind)
        for ind, fit in zip(invalid_ind, fitnesses):
            ind.fitness.values = fit
        
        pop[:] = offspring
        hof.update(pop)
    
    best_individual_ea = hof[0]
    
    # --- Local Refinement using SciPy's minimize (Parallelized) ---
    
    # Define bounds for SciPy (from Inspiration 2)
    lower_bounds = np.tile([MIN_RADIUS_EPSILON, MIN_RADIUS_EPSILON, MIN_RADIUS_EPSILON], n)
    upper_bounds = np.tile([1.0 - MIN_RADIUS_EPSILON, 1.0 - MIN_RADIUS_EPSILON, 0.5], n)
    bounds = Bounds(lower_bounds, upper_bounds)
    
    # Define constraints with analytical Jacobians (from Inspiration 2)
    containment_nlc = NonlinearConstraint(
        _containment_constraints_func, 
        np.zeros(4 * n), # Lower bound for constraints (>= 0)
        np.full(4 * n, np.inf), # Upper bound for constraints
        jac=_containment_constraints_jac
    )
    
    num_pairs = n * (n - 1) // 2
    non_overlap_nlc = NonlinearConstraint(
        _non_overlap_constraints_func,
        np.zeros(num_pairs), # Lower bound for constraints (>= 0)
        np.full(num_pairs, np.inf), # Upper bound for constraints
        jac=_non_overlap_constraints_jac
    )
    
    constraints = [containment_nlc, non_overlap_nlc]
    
    # Generate multiple initial guesses for local optimization (from Inspiration 2 and 1)
    initializations_for_local_opt = []

    # 1. Best from EA
    initializations_for_local_opt.append(np.array(best_individual_ea))
    
    # 2. Physics-based initialization (from Inspiration 2)
    current_random_state_py = random.getstate()
    current_np_random_state = np.random.get_state()
    random.seed(RANDOM_SEED + 1)
    np.random.seed(RANDOM_SEED + 1)
    initializations_for_local_opt.append(generate_initial_params_physics_based(n_circles=n, n_steps=1200, dt=0.01))
    random.setstate(current_random_state_py)
    np.random.set_state(current_np_random_state)

    # 3. Add diverse structured initializations (inspired by Inspiration 1's generate_initial_guesses)
    def _generate_diverse_local_opt_structured_guesses(n_circles: int, base_seed: int) -> list[np.ndarray]:
        guesses_structured = []
        
        # Grid-based placements (from Inspiration 1)
        grid_size = int(np.ceil(np.sqrt(n_circles)))
        initial_r_grid = 0.5 / grid_size 
        
        for offset_x in [0.0, 0.05, -0.05]: 
            for offset_y in [0.0, 0.05, -0.05]: 
                params = np.zeros(3 * n_circles)
                for i in range(n_circles):
                    row = i // grid_size
                    col = i % grid_size
                    params[3*i] = (col + 0.5 + offset_x) / grid_size      # x
                    params[3*i+1] = (row + 0.5 + offset_y) / grid_size    # y  
                    # Add some randomness to radii, ensuring non-negativity
                    np.random.seed(base_seed + i * 10 + int(offset_x*100) + int(offset_y*100))
                    params[3*i+2] = np.clip(initial_r_grid * (0.8 + 0.4 * np.random.rand()), MIN_RADIUS_EPSILON, 0.5)
                guesses_structured.append(params)
        
        # Hexagonal-inspired patterns (from Inspiration 1)
        for scale in [0.7, 0.9, 1.0]: 
            for perturbation_strength in [0.0, 0.03]: 
                params = np.zeros(3 * n_circles)
                for i in range(n_circles):
                    angle = 2 * np.pi * i / n_circles
                    radius_pos = 0.3 * scale
                    
                    np.random.seed(base_seed + i * 20 + int(scale*100) + int(perturbation_strength*1000))
                    x_pos = 0.5 + radius_pos * np.cos(angle + i*0.1) + perturbation_strength * (np.random.rand() - 0.5) 
                    y_pos = 0.5 + radius_pos * np.sin(angle + i*0.1) + perturbation_strength * (np.random.rand() - 0.5)
                    
                    params[3*i] = np.clip(x_pos, MIN_RADIUS_EPSILON, 1 - MIN_RADIUS_EPSILON)
                    params[3*i+1] = np.clip(y_pos, MIN_RADIUS_EPSILON, 1 - MIN_RADIUS_EPSILON)
                    params[3*i+2] = np.clip(0.04 * (0.8 + 0.4 * np.random.rand()), MIN_RADIUS_EPSILON, 0.5)
                guesses_structured.append(params)
                
        # Central cluster/spiral pattern (from Inspiration 1)
        for dispersion in [0.0, 0.05]: 
            params = np.zeros(3 * n_circles)
            center_x, center_y = 0.5, 0.5
            base_radius_cluster = 0.07 
            for i in range(n_circles):
                ring_idx = 0
                if i < 6: ring_idx = 0 
                elif i < 16: ring_idx = 1 
                else: ring_idx = 2 

                r_offset_base = [0.1, 0.25, 0.4][ring_idx] 
                np.random.seed(base_seed + i * 30 + int(dispersion*1000))
                r_offset = r_offset_base + dispersion * (np.random.rand() - 0.5)
                
                angle = 2 * np.pi * i / n_circles + i * 0.2 
                
                params[3*i] = np.clip(center_x + r_offset * np.cos(angle), MIN_RADIUS_EPSILON, 1 - MIN_RADIUS_EPSILON)
                params[3*i+1] = np.clip(center_y + r_offset * np.sin(angle), MIN_RADIUS_EPSILON, 1 - MIN_RADIUS_EPSILON)
                params[3*i+2] = np.clip(base_radius_cluster * (0.7 + 0.6 * np.random.rand()), MIN_RADIUS_EPSILON, 0.5)
            guesses_structured.append(params)

        return guesses_structured

    current_random_state_py = random.getstate()
    current_np_random_state = np.random.get_state()
    np.random.seed(RANDOM_SEED + 3) # Use a different seed for these structured starts
    initializations_for_local_opt.extend(_generate_diverse_local_opt_structured_guesses(n, RANDOM_SEED + 3))
    random.setstate(current_random_state_py)
    np.random.set_state(current_np_random_state)

    # 4. Add a few more purely random starts (with slightly higher initial radii)
    current_random_state_py = random.getstate()
    current_np_random_state = np.random.get_state()
    np.random.seed(RANDOM_SEED + 4) # Use a different seed for these random starts
    for _ in range(3): # Three additional random starts
        initial_r_rand = np.random.uniform(0.04, 0.07)
        x_rand = np.random.uniform(initial_r_rand, 1 - initial_r_rand, n)
        y_rand = np.random.uniform(initial_r_rand, 1 - initial_r_rand, n)
        r_rand = np.full(n, initial_r_rand)
        rand_params = np.zeros(3 * n)
        rand_params[0::3] = x_rand
        rand_params[1::3] = y_rand
        rand_params[2::3] = r_rand
        initializations_for_local_opt.append(rand_params)
    random.setstate(current_random_state_py)
    np.random.set_state(current_np_random_state)


    # Helper function to run a single SciPy minimize call (from Inspiration 2)
    def run_scipy_minimize(initial_params, global_time_limit):
        if time.time() > global_time_limit:
            return None # Skip if global time budget exceeded
            
        options = {
            'maxiter': 4000, # Increased maxiter for trust-constr
            'disp': False, 
            'verbose': 0,
            'gtol': 1e-9, # Tighter gradient tolerance for trust-constr
            'xtol': 1e-9, # Tighter step tolerance for trust-constr
        }
        
        try:
            result = minimize(
                _scipy_objective,
                initial_params,
                method='trust-constr', # Switched to trust-constr for robustness
                bounds=bounds,
                constraints=constraints,
                jac=_scipy_objective_jac, # Pass analytical Jacobian for objective
                options=options
            )
            
            # Additional check: ensure constraints are satisfied at the end within a tolerance
            final_params = result.x
            # Check containment and non-overlap constraints directly
            if np.all(_containment_constraints_func(final_params) >= -1e-6) and \
               np.all(_non_overlap_constraints_func(final_params) >= -1e-6) and \
               np.all(final_params[2::3] >= MIN_RADIUS_EPSILON - 1e-6) and \
               result.success: # Also check if the optimizer reported success
                return -result.fun, final_params # Return sum_radii and params
            else:
                return None
            
        except Exception as e:
            # print(f"Optimization failed with exception: {e}") # Debugging
            return None # Optimization failed
            
    # Allocate remaining time to local optimization
    global_end_time = start_time + MAX_TIME
    
    # Run local optimizations in parallel (from Inspiration 2)
    results = Parallel(n_jobs=-1, verbose=0)(
        delayed(run_scipy_minimize)(init_params, global_end_time)
        for init_params in initializations_for_local_opt
    )

    best_sum_radii = -np.inf
    best_optimized_params = None
    
    for res in results:
        if res is not None:
            current_sum_radii, current_params = res
            if current_sum_radii > best_sum_radii:
                best_sum_radii = current_sum_radii
                best_optimized_params = current_params
    
    # Fallback if no successful optimization found
    if best_optimized_params is None:
        # If all local optimizations fail, use the best from EA as a fallback
        final_circles_result = np.array(best_individual_ea).reshape((n, 3))
    else:
        final_circles_result = best_optimized_params.reshape((n, 3))
        
    # Final clamping for radii to ensure strict positivity and upper bound
    final_circles_result[:, 2] = np.clip(final_circles_result[:, 2], MIN_RADIUS_EPSILON, 0.5)
    
    return final_circles_result


# EVOLVE-BLOCK-END
