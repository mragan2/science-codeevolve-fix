# EVOLVE-BLOCK-START
import numpy as np
import random
import itertools # Re-added for precomputing combinations of indices
import multiprocessing # Added for parallelization
from scipy.spatial import ConvexHull, QhullError
from deap import base, creator, tools, algorithms
from numba import jit # Added for performance
import sobol_seq # Added for low-discrepancy initialization

# --- Constants and Seeds ---
N_POINTS = 14
RANDOM_SEED = 42
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

# Precompute all unique combinations of 3 indices for N_POINTS once globally.
# This array is used by the Numba-jitted function, avoiding itertools within Numba.
_TRIANGLE_INDICES = np.array(list(itertools.combinations(range(N_POINTS), 3)), dtype=np.int32)

# Genetic Algorithm Parameters (tuned for N=14, combining best practices from inspirations)
POP_SIZE = 500
NGEN = 4000    # Increased generations significantly due to parallelization (from 3000)
CXPB = 0.7     # Crossover probability
MUTPB = 0.2    # Mutation probability

# Adaptive Mutation Parameters
INITIAL_SIGMA = 0.1  # Focused initial exploration (from Inspiration 1)
FINAL_SIGMA = 0.001
IND_MUT_PROB = 0.1 # Individual gene mutation probability (from Inspiration 3)

# Local Refinement Parameters (tuned for increased refinement given time budget, with boundary bias)
LOCAL_ITERATIONS = 250000 # Increased iterations for more thorough local refinement (from 150000)
LOCAL_SCALE = 0.0005 # Reduced scale for finer tuning during local refinement (from Inspiration 1)
BOUNDARY_EPSILON = 0.01 # Epsilon for boundary bias in local search (from Inspiration 3)

# --- Numba-optimized Helper Functions ---
@jit(nopython=True, cache=True, fastmath=True)
def _calculate_triangle_area(p1, p2, p3):
    """Calculates the absolute area of a triangle given three 2D points."""
    return 0.5 * abs(p1[0]*(p2[1] - p3[1]) + p2[0]*(p3[1] - p1[1]) + p3[0]*(p1[1] - p2[1]))

@jit(nopython=True, cache=True, fastmath=True)
def _find_min_area_optimized(points: np.ndarray, triangle_indices_arr: np.ndarray) -> float:
    """
    Finds the minimum triangle area for a set of points using Numba-optimized loops
    and precomputed triangle indices.
    Returns 0.0 if any triangle is degenerate (area 0).
    (Adapted from Inspiration 3)
    """
    min_area = np.finfo(points.dtype).max # Use numpy's float max for robustness
    num_combinations = triangle_indices_arr.shape[0]
    for combo_idx in range(num_combinations):
        i, j, k = triangle_indices_arr[combo_idx]
        p1, p2, p3 = points[i], points[j], points[k]
        area = _calculate_triangle_area(p1, p2, p3)
        if area < min_area:
            min_area = area
        # Early exit for collinear points
        if min_area == 0.0:
            return 0.0
    return min_area

# Helper function to calculate convex hull area (modified to handle QhullError)
def _calculate_convex_hull_area(points: np.ndarray) -> float:
    """
    Calculates the area of the convex hull of the given points.
    Returns 0.0 if fewer than 3 non-collinear points are provided, or if they are collinear.
    """
    if len(points) < 3:
        return 0.0
    try:
        hull = ConvexHull(points)
        return hull.area
    except QhullError: # Catching specific QhullError for clarity
        return 0.0
    except Exception: # Catch any other unexpected exceptions
        return 0.0

# --- DEAP Fitness Function ---
def evaluate_points(individual):
    """
    Evaluates an individual (a flattened list of coordinates) for the GA.
    This is the core fitness function.
    Returns a tuple (normalized_min_area,) as required by DEAP for single-objective.
    """
    points = np.array(individual).reshape((N_POINTS, 2))
    points = np.clip(points, 0.0, 1.0) # Enforce [0,1] bounds

    # Use the fast Numba function to find the minimum triangle area, passing precomputed indices
    min_tri_area = _find_min_area_optimized(points, _TRIANGLE_INDICES) # Now passes _TRIANGLE_INDICES
    
    # Penalize configurations with collinear or nearly-collinear points
    if min_tri_area < 1e-12: # Threshold for considering area as effectively zero
        return -1.0, # Strong penalty

    hull_area = _calculate_convex_hull_area(points) # Using the helper function
    
    # Penalize configurations with a zero or negligible convex hull area
    if hull_area < 1e-9: # Threshold for considering hull_area as effectively zero
        return -1.0, # Strong penalty

    normalized_min_area = min_tri_area / hull_area
    return normalized_min_area,

# --- Local Search Function with Boundary Bias (from Inspiration 3, refined) ---
def local_search(points: np.ndarray, evaluate_func, num_evaluations: int, perturbation_sigma: float, boundary_epsilon: float) -> np.ndarray:
    """
    Performs a greedy local search (hill-climbing) with a boundary bias to refine a solution.
    (Adapted and refined from Inspiration 3)
    """
    current_points = points.copy()
    current_fitness = evaluate_func(current_points.flatten().tolist())[0]
    
    # Use a separate RNG for local search for reproducibility
    rng_local = np.random.default_rng(seed=RANDOM_SEED + 1) 
    
    for _ in range(num_evaluations):
        candidate_points = np.copy(current_points)
        idx_to_perturb = rng_local.integers(N_POINTS) # Randomly pick one point to perturb
        original_point = candidate_points[idx_to_perturb].copy()
        perturbation = rng_local.normal(0, perturbation_sigma, 2)
        
        # Apply boundary bias: make it harder to move away from the boundary, easier to move along it.
        # This pushes points towards and along the boundary.
        if original_point[0] < boundary_epsilon: # Near left boundary
            # Reduce outward (left) movement, enhance parallel (Y) movement
            perturbation[0] = abs(perturbation[0]) * 0.2 if perturbation[0] < 0 else perturbation[0] # Dampen outward, allow inward
            perturbation[1] *= 1.5 # Boost Y perturbation
        elif original_point[0] > 1.0 - boundary_epsilon: # Near right boundary
            # Reduce outward (right) movement, enhance parallel (Y) movement
            perturbation[0] = -abs(perturbation[0]) * 0.2 if perturbation[0] > 0 else perturbation[0] # Dampen outward, allow inward
            perturbation[1] *= 1.5 # Boost Y perturbation
        
        if original_point[1] < boundary_epsilon: # Near bottom boundary
            # Reduce outward (down) movement, enhance parallel (X) movement
            perturbation[1] = abs(perturbation[1]) * 0.2 if perturbation[1] < 0 else perturbation[1] # Dampen outward, allow inward
            perturbation[0] *= 1.5 # Boost X perturbation
        elif original_point[1] > 1.0 - boundary_epsilon: # Near top boundary
            # Reduce outward (up) movement, enhance parallel (X) movement
            perturbation[1] = -abs(perturbation[1]) * 0.2 if perturbation[1] > 0 else perturbation[1] # Dampen outward, allow inward
            perturbation[0] *= 1.5 # Boost X perturbation

        candidate_points[idx_to_perturb] = np.clip(original_point + perturbation, 0.0, 1.0)
        new_fitness = evaluate_func(candidate_points.flatten().tolist())[0]
        
        # If the candidate is better, accept it
        if new_fitness > current_fitness:
            current_points = candidate_points
            current_fitness = new_fitness
            
    return current_points


def heilbronn_convex14()->np.ndarray:
    """
    Constructs an optimal arrangement of 14 points using a Genetic Algorithm (DEAP)
    to maximize the normalized minimum triangle area.
    (Re-optimized based on the best-performing inspiration programs, incorporating parallelization,
    precomputed indices, adaptive mutation, and boundary-biased local refinement)
    """
    # --- DEAP Toolbox Setup ---
    # Ensure creator types are defined only once to prevent KeyError if function is called multiple times
    if not hasattr(creator, "FitnessMax"):
        creator.create("FitnessMax", base.Fitness, weights=(1.0,)) # Maximize fitness
    if not hasattr(creator, "Individual"):
        creator.create("Individual", list, fitness=creator.FitnessMax)

    toolbox = base.Toolbox()
    
    # Initialize multiprocessing pool for parallel evaluation (from Inspiration 3)
    pool = multiprocessing.Pool()
    toolbox.register("map", pool.map) # Register the pool's map function for parallel evaluation

    toolbox.register("evaluate", evaluate_points)
    toolbox.register("mate", tools.cxBlend, alpha=0.5) # Blend crossover
    # Sigma for mutGaussian will be set adaptively within the evolution loop, so it's not registered here initially
    toolbox.register("mutate", tools.mutGaussian, mu=0, indpb=IND_MUT_PROB) # Use IND_MUT_PROB constant
    toolbox.register("select", tools.selTournament, tournsize=3) # Tournament selection

    # --- Population Initialization using Sobol Sequence ---
    # Generate POP_SIZE individuals, each with 2 * N_POINTS genes (coordinates)
    sobol_coords = sobol_seq.i4_sobol_generate(2 * N_POINTS, POP_SIZE, skip=100) # skip for variety
    pop = [creator.Individual(ind.tolist()) for ind in sobol_coords]

    # Evaluate the initial population in parallel
    fitnesses = toolbox.map(toolbox.evaluate, pop)
    for ind, fit in zip(pop, fitnesses):
        ind.fitness.values = fit

    # --- Hall of Fame to store the best individual ---
    hof = tools.HallOfFame(1)
    hof.update(pop) # Update hof with the initial population's best
    
    # --- Manual GA Loop for Adaptive Mutation ---
    # `eaSimple` doesn't easily support adaptive mutation, so we implement the loop manually.
    for g in range(NGEN):
        # Select the next generation individuals
        offspring = toolbox.select(pop, len(pop))
        # Clone the selected individuals to avoid modifying parents directly
        offspring = list(map(toolbox.clone, offspring))

        # Apply crossover on the offspring
        for child1, child2 in zip(offspring[::2], offspring[1::2]):
            if random.random() < CXPB:
                toolbox.mate(child1, child2)
                # Invalidate fitness values of the children as they have changed
                del child1.fitness.values
                del child2.fitness.values
        
        # Calculate current sigma for adaptive mutation
        # Sigma decreases linearly from INITIAL_SIGMA to FINAL_SIGMA over NGEN generations
        current_sigma = INITIAL_SIGMA - (INITIAL_SIGMA - FINAL_SIGMA) * (g / NGEN)
        
        # Apply mutation on the offspring
        for mutant in offspring:
            if random.random() < MUTPB:
                # Use the adaptive sigma for mutation
                tools.mutGaussian(mutant, mu=0, sigma=current_sigma, indpb=IND_MUT_PROB) # Use IND_MUT_PROB constant
                # Invalidate fitness value of the mutant
                del mutant.fitness.values

        # Evaluate the individuals with an invalid fitness in parallel
        invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
        fitnesses = toolbox.map(toolbox.evaluate, invalid_ind)
        for ind, fit in zip(invalid_ind, fitnesses):
            ind.fitness.values = fit

        # The population is replaced by the offspring
        pop[:] = offspring
        hof.update(pop) # Update the Hall of Fame with the current population's best
    
    # Close the multiprocessing pool after the GA loop
    pool.close()
    pool.join()

    # --- Post-GA Intensive Local Refinement (using boundary-biased local search) ---
    # Take the best individual from the GA (from the Hall of Fame) and perform a local search around it.
    best_individual = hof[0]
    optimal_points_ga = np.array(best_individual).reshape((N_POINTS, 2))
    
    # Apply the boundary-biased local search
    optimal_points = local_search(
        optimal_points_ga, 
        evaluate_points, 
        num_evaluations=LOCAL_ITERATIONS, 
        perturbation_sigma=LOCAL_SCALE,
        boundary_epsilon=BOUNDARY_EPSILON
    )
            
    # Final clip to ensure points are strictly within the [0,1]x[0,1] square for the output
    optimal_points = np.clip(optimal_points, 0.0, 1.0)
    return optimal_points

# EVOLVE-BLOCK-END