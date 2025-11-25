# EVOLVE-BLOCK-START
import numpy as np
import multiprocessing # Added for parallelization of DEAP evaluation
from scipy.spatial import ConvexHull, QhullError
from numba import jit # Import Numba for JIT compilation
import random
import sobol_seq # For low-discrepancy sequence initialization
from deap import base, creator, tools, algorithms # For Genetic Algorithm
from scipy.optimize import minimize # Added for local search refinement

# --- Constants and Seeds ---
N_POINTS = 14
RANDOM_SEED = 42
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

# Genetic Algorithm Parameters (tuned based on inspiration programs for N=14)
POP_SIZE = 500
NGEN = 4000 # Increased generations, leveraging parallelization for more thorough global search
CXPB = 0.7  # Crossover probability
MUTPB = 0.2 # Mutation probability

# Adaptive Mutation Parameters
INITIAL_SIGMA = 0.1
FINAL_SIGMA = 0.001

# Removed custom local refinement parameters as scipy.optimize.minimize is used instead


# --- Numba-optimized Helper Functions ---
@jit(nopython=True, cache=True, fastmath=True) # Added fastmath=True for potential minor speedups
def _calculate_triangle_area(p1, p2, p3):
    """Calculates the absolute area of a triangle given three 2D points."""
    return 0.5 * abs(p1[0]*(p2[1] - p3[1]) + p2[0]*(p3[1] - p1[1]) + p3[0]*(p1[1] - p2[1]))

@jit(nopython=True, cache=True, fastmath=True) # Added fastmath=True for potential minor speedups
def _find_min_area_optimized(points: np.ndarray) -> float:
    """
    Finds the minimum triangle area for a set of points using Numba-optimized loops.
    Returns 0.0 if any triangle is degenerate (area 0).
    """
    min_area = float('inf')
    num_points = points.shape[0]
    # Numba-compiled nested loops for efficiency (faster than itertools for this N)
    for i in range(num_points):
        for j in range(i + 1, num_points):
            for k in range(j + 1, num_points):
                area = _calculate_triangle_area(points[i], points[j], points[k])
                if area < min_area:
                    min_area = area
                # Early exit for collinear or near-collinear points, as min_area < 1e-12 is a strong penalty
                if min_area < 1e-12: # More robust check for floating point issues
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
        # Use 'QJ' option for robustness as observed in inspiration programs
        hull = ConvexHull(points, qhull_options='QJ')
        return hull.area
    except QhullError: # Catching specific QhullError for clarity
        return 0.0
    except Exception: # Catch any other unexpected exceptions
        return 0.0

# The objective function for optimization (to be maximized by DEAP)
def evaluate_points(individual):
    """
    Evaluates an individual (a flattened list of coordinates) for the GA.
    This is the core fitness function.
    Returns a tuple (normalized_min_area,) as required by DEAP for single-objective.
    """
    points = np.array(individual).reshape((N_POINTS, 2))
    points = np.clip(points, 0.0, 1.0) # Enforce [0,1] bounds

    # Removed redundant check for non-distinct points.
    # The min_tri_area check below already handles cases where points are coincident
    # by returning 0.0 if a triangle with identical vertices is formed.
    # if len(np.unique(points.round(decimals=8), axis=0)) < N_POINTS:
    #     return -1.0, # Strong penalty

    # Use the fast Numba function to find the minimum triangle area
    min_tri_area = _find_min_area_optimized(points)
    
    # Penalize configurations with collinear or nearly-collinear points
    if min_tri_area < 1e-12: # Threshold for considering area as effectively zero
        return -1.0, # Strong penalty

    hull_area = _calculate_convex_hull_area(points)
    
    # Penalize configurations with a zero or negligible convex hull area
    if hull_area < 1e-9: # Threshold for considering hull_area as effectively zero
        return -1.0, # Strong penalty

    normalized_min_area = min_tri_area / hull_area
    return normalized_min_area, # DEAP maximizes this value

def heilbronn_convex14() -> np.ndarray:
    """
    Construct an arrangement of n points on or inside a convex region in order to maximize the area of the
    smallest triangle formed by these points. Here n = 14.
    Uses a Genetic Algorithm with Numba optimization and post-GA local refinement.

    Returns:
        points: np.ndarray of shape (14,2) with the x,y coordinates of the points.
    """
    # --- DEAP Toolbox Setup ---
    # Ensure creator types are defined only once to prevent KeyError if function is called multiple times
    if not hasattr(creator, "FitnessMax"):
        creator.create("FitnessMax", base.Fitness, weights=(1.0,)) # Maximize fitness
    if not hasattr(creator, "Individual"):
        creator.create("Individual", list, fitness=creator.FitnessMax)

    toolbox = base.Toolbox()
    
    # --- Parallelization Setup (Inspired by Inspiration Program 1) ---
    pool = multiprocessing.Pool()
    toolbox.register("map", pool.map) # Register the parallel map function

    toolbox.register("evaluate", evaluate_points)
    toolbox.register("mate", tools.cxBlend, alpha=0.5) # Blend crossover
    # Sigma for mutGaussian will be set adaptively within the evolution loop, so it's not registered here initially
    toolbox.register("mutate", tools.mutGaussian, mu=0, indpb=0.1) 
    toolbox.register("select", tools.selTournament, tournsize=3) # Tournament selection

    # --- Population Initialization using Sobol Sequence ---
    # This provides a better-than-random starting population distribution.
    # The search space is 2*N_POINTS dimensions (x,y for each point).
    sobol_coords = sobol_seq.i4_sobol_generate(2 * N_POINTS, POP_SIZE, skip=100)
    # Convert Sobol sequence to DEAP individuals
    pop = [creator.Individual(ind.tolist()) for ind in sobol_coords]

    # Evaluate the initial population using the parallel map
    fitnesses = toolbox.map(toolbox.evaluate, pop) # Use toolbox.map for parallel evaluation
    for ind, fit in zip(pop, fitnesses):
        ind.fitness.values = fit

    # --- Hall of Fame to store the single best individual found ---
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
                tools.mutGaussian(mutant, mu=0, sigma=current_sigma, indpb=0.1)
                # Invalidate fitness value of the mutant
                del mutant.fitness.values

        # Evaluate the individuals with an invalid fitness using the parallel map
        invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
        fitnesses = toolbox.map(toolbox.evaluate, invalid_ind) # Use toolbox.map for parallel evaluation
        for ind, fit in zip(invalid_ind, fitnesses):
            ind.fitness.values = fit

        # The population is replaced by the offspring
        pop[:] = offspring
        hof.update(pop) # Update the Hall of Fame with the current population's best

    # Close the multiprocessing pool after the GA loop
    pool.close()
    pool.join()

    # --- Post-GA Intensive Local Refinement with scipy.optimize.minimize ---
    best_individual_ga = np.array(hof[0])

    # Define the objective function for scipy.minimize (which minimizes)
    def local_objective(flat_points):
        """Objective function for scipy.minimize (minimization)."""
        # The evaluate_points returns (score,), so we take [0]
        score = evaluate_points(flat_points)[0]
        # scipy.minimize minimizes, so we return negative of our maximization score.
        # Return a large penalty for invalid configurations (score < 0) to guide optimizer away.
        return 1.0 if score < 0 else -score

    # Bounds for each coordinate (0 to 1 for x, and 0 to 1 for y)
    bounds = [(0, 1)] * (N_POINTS * 2)

    # Perform local optimization using L-BFGS-B method
    polish_result = minimize(
        local_objective,
        x0=best_individual_ga.flatten(), # Initial guess from the best GA individual
        method='L-BFGS-B',
        bounds=bounds,
        options={'ftol': 1e-12, 'gtol': 1e-8, 'maxiter': 1500} # Increased maxiter for more thorough local search
    )

    # Reshape the optimized flattened array back into (N_POINTS, 2)
    # Ensure points are within bounds, although L-BFGS-B with bounds should handle this.
    optimal_points = np.clip(polish_result.x.reshape((N_POINTS, 2)), 0.0, 1.0)
    return optimal_points

# EVOLVE-BLOCK-END