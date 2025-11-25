# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize # Added import

# Helper function for validation (moved outside for general utility)
def validate_circles(circles: np.ndarray) -> dict:
    n = circles.shape[0]
    x = circles[:, 0]
    y = circles[:, 1]
    r = circles[:, 2]

    results = {
        'all_positive_radii': bool(np.all(r > 0)),
        'all_contained': True,
        'no_overlaps': True,
        'violated_containment_count': 0,
        'violated_overlap_count': 0,
        'sum_radii': np.sum(r)
    }

    # Check containment
    # Using a small tolerance for floating point comparisons to avoid spurious violations
    containment_violations_x_min = np.sum(x - r < -1e-9)
    containment_violations_x_max = np.sum(1 - x - r < -1e-9)
    containment_violations_y_min = np.sum(y - r < -1e-9)
    containment_violations_y_max = np.sum(1 - y - r < -1e-9)

    results['violated_containment_count'] = (
        containment_violations_x_min + containment_violations_x_max +
        containment_violations_y_min + containment_violations_y_max
    )
    if results['violated_containment_count'] > 0:
        results['all_contained'] = False

    # Check non-overlap
    overlap_violations = 0
    for i in range(n):
        for j in range(i + 1, n): # Iterate over unique pairs i < j
            dist_sq = (x[i] - x[j])**2 + (y[i] - y[j])**2
            min_dist_sq = (r[i] + r[j])**2
            if dist_sq < min_dist_sq - 1e-9: # Allow small tolerance
                overlap_violations += 1
    
    results['violated_overlap_count'] = overlap_violations
    if overlap_violations > 0:
        results['no_overlaps'] = False
        
    return results


import random
from deap import base, creator, tools, algorithms

def circle_packing32() -> np.ndarray:
    """
    Places 32 non-overlapping circles in the unit square in order to maximize the sum of radii,
    using a hybrid Genetic Algorithm + Local Search (SLSQP) approach.
    """
    N_CIRCLES = 32
    BENCHMARK_SUM_RADII = 2.937944526205518

    # --- STAGE 1: GLOBAL SEARCH WITH GENETIC ALGORITHM ---
    
    # Set fixed seeds for reproducibility
    random.seed(42)
    np.random.seed(42)

    # DEAP setup: Fitness aims for maximization, Individual is a numpy array
    creator.create("FitnessMax", base.Fitness, weights=(1.0,))
    creator.create("Individual", np.ndarray, fitness=creator.FitnessMax)

    toolbox = base.Toolbox()

    # Initialization: Grid-based placement for a high-quality, non-overlapping start
    def init_individual():
        grid_size = 6  # 6x6 grid provides 36 cells for 32 circles
        coords = []
        for i in range(grid_size):
            for j in range(grid_size):
                coords.append(((i + 0.5) / grid_size, (j + 0.5) / grid_size))
        
        selected_coords = random.sample(coords, N_CIRCLES)
        params = np.zeros(N_CIRCLES * 3)
        initial_radius = 1 / (2 * grid_size) - 1e-4 # Max radius for grid cell, minus epsilon
        for i in range(N_CIRCLES):
            params[3 * i] = selected_coords[i][0]
            params[3 * i + 1] = selected_coords[i][1]
            params[3 * i + 2] = initial_radius
        return creator.Individual(params)

    toolbox.register("individual", init_individual)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)

    # Fitness function with penalties for constraint violations
    PENALTY_OVERLAP = 15.0
    PENALTY_BOUNDARY = 25.0

    def evaluate(individual):
        x, y, r = individual[0::3], individual[1::3], individual[2::3]
        sum_radii = np.sum(r)
        
        # Overlap penalty
        total_overlap = 0.0
        for i in range(N_CIRCLES):
            for j in range(i + 1, N_CIRCLES):
                dist = np.sqrt((x[i] - x[j])**2 + (y[i] - y[j])**2)
                overlap = r[i] + r[j] - dist
                if overlap > 0:
                    total_overlap += overlap

        # Boundary penalty
        total_boundary_violation = np.sum(np.maximum(0, r - x)) + \
                                   np.sum(np.maximum(0, x + r - 1)) + \
                                   np.sum(np.maximum(0, r - y)) + \
                                   np.sum(np.maximum(0, y + r - 1))
        
        fitness = sum_radii - PENALTY_OVERLAP * total_overlap - PENALTY_BOUNDARY * total_boundary_violation
        return (fitness,)

    toolbox.register("evaluate", evaluate)
    
    # Genetic Operators
    toolbox.register("mate", tools.cxBlend, alpha=0.5)
    toolbox.register("mutate", tools.mutGaussian, mu=0, sigma=0.015, indpb=0.1)
    toolbox.register("select", tools.selTournament, tournsize=3)

    # Run the GA
    POP_SIZE, N_GEN, CXPB, MUTPB = 200, 250, 0.8, 0.3
    pop = toolbox.population(n=POP_SIZE)
    hof = tools.HallOfFame(1)
    algorithms.eaSimple(pop, toolbox, cxpb=CXPB, mutpb=MUTPB, ngen=N_GEN, 
                        halloffame=hof, verbose=False)

    # --- STAGE 2: LOCAL SEARCH REFINEMENT ---
    
    ga_best_solution = np.array(hof[0])

    def unpack_params(params):
        return params[0::3], params[1::3], params[2::3]

    def objective(params):
        return -np.sum(params[2::3])

    constraints = []
    constraints.append({'type': 'ineq', 'fun': lambda p: p[0::3] - p[2::3]})
    constraints.append({'type': 'ineq', 'fun': lambda p: 1 - p[0::3] - p[2::3]})
    constraints.append({'type': 'ineq', 'fun': lambda p: p[1::3] - p[2::3]})
    constraints.append({'type': 'ineq', 'fun': lambda p: 1 - p[1::3] - p[2::3]})
    
    def non_overlap_constraint(params):
        x, y, r = unpack_params(params)
        violations = []
        for i in range(N_CIRCLES):
            for j in range(i + 1, N_CIRCLES):
                dist_sq = (x[i] - x[j])**2 + (y[i] - y[j])**2
                min_dist_sq = (r[i] + r[j])**2
                violations.append(dist_sq - min_dist_sq)
        return np.array(violations)
    
    constraints.append({'type': 'ineq', 'fun': non_overlap_constraint})
    
    bounds = []
    for _ in range(N_CIRCLES):
        bounds.extend([(0, 1), (0, 1), (1e-6, 0.5)])

    res = minimize(
        objective,
        ga_best_solution,
        method='SLSQP',
        bounds=bounds,
        constraints=constraints,
        options={'disp': False, 'maxiter': 2000, 'ftol': 1e-10}
    )

    if res.success:
        best_params = res.x
    else:
        print(f"Warning: Local search failed after GA: {res.message}. Using GA result directly.")
        best_params = ga_best_solution

    final_x, final_y, final_r = unpack_params(best_params)
    final_circles = np.vstack((final_x, final_y, final_r)).T

    # --- Reporting ---
    final_sum_radii = np.sum(final_r)
    benchmark_ratio = final_sum_radii / BENCHMARK_SUM_RADII
    final_validation = validate_circles(final_circles)

    print(f"--- Optimization Results for {N_CIRCLES} Circles ---")
    print(f"Sum of Radii: {final_sum_radii:.15f}")
    print(f"Benchmark Ratio: {benchmark_ratio:.15f}")
    print(f"Final Solution Valid: {final_validation['all_contained'] and final_validation['no_overlaps']}")
    # print(f"Validation Details:")
    # for key, value in final_validation.items():
    #     print(f"  {key}: {value}")
    print(f"-------------------------------------------")

    return final_circles


# EVOLVE-BLOCK-END
