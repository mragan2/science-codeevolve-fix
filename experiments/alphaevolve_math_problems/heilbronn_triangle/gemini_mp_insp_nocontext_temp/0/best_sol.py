# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import qmc
from itertools import combinations
import math

# --- Constants and Configuration ---
# Using private-like names to avoid polluting module namespace if this were part of a larger library.
_N_POINTS = 11
_SEED = 42
_SQRT3 = math.sqrt(3)

# Pre-compute combinations of point indices for speed, as this is constant.
_INDICES = np.array(list(combinations(range(_N_POINTS), 3)))

# --- Helper Functions ---

def _is_inside_triangle(points: np.ndarray) -> np.ndarray:
    """
    Vectorized check if points are inside the equilateral triangle with vertices
    at (0,0), (1,0), and (0.5, sqrt(3)/2).
    """
    # Ensure input is a 2D array for consistent processing.
    if points.ndim == 1:
        points = points.reshape(1, -1)
    
    x, y = points[:, 0], points[:, 1]
    
    # The three boundary conditions for the triangle.
    # A small tolerance (1e-9) is used for floating-point robustness,
    # ensuring points on the boundary are considered inside.
    c1 = y >= -1e-9
    c2 = y - _SQRT3 * x <= 1e-9
    c3 = y + _SQRT3 * x - _SQRT3 <= 1e-9
    
    return np.all(np.vstack([c1, c2, c3]), axis=0)

def _calculate_min_area(points: np.ndarray) -> float:
    """
    Calculates the minimum triangle area formed by any 3 points using a
    vectorized implementation of the Shoelace formula.
    """
    # Select all triplets of points based on pre-computed indices.
    p = points[_INDICES]
    
    # Shoelace formula for area, applied to all triplets at once.
    areas = 0.5 * np.abs(
        p[:, 0, 0] * (p[:, 1, 1] - p[:, 2, 1]) +
        p[:, 1, 0] * (p[:, 2, 1] - p[:, 0, 1]) +
        p[:, 2, 0] * (p[:, 0, 1] - p[:, 1, 1])
    )
    return np.min(areas)

def _generate_initial_points(n: int, seed: int) -> np.ndarray:
    """
    Generates a high-quality initial distribution of points using a Sobol
    sequence, which is then mapped from the unit square to the equilateral triangle.
    """
    sampler = qmc.Sobol(d=2, seed=seed)
    uv_points = sampler.random(n)
    
    u, v = uv_points[:, 0], uv_points[:, 1]
    
    # Fold the unit square sample into a right-angled triangle.
    # This ensures uniform distribution within the triangle domain after mapping.
    mask = u + v > 1
    u[mask], v[mask] = 1 - u[mask], 1 - v[mask]
    
    # Affine map from the right-angled triangle to the target equilateral triangle.
    x = u + 0.5 * v
    y = (_SQRT3 / 2.0) * v
    
    return np.vstack([x, y]).T

def heilbronn_triangle11() -> np.ndarray:
    """
    Constructs an arrangement of 11 points within an equilateral triangle
    to maximize the minimum triangle area formed by any three points,
    using a Simulated Annealing metaheuristic.

    Returns:
        points: np.ndarray of shape (11, 2) with the x,y coordinates of the optimal points.
    """
    # Simulated Annealing parameters, tuned for a balance of solution quality and execution time.
    T_MAX = 0.0005  # Initial temperature, related to raw area scale.
    T_MIN = 1e-8    # Final temperature to stop optimization.
    ALPHA = 0.9998  # Cooling rate.
    MAX_STEPS = 40000 # Total optimization steps.
    MAX_PERTURB_ATTEMPTS = 10 # Attempts to find a valid neighbor point.

    # Set seed for complete reproducibility of the optimization process.
    np.random.seed(_SEED)

    # 1. Initialize State
    current_points = _generate_initial_points(_N_POINTS, _SEED)
    current_score = _calculate_min_area(current_points)
    
    best_points = np.copy(current_points)
    best_score = current_score
    
    temp = T_MAX
    
    # 2. Optimization Loop
    for _ in range(MAX_STEPS):
        if temp <= T_MIN:
            break

        # 3. Generate a Neighbor State by perturbing one point.
        new_points = np.copy(current_points)
        idx_to_move = np.random.randint(0, _N_POINTS)
        
        # Perturbation magnitude scales with temperature, allowing large jumps initially
        # and fine-tuning towards the end.
        perturb_scale = (temp / T_MAX) * 0.05
        
        valid_move = False
        for _ in range(MAX_PERTURB_ATTEMPTS):
            perturbation = np.random.randn(2) * perturb_scale
            candidate_point = new_points[idx_to_move] + perturbation
            
            # Use rejection sampling: ensure the new point is inside the domain.
            if _is_inside_triangle(candidate_point):
                new_points[idx_to_move] = candidate_point
                valid_move = True
                break
        
        if not valid_move:
            temp *= ALPHA
            continue

        # 4. Evaluate the new state.
        new_score = _calculate_min_area(new_points)

        # 5. Metropolis Acceptance Criterion.
        delta_score = new_score - current_score
        if delta_score > 0 or np.random.rand() < math.exp(delta_score / temp):
            # Accept the new state.
            current_points = new_points
            current_score = new_score
            
            # Update the best-found solution if this one is better.
            if current_score > best_score:
                best_score = current_score
                best_points = np.copy(current_points)
        
        # 6. Cool Down.
        temp *= ALPHA

    return best_points

# EVOLVE-BLOCK-END
