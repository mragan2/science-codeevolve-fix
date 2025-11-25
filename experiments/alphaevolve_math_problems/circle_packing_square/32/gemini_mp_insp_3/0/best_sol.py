# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize, Bounds
from joblib import Parallel, delayed

# --- Global Constants ---
N_CIRCLES = 32
SQUARE_SIZE = 1.0
_UPPER_TRIANGLE_INDICES = np.triu_indices(N_CIRCLES, k=1)
_EPSILON = 1e-8 # Small value for numerical stability and bounds

# --- Helper Functions ---
def _unpack_params(params: np.ndarray) -> np.ndarray:
    """Unpacks the 1D parameter array into (N, 3) circles array."""
    return params.reshape((N_CIRCLES, 3))

def _objective(params: np.ndarray) -> float:
    """
    Objective function to minimize: negative sum of radii.
    Includes a penalty for non-positive radii.
    """
    radii = params[2::3]
    if np.any(radii <= _EPSILON):
        return np.inf # Heavily penalize non-positive radii
    return -np.sum(radii)

def _get_containment_constraints_values(circles: np.ndarray) -> np.ndarray:
    """Returns an array of values for containment constraints (must be >= 0)."""
    x, y, r = circles[:, 0], circles[:, 1], circles[:, 2]
    return np.concatenate([
        x - r,
        SQUARE_SIZE - r - x,
        y - r,
        SQUARE_SIZE - r - y
    ])

def _get_overlap_constraints_values(circles: np.ndarray) -> np.ndarray:
    """Returns a vectorized array of values for non-overlap constraints (must be >= 0)."""
    i, j = _UPPER_TRIANGLE_INDICES
    pos_i, pos_j = circles[i, :2], circles[j, :2]
    radii_i, radii_j = circles[i, 2], circles[j, 2]
    dist_sq = np.sum((pos_i - pos_j)**2, axis=1)
    radii_sum_sq = (radii_i + radii_j)**2
    return dist_sq - radii_sum_sq

def _constraints(params: np.ndarray) -> np.ndarray:
    """Consolidated constraint function combining containment and non-overlap."""
    circles = _unpack_params(params)
    containment = _get_containment_constraints_values(circles)
    overlap = _get_overlap_constraints_values(circles)
    return np.concatenate([containment, overlap])

# Helper function for parallel execution, tuned for Stage 1 exploration.
def _run_optimization_task(initial_guess, objective_func, bounds, constraints_list):
    """
    Worker function for a single optimization run (Stage 1: Exploration).
    Parameters are tuned for broad exploration.
    """
    res = minimize(objective_func, initial_guess, method='SLSQP', bounds=bounds,
                   constraints=constraints_list,
                   options={'disp': False, 'maxiter': 5000, 'ftol': 1e-9}) # Maxiter increased to 5000

    # Accept success or max iterations reached, as both can yield good starting points for refinement.
    if res.success or res.status == 2:
        return res.x, -res.fun
    else:
        return None, -np.inf


def circle_packing32() -> np.ndarray:
    """
    Places 32 non-overlapping circles in the unit square to maximize the sum of radii.
    This version uses a two-stage optimization process:
    1. Parallel exploration from diverse, heuristically-generated starting points.
    2. A single, high-precision refinement of the best candidate from stage 1.
    """
    # --- 1. Bounds and Constraints ---
    bounds = Bounds([0.0, 0.0, _EPSILON] * N_CIRCLES, [SQUARE_SIZE, SQUARE_SIZE, SQUARE_SIZE / 2.0] * N_CIRCLES)
    constraints = [{'type': 'ineq', 'fun': _constraints}]

    # --- 2. Stage 1: Parallel Exploration ---
    num_restarts = 96 # Reduced number of restarts for better time performance
    rng = np.random.RandomState(42) # Fixed seed for reproducibility
    
    initial_guesses = []
    for i in range(num_restarts):
        x0_circles = np.zeros((N_CIRCLES, 3))
        if i % 3 == 0: # Strategy 1: Rectangular grid-based initial guess (4x8)
            rows, cols = 4, 8
            grid_radii = min(SQUARE_SIZE / cols, SQUARE_SIZE / rows) * 0.9 / 2 # Adjusted for radius
            grid_x, grid_y = np.meshgrid(
                np.linspace(grid_radii, SQUARE_SIZE - grid_radii, cols), # Use linspace for centers
                np.linspace(grid_radii, SQUARE_SIZE - grid_radii, rows)
            )
            x0_circles[:, 0], x0_circles[:, 1], x0_circles[:, 2] = grid_x.flatten(), grid_y.flatten(), grid_radii
            x0_circles[:, :2] += rng.uniform(-0.02, 0.02, size=(N_CIRCLES, 2))
            x0_circles[:, 2] *= rng.uniform(0.8, 1.2, size=N_CIRCLES)
        elif i % 3 == 1: # Strategy 2: Hexagonal grid approximation initial guess
            hex_rows, circles_per_row, r_hex_approx = 6, [5, 6, 5, 6, 5, 5], 0.045
            current_idx = 0
            for r_idx in range(hex_rows):
                num_circles_in_row = circles_per_row[r_idx]
                cy = r_hex_approx + r_idx * r_hex_approx * np.sqrt(3)
                x_offset = 0 if r_idx % 2 == 0 else r_hex_approx
                for c_idx in range(num_circles_in_row):
                    if current_idx < N_CIRCLES:
                        cx = r_hex_approx + x_offset + c_idx * 2 * r_hex_approx
                        x0_circles[current_idx] = [cx, cy, r_hex_approx]
                        current_idx += 1
            x0_circles[:, :2] += rng.uniform(-0.015, 0.015, size=(N_CIRCLES, 2))
            x0_circles[:, 2] *= rng.uniform(0.9, 1.1, size=N_CIRCLES)
        else: # Strategy 3: Fully random initial guess
            x0_circles[:, 0] = rng.rand(N_CIRCLES)
            x0_circles[:, 1] = rng.rand(N_CIRCLES)
            x0_circles[:, 2] = rng.uniform(0.008, 0.025, N_CIRCLES)

        # Ensure initial guess respects the defined bounds
        x0_circles[:, 0] = np.clip(x0_circles[:, 0], _EPSILON, SQUARE_SIZE - _EPSILON)
        x0_circles[:, 1] = np.clip(x0_circles[:, 1], _EPSILON, SQUARE_SIZE - _EPSILON)
        x0_circles[:, 2] = np.clip(x0_circles[:, 2], _EPSILON, SQUARE_SIZE / 2.0)
        initial_guesses.append(x0_circles.flatten())

    results = Parallel(n_jobs=-1)(
        delayed(_run_optimization_task)(x0, _objective, bounds, constraints) for x0 in initial_guesses
    )

    # --- 3. Process Stage 1 Results ---
    best_sum_radii = -np.inf
    best_solution_flat = None
    for x_sol, sum_radii in results:
        if x_sol is not None and sum_radii > best_sum_radii:
            best_sum_radii = sum_radii
            best_solution_flat = x_sol
    
    if best_solution_flat is None:
        return np.zeros((N_CIRCLES, 3))

    # --- 4. Stage 2: High-Precision Refinement ---
    refine_res = minimize(_objective, best_solution_flat, method='SLSQP', bounds=bounds,
                            constraints=constraints,
                            options={'disp': False, 'maxiter': 10000, 'ftol': 1e-11}) # High maxiter and tight ftol
    
    # Only update if refinement improved the best sum of radii
    if (refine_res.success or refine_res.status == 2) and -refine_res.fun > best_sum_radii:
        best_solution_flat = refine_res.x

    final_circles = _unpack_params(best_solution_flat)
    
    # --- 5. Final Clipping for Robustness ---
    # Ensure radii are within valid range
    final_circles[:, 2] = np.clip(final_circles[:, 2], _EPSILON, SQUARE_SIZE / 2.0)
    # Ensure centers are within valid range based on their radii
    r = final_circles[:, 2]
    final_circles[:, 0] = np.clip(final_circles[:, 0], r, SQUARE_SIZE - r)
    final_circles[:, 1] = np.clip(final_circles[:, 1], r, SQUARE_SIZE - r)
    
    return final_circles


# EVOLVE-BLOCK-END
