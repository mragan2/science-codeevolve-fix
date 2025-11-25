# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize, differential_evolution
from numba import njit # New import for JIT compilation

# Global Constants (for Numba compatibility and consistency)
N_CIRCLES = 26
UNIT_SQUARE_SIDE = 1.0
EPSILON_R = 1e-7 # Minimum radius to avoid numerical issues and ensure positive radii, adjusted for tighter precision

# Numba-jitted function for fast constraint evaluation, adapted from Inspiration 1
@njit(cache=True)
def _evaluate_constraints_numba(params: np.ndarray) -> np.ndarray:
    """
    Evaluates all constraints for a given set of circle parameters.
    Returns an array of constraint values (g(x) >= 0).
    Negative values indicate violations.
    """
    circles = params.reshape(N_CIRCLES, 3)
    
    xs = circles[:, 0]
    ys = circles[:, 1]
    rs = circles[:, 2]

    # Total number of constraints:
    # N_CIRCLES (radii positivity)
    # 4 * N_CIRCLES (containment)
    # N_CIRCLES * (N_CIRCLES - 1) // 2 (non-overlap)
    num_constraints = N_CIRCLES + 4 * N_CIRCLES + N_CIRCLES * (N_CIRCLES - 1) // 2
    constraints = np.empty(num_constraints, dtype=params.dtype)
    constraint_idx = 0

    # 1. Radius positivity constraint (r >= EPSILON_R)
    for i in range(N_CIRCLES):
        constraints[constraint_idx] = rs[i] - EPSILON_R
        constraint_idx += 1

    # 2. Containment constraints (ri <= xi <= 1-ri, ri <= yi <= 1-ri)
    for i in range(N_CIRCLES):
        constraints[constraint_idx] = xs[i] - rs[i] # xi - ri >= 0
        constraint_idx += 1
        constraints[constraint_idx] = UNIT_SQUARE_SIDE - xs[i] - rs[i] # 1 - xi - ri >= 0
        constraint_idx += 1
        constraints[constraint_idx] = ys[i] - rs[i] # yi - ri >= 0
        constraint_idx += 1
        constraints[constraint_idx] = UNIT_SQUARE_SIDE - ys[i] - rs[i] # 1 - yi - ri >= 0
        constraint_idx += 1

    # 3. Non-overlap constraints
    for i in range(N_CIRCLES):
        for j in range(i + 1, N_CIRCLES):
            dx = xs[i] - xs[j]
            dy = ys[i] - ys[j]
            dist_sq = dx*dx + dy*dy
            min_dist_sq = (rs[i] + rs[j])**2
            constraints[constraint_idx] = dist_sq - min_dist_sq # dist_sq - (ri+rj)^2 >= 0
            constraint_idx += 1

    return constraints


# Helper functions for initial guesses, adapted from Inspiration 3
def _generate_hex_guess(rows_circles: list[int], n: int, initial_r: float) -> np.ndarray:
    initial_circles = []
    y_current = initial_r
    x_step = 2 * initial_r
    y_step = np.sqrt(3) * initial_r

    for row_idx, num_circles_in_row in enumerate(rows_circles):
        x_start = initial_r if row_idx % 2 == 0 else initial_r + initial_r
        for col_idx in range(num_circles_in_row):
            if len(initial_circles) >= n: break
            x = x_start + col_idx * x_step
            y = y_current
            # Clip to ensure initial guess is within bounds, considering initial_r
            x = np.clip(x, initial_r, UNIT_SQUARE_SIDE - initial_r)
            y = np.clip(y, initial_r, UNIT_SQUARE_SIDE - initial_r)
            initial_circles.append([x, y, initial_r])
        y_current += y_step
    return np.array(initial_circles).flatten()

def _generate_square_guess(rows_circles: list[int], n: int, initial_r: float) -> np.ndarray:
    initial_circles = []
    y_current = initial_r
    x_step = 2 * initial_r
    y_step = 2 * initial_r

    for row_idx, num_circles_in_row in enumerate(rows_circles):
        total_width_occupied = (num_circles_in_row - 1) * x_step + 2 * initial_r
        x_offset_for_centering = (UNIT_SQUARE_SIDE - total_width_occupied) / 2.0
        x_start_centered = x_offset_for_centering + initial_r if x_offset_for_centering > 1e-9 else initial_r

        for col_idx in range(num_circles_in_row):
            if len(initial_circles) >= n: break
            x = x_start_centered + col_idx * x_step
            y = y_current
            x = np.clip(x, initial_r, UNIT_SQUARE_SIDE - initial_r)
            y = np.clip(y, initial_r, UNIT_SQUARE_SIDE - initial_r)
            initial_circles.append([x, y, initial_r])
        y_current += y_step
    return np.array(initial_circles).flatten()

def _generate_random_guess(n: int, initial_r: float, rng=None) -> np.ndarray:
    if rng is None:
        rng = np.random.default_rng()
    initial_circles = []
    for _ in range(n):
        x = rng.uniform(initial_r, UNIT_SQUARE_SIDE - initial_r)
        y = rng.uniform(initial_r, UNIT_SQUARE_SIDE - initial_r)
        initial_circles.append([x, y, initial_r])
    return np.array(initial_circles).flatten()


def circle_packing26() -> np.ndarray:
    """
    Generates an optimal arrangement of 26 non-overlapping circles in a unit square
    to maximize the sum of their radii, using a hybrid Differential Evolution (global)
    and SLSQP (local) optimization strategy.
    """
    np.random.seed(42)  # For reproducibility
    rng = np.random.default_rng(42) # Dedicated RNG for perturbations

    # Pre-compile numba function by calling it once with dummy data
    _ = _evaluate_constraints_numba(np.zeros(N_CIRCLES * 3, dtype=np.float64))

    # --- 1. Objective Function ---
    def objective(params: np.ndarray) -> float:
        return -np.sum(params[2::3])

    # --- 2. Constraint Function ---
    # The `_evaluate_constraints_numba` directly uses global N_CIRCLES and EPSILON_R.
    # We pass it as the constraint function.
    cons = ({'type': 'ineq', 'fun': _evaluate_constraints_numba})

    # --- 3. Bounds for variables [x, y, r] ---
    bounds = []
    for _ in range(N_CIRCLES):
        bounds.extend([(0.0, UNIT_SQUARE_SIDE), (0.0, UNIT_SQUARE_SIDE), (EPSILON_R, UNIT_SQUARE_SIDE / 2.0)])

    # --- 4. Initial Guess for Fallback / DE Warm-start ---
    # A robust, simple initial guess for potential SLSQP fallback if DE fails,
    # or as a custom initial population for DE.
    # Using a hexagonal pattern as it's generally good for circle packing.
    initial_x0_fallback = _generate_hex_guess([5, 5, 5, 5, 6], N_CIRCLES, initial_r=0.08)
    # Ensure initial radii and positions are within their respective bounds after generation
    initial_x0_fallback[2::3] = np.clip(initial_x0_fallback[2::3], EPSILON_R, UNIT_SQUARE_SIDE / 2.0)
    initial_x0_fallback[0::3] = np.clip(initial_x0_fallback[0::3], initial_x0_fallback[2::3], UNIT_SQUARE_SIDE - initial_x0_fallback[2::3])
    initial_x0_fallback[1::3] = np.clip(initial_x0_fallback[1::3], initial_x0_fallback[2::3], UNIT_SQUARE_SIDE - initial_x0_fallback[2::3])

    # --- 5. Phase 1: Global Optimization with Differential Evolution ---
    print("Starting global optimization with Differential Evolution...")
    de_result_x = initial_x0_fallback.copy() # Default to fallback if DE fails
    de_success = False
    try:
        de_res = differential_evolution(
            objective,
            bounds,
            constraints=cons, # Pass constraints directly as dictionary
            maxiter=2500,      # Increased maxiter from Insp3's 1500 for deeper global search
            popsize=40,        # Increased population size from Insp3's 20 for better exploration
            seed=42,           # For reproducibility
            disp=False,        # Set to True for verbose output
            polish=False,      # SLSQP will do the final polishing
            workers=-1         # Use all available CPU cores for parallelization
        )
        
        if de_res.success:
            de_result_x = de_res.x
            de_success = True
            print(f"Differential Evolution finished successfully. Best sum of radii found: {-de_res.fun:.6f}")
        else:
            print(f"Differential Evolution finished with message: {de_res.message}. Using initial fallback guess.")
    except Exception as e:
        print(f"An error occurred during Differential Evolution: {e}. Using initial fallback guess.")

    # --- 6. Phase 2: Local Refinement with SLSQP ---
    print("Starting local refinement with SLSQP...")
    best_result = None
    try:
        # Ensure the starting point for SLSQP is within bounds
        # Clip coordinates and radii to ensure validity before perturbation
        de_result_x[0::3] = np.clip(de_result_x[0::3], 0.0, UNIT_SQUARE_SIDE)
        de_result_x[1::3] = np.clip(de_result_x[1::3], 0.0, UNIT_SQUARE_SIDE)
        de_result_x[2::3] = np.clip(de_result_x[2::3], EPSILON_R, UNIT_SQUARE_SIDE / 2.0)

        # Apply a small perturbation to the DE result before SLSQP to help escape very shallow local minima
        # This is a common trick in hybrid optimization.
        perturb_scale_xy = 0.005 # Moderate perturbation for positions
        perturb_scale_r = 0.001  # Smaller perturbation for radii
        
        de_result_x[0::3] += rng.uniform(-perturb_scale_xy, perturb_scale_xy, N_CIRCLES)
        de_result_x[1::3] += rng.uniform(-perturb_scale_xy, perturb_scale_xy, N_CIRCLES)
        de_result_x[2::3] += rng.uniform(-perturb_scale_r, perturb_scale_r, N_CIRCLES)
        
        # Clip again after perturbation to ensure adherence to bounds, respecting radii for positions
        de_result_x[2::3] = np.clip(de_result_x[2::3], EPSILON_R, UNIT_SQUARE_SIDE / 2.0)
        de_result_x[0::3] = np.clip(de_result_x[0::3], de_result_x[2::3], UNIT_SQUARE_SIDE - de_result_x[2::3])
        de_result_x[1::3] = np.clip(de_result_x[1::3], de_result_x[2::3], UNIT_SQUARE_SIDE - de_result_x[2::3])

        slsqp_res = minimize(
            objective,
            de_result_x,
            method='SLSQP',
            bounds=bounds,
            constraints=cons,
            options={'maxiter': 30000, 'ftol': 1e-13, 'disp': False, 'eps': 1e-8} # Further increased maxiter and tightened ftol for higher precision
        )
        best_result = slsqp_res

        if not best_result.success:
            print(f"SLSQP refinement finished with message: {best_result.message}")
        else:
            print(f"SLSQP refinement successful. Final sum of radii: {-best_result.fun:.6f}")
            
    except Exception as e:
        print(f"An error occurred during SLSQP refinement: {e}. Using the result from DE if available, otherwise fallback.")
        # If SLSQP fails, try to use DE's result if it was successful.
        if de_success:
            best_result = de_res # Use the raw DE result if SLSQP failed
        else:
            # Last resort, if both fail, initialize with the fallback guess and run a quick SLSQP
            print("Running final basic SLSQP fallback.")
            best_result = minimize(
                objective, initial_x0_fallback, method='SLSQP', bounds=bounds,
                constraints=cons, options={'maxiter': 5000, 'ftol': 1e-9, 'disp': False}
            )
            if not best_result.success:
                print(f"Critical: Final fallback SLSQP also failed: {best_result.message}")

    # --- 7. Final Processing ---
    if best_result is None or not best_result.success:
        print("Warning: Optimization did not converge successfully. Returning initial fallback guess.")
        optimized_circles = initial_x0_fallback.reshape((N_CIRCLES, 3))
    else:
        optimized_circles = best_result.x.reshape((N_CIRCLES, 3))
    
    # Final validation: Ensure all radii are strictly positive,
    # clipping to the minimum allowed radius from bounds.
    optimized_circles[:, 2] = np.maximum(optimized_circles[:, 2], EPSILON_R)

    return optimized_circles


# EVOLVE-BLOCK-END
