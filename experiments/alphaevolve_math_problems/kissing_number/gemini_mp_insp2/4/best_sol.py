# EVOLVE-BLOCK-START
import numpy as np
from itertools import permutations, product
import time
from numba import jit

# Fixed random seed for reproducibility if any stochasticity is introduced later
np.random.seed(42)

def generate_integer_points(d: int, N: int) -> list[np.ndarray]:
    """
    Generates all unique integer points p in Z^d such that ||p||_2^2 = N.
    Uses a recursive backtracking approach to find canonical forms (non-negative, non-increasing coordinates),
    then applies permutations and sign changes.
    """
    canonical_forms = set()

    def find_sums_of_squares_recursive(current_sum_sq, k, current_coords_list):
        # Base case: sum of squares equals N
        if current_sum_sq == N:
            # Pad with zeros if necessary to reach 'd' dimensions
            padded_coords = tuple(current_coords_list + [0] * (d - len(current_coords_list)))
            if len(padded_coords) == d: # This check should always pass if logic is correct
                canonical_forms.add(padded_coords)
            return

        # Pruning conditions: if dimensions exceed 'd' or sum exceeds 'N'
        if k == d or current_sum_sq > N:
            return

        # Determine maximum value for the current coordinate to maintain non-increasing order
        # and ensure the sum of squares does not exceed N.
        max_val_for_coord = int(np.sqrt(N - current_sum_sq))
        if current_coords_list: # If not the first coordinate, respect non-increasing order
            max_val_for_coord = min(max_val_for_coord, current_coords_list[-1])

        # Iterate downwards for values to build canonical forms efficiently
        for val in range(max_val_for_coord, -1, -1):
            find_sums_of_squares_recursive(current_sum_sq + val*val, k + 1, current_coords_list + [val])

    find_sums_of_squares_recursive(0, 0, [])

    all_points_tuples = set()
    for canonical_tuple in canonical_forms:
        # Generate all unique permutations of the canonical tuple
        for p_tuple in set(permutations(canonical_tuple)):
            # Identify indices of non-zero elements to apply sign changes
            non_zero_indices = [i for i, x in enumerate(p_tuple) if x != 0]
            
            # Generate all possible sign combinations for non-zero elements
            for signs in product([-1, 1], repeat=len(non_zero_indices)):
                point_list = list(p_tuple)
                for i, sign_val in zip(non_zero_indices, signs):
                    point_list[i] *= sign_val
                all_points_tuples.add(tuple(point_list))
    
    return [np.array(p, dtype=np.int64) for p in all_points_tuples]

@jit(nopython=True)
def _greedy_select_points_numba(candidate_array: np.ndarray, N_half: float) -> np.ndarray:
    """
    Numba-optimized inner function for greedy selection.
    Accepts candidate points as a 2D numpy array and returns selected points as a 2D numpy array.
    The candidate_array is assumed to be sorted lexicographically before being passed.
    """
    num_candidates = candidate_array.shape[0]
    if num_candidates == 0:
        return np.empty((0, candidate_array.shape[1]), dtype=np.int64)

    # Use a list to store indices of selected points, which is then converted to an array
    selected_indices = []
    
    for i in range(num_candidates):
        p_candidate = candidate_array[i]
        is_valid = True
        
        # Check dot product constraint against all currently selected points
        for selected_idx in selected_indices:
            p_selected = candidate_array[selected_idx]
            # Numba's np.dot can have issues with 1D int64 arrays,
            # using element-wise multiplication and sum as a robust alternative for dot product.
            if np.sum(p_candidate * p_selected) > N_half:
                is_valid = False
                break
        
        if is_valid:
            selected_indices.append(i)
            
    # Construct the result array from the selected indices
    if not selected_indices:
        return np.empty((0, candidate_array.shape[1]), dtype=np.int64)
    
    result_array = np.empty((len(selected_indices), candidate_array.shape[1]), dtype=np.int64)
    for j, idx in enumerate(selected_indices):
        result_array[j] = candidate_array[idx]
        
    return result_array

def _run_greedy_selection_single_pass(candidate_array: np.ndarray, N_half: float) -> np.ndarray:
    """
    Wrapper for the Numba-optimized greedy selection.
    Expects a 2D numpy array of candidate points (already sorted or shuffled).
    """
    if candidate_array.shape[0] == 0:
        return np.empty((0, candidate_array.shape[1]), dtype=np.int64)
    
    return _greedy_select_points_numba(candidate_array, N_half)


def greedy_select_points(candidate_points_list: list[np.ndarray], N: int, num_runs: int = 5) -> np.ndarray:
    """
    Performs multiple runs of greedy selection with shuffled candidate points
    and returns the largest set found.
    
    Args:
        candidate_points_list: List of candidate points (np.ndarray).
        N: The squared L2 norm of points.
        num_runs: Number of greedy selection runs with different shuffles.
                  Defaults to 5 for a balance between exploration and performance.
    
    Returns:
        np.ndarray: The largest set of selected points found across all runs.
    """
    if not candidate_points_list:
        return np.empty((0, 11), dtype=np.int64) # Assuming d=11 from problem context

    best_selected_points = np.empty((0, 11), dtype=np.int64)
    best_num_points = 0
    
    # Convert list of arrays to a single 2D numpy array once for efficiency
    base_candidate_array = np.array(candidate_points_list, dtype=np.int64)
    N_half = N / 2.0

    # Perform the first run with lexicographical sort (original deterministic behavior).
    # This ensures a reproducible baseline and often provides a good starting point.
    # Create a sorted copy for the first run.
    sorted_candidate_array = np.array(sorted(candidate_points_list, key=lambda p: tuple(p)), dtype=np.int64)
    selected_points_array = _run_greedy_selection_single_pass(sorted_candidate_array, N_half)
    
    best_num_points = selected_points_array.shape[0]
    best_selected_points = selected_points_array.copy() # Use copy to store the best result

    # Subsequent runs with shuffled arrays to explore different greedy paths
    # Only run if num_runs > 1, as the first run already accounts for one.
    for _ in range(num_runs - 1): 
        current_candidate_array = base_candidate_array.copy()
        np.random.shuffle(current_candidate_array) # Shuffle in-place for randomized greedy

        selected_points_array = _run_greedy_selection_single_pass(current_candidate_array, N_half)
        
        current_num_points = selected_points_array.shape[0]
        if current_num_points > best_num_points:
            best_num_points = current_num_points
            best_selected_points = selected_points_array
            
    return best_selected_points


def kissing_number11()->np.ndarray:
    """
    Constructs a collection of 11-dimensional points with integral coordinates such that their maximum norm is smaller than their minimum pairwise distance, aiming to maximize the number of points. 

    Returns:
        points: np.ndarray of shape (num_points,11)
    """
    d = 11
    
    best_num_points = 0
    best_points = np.array([]).reshape(0, d) # Initialize with an empty 2D array
    
    # Iterate over plausible integer values for N (squared L2 norm)
    # The search range for N is chosen to balance computational time and likelihood of finding
    # a large set. Higher N values generate significantly more candidate points.
    # N=1: 22 points
    # N=2: 220 points (all candidates satisfy the condition)
    # N=3: 1320 candidates. Greedy selection will reduce this.
    # N=4: 5302 candidates. Greedy selection will reduce this.
    # N=5: 15004 candidates. Greedy selection will reduce this.
    # N=6: 33528 candidates. Greedy selection will reduce this.
    for N_val in range(1, 7): # Test N=1, 2, 3, 4, 5, 6
        candidate_points_list = generate_integer_points(d, N_val)
        selected_points_array = greedy_select_points(candidate_points_list, N_val)
        
        current_num_points = selected_points_array.shape[0]
        if current_num_points > best_num_points:
            best_num_points = current_num_points
            best_points = selected_points_array
            
    return best_points

# EVOLVE-BLOCK-END