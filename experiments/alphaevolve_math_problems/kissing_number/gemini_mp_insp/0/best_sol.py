# EVOLVE-BLOCK-START
import numpy as np
import itertools
from scipy.spatial import KDTree
import time
import random

# Set a fixed random seed for reproducibility
np.random.seed(42)
random.seed(42)

D = 11 # Dimension

def _get_squared_norm(point):
    """Calculates the squared L2 norm of a single point."""
    return np.sum(point**2)

def _get_pairwise_squared_distances(points_array):
    """
    Calculates all pairwise squared L2 distances between points in a set.
    Returns np.inf for self-distances.
    """
    if len(points_array) < 2:
        return np.array([np.inf])
    
    # Compute all pairwise squared L2 distances using broadcasting
    diff = points_array[:, np.newaxis, :] - points_array[np.newaxis, :, :]
    # Ensure distances_sq is float dtype before filling with np.inf
    distances_sq = np.sum(diff**2, axis=2).astype(np.float64)
    
    # Set diagonal to infinity to avoid self-distance
    np.fill_diagonal(distances_sq, np.inf) # This now works as distances_sq is float
    return distances_sq

def check_constraint(points_array):
    """
    Checks if a given set of points satisfies the geometric constraint:
    max_i ||p_i||_2 <= min_{i != j} ||p_i - p_j||_2
    Equivalent to: max_i ||p_i||_2^2 <= min_{i != j} ||p_i - p_j||_2^2
    
    Returns: (is_valid, N_sq_max, D_sq_min)
    """
    if len(points_array) == 0:
        return True, 0, np.inf # No points, trivially true
    if len(points_array) == 1:
        # One point: N_sq_max is its norm, D_sq_min is inf (no other points)
        return True, _get_squared_norm(points_array[0]), np.inf 

    N_sq_max = np.max(np.sum(points_array**2, axis=1))
    distances_sq = _get_pairwise_squared_distances(points_array)
    D_sq_min = np.min(distances_sq)

    return N_sq_max <= D_sq_min, N_sq_max, D_sq_min

def generate_symmetric_points(base_vector):
    """
    Generates all unique points by permuting coordinates and changing signs of a base vector.
    Assumes base_vector has non-negative integer coordinates.
    """
    base_vector = np.array(base_vector, dtype=np.int64)
    unique_points = set()

    for p_tuple in set(itertools.permutations(base_vector)):
        p = np.array(p_tuple, dtype=np.int64)
        for signs in itertools.product([-1, 1], repeat=D):
            signed_p = p * np.array(signs, dtype=np.int64)
            unique_points.add(tuple(signed_p))
    
    # Special handling for zero vector to ensure (0,0,...,0) is unique
    if np.all(base_vector == 0):
        return [np.zeros(D, dtype=np.int64)]

    return [np.array(pt, dtype=np.int64) for pt in unique_points]

def _generate_base_vectors_recursive(target_sq_norm, current_sum_sq, current_vector_parts, remaining_dims, max_val_for_next_coord, results_list):
    """
    Recursively finds all unique base vectors (non-negative, sorted non-increasingly)
    that sum to target_sq_norm when their coordinates are squared, in 'remaining_dims'.
    """
    if remaining_dims == 0:
        if current_sum_sq == target_sq_norm:
            # Pad with zeros to D dimensions
            full_vector = tuple(current_vector_parts + [0] * (D - len(current_vector_parts)))
            results_list.append(full_vector)
        return

    if current_sum_sq > target_sq_norm:
        return

    # Try adding a new coordinate value
    # Start from max_val_for_next_coord down to 0 to ensure non-increasing order
    for val in range(max_val_for_next_coord, -1, -1):
        if current_sum_sq + val**2 <= target_sq_norm:
            _generate_base_vectors_recursive(target_sq_norm, current_sum_sq + val**2, current_vector_parts + [val], remaining_dims - 1, val, results_list)

def get_candidates_for_specific_norm(target_sq_norm):
    """
    Generates a pool of candidate integer points with squared L2 norm *exactly* target_sq_norm.
    Does not include (0,0,...,0) if target_sq_norm > 0 (as per problem analysis, origin is excluded for non-trivial sets).
    """
    if target_sq_norm == 0:
        return [] # Origin is excluded for sets with > 1 point.

    all_unique_points_set = set()
    max_coord_val = int(np.sqrt(target_sq_norm))
    
    base_vectors_list = []
    _generate_base_vectors_recursive(target_sq_norm, 0, [], D, max_coord_val, base_vectors_list)
    
    for base_vec_tuple in base_vectors_list:
        base_vec_array = np.array(base_vec_tuple, dtype=np.int64)
        symmetric_points = generate_symmetric_points(base_vec_array)
        all_unique_points_set.update(tuple(p) for p in symmetric_points)

    sorted_candidates = sorted([np.array(p, dtype=np.int64) for p in all_unique_points_set], key=lambda x: (tuple(x)))
    return sorted_candidates

def get_initial_greedy_set_fixed_norm_and_dist(candidate_points_of_fixed_norm, target_R_sq):
    """
    Greedily constructs an initial set of points where:
    1. All points have squared norm == target_R_sq.
    2. All pairwise squared distances >= target_R_sq.
    """
    current_set = set()
    current_points_array = np.empty((0, D), dtype=np.int64)
    
    kdtree = None

    for p_new in candidate_points_of_fixed_norm:
        if tuple(p_new) in current_set:
            continue

        is_valid_to_add = True
        if len(current_points_array) > 0:
            # KDTree.query returns distances, not squared. Square them for comparison.
            distances, _ = kdtree.query(p_new, k=min(len(current_points_array), 100)) # Query top 100 neighbors for speed
            distances_sq = distances**2
            
            if isinstance(distances_sq, np.ndarray):
                min_dist_to_new_sq = np.min(distances_sq)
            else: # Only one neighbor
                min_dist_to_new_sq = distances_sq
            
            if min_dist_to_new_sq < target_R_sq:
                is_valid_to_add = False
        
        if is_valid_to_add:
            current_set.add(tuple(p_new))
            current_points_array = np.vstack([current_points_array, p_new])
            kdtree = KDTree(current_points_array) # Rebuild KDTree after addition
            
    return current_points_array

def simulated_annealing_refinement_fixed_norm_and_dist(initial_points, candidate_pool_for_norm, target_R_sq, max_iterations=100000, initial_temp=10.0, cooling_rate=0.999):
    """
    Refines the set of points using Simulated Annealing, maintaining:
    1. All points have squared norm == target_R_sq.
    2. All pairwise squared distances >= target_R_sq.
    """
    current_points_list = [tuple(p) for p in initial_points]
    current_points_set = set(current_points_list)
    current_points_array = np.array(current_points_list, dtype=np.int64)

    # Energy function: we want to maximize len(points) while strictly maintaining R_sq, R_sq conditions
    def calculate_energy(points_arr_len, is_valid):
        if not is_valid:
            return np.inf # Heavily penalize invalid states
        return -points_arr_len # Maximize point count

    # Initial validation of the input `initial_points`
    is_valid_initial, N_sq_max_initial, D_sq_min_initial = check_constraint(current_points_array)
    if not is_valid_initial or (len(current_points_array) > 0 and (N_sq_max_initial != target_R_sq or D_sq_min_initial < target_R_sq)):
        # If initial points are invalid or don't meet strict criteria, SA cannot proceed meaningfully to improve.
        # It's better to return the initial, potentially empty, set.
        print(f"  SA Warning: Initial points for R_sq={target_R_sq} are invalid or don't match target_R_sq conditions. N_sq_max={N_sq_max_initial}, D_sq_min={D_sq_min_initial}. Returning initial points.")
        return initial_points

    current_energy = calculate_energy(len(current_points_array), True) # Initial state is assumed valid
    
    best_points_array = current_points_array.copy()
    best_energy = current_energy
    
    temp = initial_temp

    # Candidate pool for adding points: points with `target_R_sq` norm not currently in `current_points_set`
    # This list will be dynamically modified by SA moves.
    add_candidate_pool_list = [p for p in candidate_pool_for_norm if tuple(p) not in current_points_set]
    
    kdtree = KDTree(current_points_array) if len(current_points_array) > 0 else None

    for iteration in range(max_iterations):
        if iteration % (max_iterations // 10) == 0:
            print(f"  SA Iter {iteration}/{max_iterations}, Temp: {temp:.4f}, Current points: {len(current_points_array)}, Best points: {len(best_points_array)}")

        # Choose a move type: add a new point, remove an existing point, or swap an existing point with a new candidate
        move_type = random.choices(['add', 'remove', 'swap'], weights=[0.45, 0.2, 0.35], k=1)[0]
        
        # Variables to track potential changes for add_candidate_pool_list management
        p_added_or_swapped = None
        p_removed_or_swapped = None
        idx_taken_from_pool = -1 # Index in add_candidate_pool_list if a point was selected from it

        next_points_array = current_points_array.copy()
        next_points_set = set(current_points_list) # Rebuild set for clarity
        proposed_move_valid = False

        if move_type == 'add' and len(add_candidate_pool_list) > 0:
            idx_candidate = random.randint(0, len(add_candidate_pool_list) - 1)
            p_add_candidate = add_candidate_pool_list[idx_candidate]

            is_valid_dist = True
            if len(next_points_array) > 0:
                # Query for nearest neighbors to p_add_candidate
                distances, _ = kdtree.query(p_add_candidate, k=min(len(next_points_array), 100)) # Query top 100 neighbors
                min_dist_to_add_sq = np.min(distances**2) if isinstance(distances, np.ndarray) else distances**2
                if min_dist_to_add_sq < target_R_sq:
                    is_valid_dist = False
            
            if is_valid_dist:
                next_points_array = np.vstack([next_points_array, p_add_candidate])
                next_points_set.add(tuple(p_add_candidate))
                p_added_or_swapped = p_add_candidate
                idx_taken_from_pool = idx_candidate
                proposed_move_valid = True

        elif move_type == 'remove' and len(current_points_array) > 1: # Must have at least one point remaining
            idx_remove = random.randint(0, len(current_points_array) - 1)
            p_removed_actual = current_points_array[idx_remove]
            
            next_points_array = np.delete(current_points_array, idx_remove, axis=0)
            next_points_set.remove(tuple(p_removed_actual))
            p_removed_or_swapped = p_removed_actual
            proposed_move_valid = True

        elif move_type == 'swap' and len(current_points_array) > 0 and len(add_candidate_pool_list) > 0:
            idx_old = random.randint(0, len(current_points_array) - 1)
            p_old_actual = current_points_array[idx_old]
            
            idx_new_candidate = random.randint(0, len(add_candidate_pool_list) - 1)
            p_new_candidate = add_candidate_pool_list[idx_new_candidate]

            if tuple(p_new_candidate) == tuple(p_old_actual): continue # No actual change

            # Check if p_new_candidate can replace p_old_actual without violating constraints
            temp_points_array = np.delete(current_points_array, idx_old, axis=0)
            
            is_valid_swap_dist = True
            if len(temp_points_array) > 0:
                temp_kdtree = KDTree(temp_points_array) # Rebuild temporary KDTree for check
                distances, _ = temp_kdtree.query(p_new_candidate, k=min(len(temp_points_array), 100)) # Query top 100 neighbors
                min_dist_to_new_sq = np.min(distances**2) if isinstance(distances, np.ndarray) else distances**2
                if min_dist_to_new_sq < target_R_sq:
                    is_valid_swap_dist = False
            
            if is_valid_swap_dist:
                next_points_array = np.vstack([temp_points_array, p_new_candidate])
                next_points_set.remove(tuple(p_old_actual))
                next_points_set.add(tuple(p_new_candidate))
                
                p_added_or_swapped = p_new_candidate
                p_removed_or_swapped = p_old_actual
                idx_taken_from_pool = idx_new_candidate
                proposed_move_valid = True
            
        if proposed_move_valid:
            next_energy = calculate_energy(len(next_points_array), True) # Assume valid if checks passed

            if next_energy < current_energy or (current_energy != np.inf and random.random() < np.exp((current_energy - next_energy) / temp)):
                # Accept the move
                current_points_array = next_points_array
                current_points_list = list(next_points_set) # Update list from set
                current_energy = next_energy
                
                # Update add_candidate_pool_list based on ACCEPTED move
                if move_type == 'add':
                    add_candidate_pool_list.pop(idx_taken_from_pool)
                elif move_type == 'remove':
                    add_candidate_pool_list.append(p_removed_or_swapped)
                elif move_type == 'swap':
                    add_candidate_pool_list.pop(idx_taken_from_pool)
                    add_candidate_pool_list.append(p_removed_or_swapped)
                
                kdtree = KDTree(current_points_array) if len(current_points_array) > 0 else None

                if current_energy < best_energy:
                    best_energy = current_energy
                    best_points_array = current_points_array.copy()
            # If move is rejected, `add_candidate_pool_list` remains unchanged as points were only picked temporarily.
            # No `else` block needed for `add_candidate_pool_list` management here.

        temp *= cooling_rate
        if temp < 1e-6: # Minimum temperature
            temp = 1e-6

    # Final check of the best_points_array
    is_valid_best, N_sq_max_final, D_sq_min_final = check_constraint(best_points_array)
    # Ensure the returned set strictly adheres to the fixed norm and distance criteria
    if not is_valid_best or (len(best_points_array) > 0 and (N_sq_max_final != target_R_sq or D_sq_min_final < target_R_sq)):
        print(f"  SA Warning: Best points found by SA for R_sq={target_R_sq} are not strictly valid or don't meet N_sq_max/D_sq_min. Returning initial greedy points instead.")
        return initial_points # Fallback if SA somehow drifted to an invalid state
    
    return best_points_array


def kissing_number11()->np.ndarray:
    """
    Constructs a collection of 11-dimensional points with integral coordinates such that their maximum norm
    is smaller than their minimum pairwise distance, aiming to maximize the number of points.
    """
    start_time = time.time()
    
    best_overall_points = np.empty((0, D), dtype=np.int64)
    max_num_points = 0

    # Iterate over possible target_R_sq values for N_sq_max == D_sq_min == R_sq
    # Based on analysis, R_sq=4 yields 5280 points. R_sq=5 might yield more.
    max_R_sq_to_search = 5 # Extend to 6 or 7 if time permits, but 5 is a good balance for time.

    # SA parameters - tuned for a balance of exploration and speed
    sa_max_iterations = 200000 # Increased for better search
    sa_initial_temp = 50.0     # Higher temp for more exploration
    sa_cooling_rate = 0.9999   # Slower cooling to allow more steps at higher temps

    for target_R_sq in range(1, max_R_sq_to_search + 1):
        print(f"\n--- Searching for solutions with N_sq_max = D_sq_min = {target_R_sq} ---")
        
        # 1. Generate candidate points with squared norm exactly target_R_sq
        candidate_pool_for_norm = get_candidates_for_specific_norm(target_R_sq)
        print(f"  Generated {len(candidate_pool_for_norm)} candidates for R_sq={target_R_sq}.")

        if len(candidate_pool_for_norm) == 0:
            print(f"  No candidates found for R_sq={target_R_sq}. Skipping.")
            continue

        # 2. Get initial greedy set
        initial_greedy_set = get_initial_greedy_set_fixed_norm_and_dist(candidate_pool_for_norm, target_R_sq)
        print(f"  Initial greedy set for R_sq={target_R_sq} has {len(initial_greedy_set)} points.")
        
        # Check if the greedy set itself is a new best
        if len(initial_greedy_set) > max_num_points:
            is_valid_greedy, N_sq_max_greedy, D_sq_min_greedy = check_constraint(initial_greedy_set)
            if is_valid_greedy and N_sq_max_greedy == target_R_sq and D_sq_min_greedy >= target_R_sq:
                 max_num_points = len(initial_greedy_set)
                 best_overall_points = initial_greedy_set.copy()
                 print(f"  New best found from greedy: {max_num_points} points for R_sq={target_R_sq}.")

        # 3. Refine using Simulated Annealing
        if len(initial_greedy_set) > 0: # Only refine if there's a starting set
            print(f"  Starting SA refinement for R_sq={target_R_sq}...")
            sa_refined_points = simulated_annealing_refinement_fixed_norm_and_dist(
                initial_greedy_set, 
                candidate_pool_for_norm, 
                target_R_sq,
                max_iterations=sa_max_iterations,
                initial_temp=sa_initial_temp,
                cooling_rate=sa_cooling_rate
            )
            print(f"  SA refined set for R_sq={target_R_sq} has {len(sa_refined_points)} points.")

            if len(sa_refined_points) > max_num_points:
                is_valid_sa, N_sq_max_sa, D_sq_min_sa = check_constraint(sa_refined_points)
                if is_valid_sa and N_sq_max_sa == target_R_sq and D_sq_min_sa >= target_R_sq:
                    max_num_points = len(sa_refined_points)
                    best_overall_points = sa_refined_points.copy()
                    print(f"  New best found from SA: {max_num_points} points for R_sq={target_R_sq}.")
            
    end_time = time.time()
    print(f"\nTotal execution time: {end_time - start_time:.2f} seconds.")
    print(f"Final best number of points: {len(best_overall_points)}")

    return best_overall_points

# EVOLVE-BLOCK-END