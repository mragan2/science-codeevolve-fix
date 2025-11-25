# EVOLVE-BLOCK-START
import numpy as np
import itertools
import time
from scipy.spatial import KDTree
import heapq # For priority queue

# For determinism if multiple candidates have the same norm,
# ensure consistent tie-breaking by including a unique ID or the point tuple.
_counter = itertools.count()

def generate_candidates_by_increasing_norm(dim, max_coord_abs, max_norm_sq_cutoff=float('inf')):
    """
    Generates integer points in increasing order of their squared L2 norm
    using a min-priority queue (heap). This version is optimized for speed.
    Yields (norm_sq, point_array) tuples.
    
    max_norm_sq_cutoff: Stop generating candidates once their norm_sq exceeds this value.
    max_coord_abs: The maximum absolute value for any coordinate.
    """
    visited = {tuple([0] * dim)} # To store point tuples and avoid redundant work. Start with origin.
    pq = [] # Min-priority queue: (norm_sq, counter_id, point_tuple)

    # Add initial points with norm_sq=1
    for i in range(dim):
        # We only need to add one of (+1, -1) for each dimension, e.g., (1,0,...)
        # as the symmetric points will be found as neighbors of other points.
        # But adding both is fine and doesn't hurt correctness.
        for val in [-1, 1]:
            p_list = [0] * dim
            p_list[i] = val
            p_tuple = tuple(p_list)
            # norm_sq is 1. Check against cutoff just in case it's < 1.
            if 1 <= max_norm_sq_cutoff and p_tuple not in visited:
                heapq.heappush(pq, (1, next(_counter), p_tuple))
                visited.add(p_tuple)

    while pq:
        norm_sq, _, p_tuple = heapq.heappop(pq)
        
        # The heap guarantees we process points in increasing order of norm_sq.
        # If the smallest item exceeds the cutoff, all subsequent items will too.
        if norm_sq > max_norm_sq_cutoff:
            break

        yield norm_sq, np.array(p_tuple, dtype=np.int64)

        # Generate neighbors by changing one coordinate by +/-1.
        p_list = list(p_tuple)
        for i in range(dim):
            original_val = p_list[i]
            
            # Try incrementing coordinate
            new_val_inc = original_val + 1
            if new_val_inc <= max_coord_abs:
                p_list[i] = new_val_inc
                neighbor_tuple = tuple(p_list)
                if neighbor_tuple not in visited:
                    # Optimized incremental norm calculation: norm_new = norm_old + 2*c + 1
                    neighbor_norm_sq = norm_sq + 2 * original_val + 1
                    if neighbor_norm_sq <= max_norm_sq_cutoff:
                        heapq.heappush(pq, (neighbor_norm_sq, next(_counter), neighbor_tuple))
                        visited.add(neighbor_tuple)
            
            # Try decrementing coordinate
            new_val_dec = original_val - 1
            if new_val_dec >= -max_coord_abs:
                p_list[i] = new_val_dec
                neighbor_tuple = tuple(p_list)
                if neighbor_tuple not in visited:
                    # Optimized incremental norm calculation: norm_new = norm_old - 2*c + 1
                    neighbor_norm_sq = norm_sq - 2 * original_val + 1
                    if neighbor_norm_sq <= max_norm_sq_cutoff:
                        heapq.heappush(pq, (neighbor_norm_sq, next(_counter), neighbor_tuple))
                        visited.add(neighbor_tuple)
            
            p_list[i] = original_val # Restore for next dimension's modification


def kissing_number11()->np.ndarray:
    """
    Constructs a collection of 11-dimensional points with integral coordinates such that their maximum norm is smaller than their minimum pairwise distance, aiming to maximize the number of points. 

    Returns:
        points: np.ndarray of shape (num_points,11)
    """
    d = 11
    
    # --- Greedy Constructive Algorithm ---
    
    # 1. Initialization
    S_points_list = []
    current_max_norm_sq = 0
    current_min_dist_sq = float('inf')

    # 2. Candidate Point Generation & Processing (Iterating over target R_squared values)
    # The greedy approach will be run for different target R_squared values.
    # We aim to find the largest set for which max_norm_sq == R_squared and min_dist_sq >= R_squared.
    
    # 1. Initialization for the overall best set
    best_points_found = np.empty((0, d), dtype=np.int64)
    max_k = 0

    # 2. Hyperparameters for the search space.
    # Increased `max_coord_abs` to 2 to allow for coordinates {-2, -1, 0, 1, 2}.
    # This significantly expands the search space to include points crucial for denser packings.
    # `max_target_r_squared` is adjusted to cover relevant norms for this coordinate range.
    max_coord_abs = 2 # Coordinates can be -2, -1, 0, 1, or 2
    max_target_r_squared = 8 # Covers points like (2,0,...,0) (norm 4), (1,1,1,1,1,1,1,1,0,...,0) (norm 8), etc.

    # Collect all candidates once. This is efficient given the optimized generator
    # and manageable search space.
    all_candidates_by_norm = {} # {norm_sq: [point1, point2, ...]}
    for N_cand_sq, p_candidate in generate_candidates_by_increasing_norm(d, max_coord_abs, max_target_r_squared):
        if N_cand_sq not in all_candidates_by_norm:
            all_candidates_by_norm[N_cand_sq] = []
        all_candidates_by_norm[N_cand_sq].append(p_candidate)

    # Sort the R_squared values to process them in increasing order.
    sorted_r_squared_values = sorted(all_candidates_by_norm.keys())

    # Iterate through each possible target R_squared value.
    # For each R_squared, we try to build the largest set where all points
    # have that exact R_squared norm, and their pairwise distances are >= R_squared.
    for target_r_squared in sorted_r_squared_values:
        if target_r_squared == 0: # The origin (0,0,...,0) is not a useful point for this problem's constraint
            continue

        current_s_points_list = [] # Points for the current R_squared iteration
        
        # Get candidates for the current target R_squared.
        # These are already ordered by norm, which is exactly `target_r_squared`.
        candidates_for_r_squared = all_candidates_by_norm.get(target_r_squared, [])
        
        # Greedy selection for the current target_r_squared
        for p_candidate in candidates_for_r_squared:
            if not current_s_points_list:
                current_s_points_list.append(p_candidate)
            else:
                # Convert list to array for vectorized operations.
                current_s_points_arr = np.array(current_s_points_list, dtype=np.int64)
                
                # Calculate squared Euclidean distances from the candidate to all existing points.
                # This is the most computationally intensive part.
                dist_sq_to_existing = np.sum((current_s_points_arr - p_candidate)**2, axis=1)
                min_dist_sq_to_existing = np.min(dist_sq_to_existing)
                
                # If candidate is identical to an existing point, skip it.
                if min_dist_sq_to_existing == 0:
                    continue

                # Constraint check: all points in the set must have norm_sq == target_r_squared,
                # and all pairwise distances must be >= target_r_squared.
                if min_dist_sq_to_existing >= target_r_squared:
                    current_s_points_list.append(p_candidate)
        
        # After processing all candidates for this R_squared, update the overall best set.
        if len(current_s_points_list) > max_k:
            max_k = len(current_s_points_list)
            best_points_found = np.array(current_s_points_list, dtype=np.int64)
            # print(f"Found new best for R^2={target_r_squared}: {max_k} points.")

    return best_points_found

# EVOLVE-BLOCK-END