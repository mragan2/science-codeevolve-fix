# EVOLVE-BLOCK-START
import numpy as np
from numba import njit
import itertools
import math
import collections
from scipy.spatial import KDTree # New import for efficient neighbor search

@njit(cache=True)
def _calculate_squared_norm(point: np.ndarray) -> int:
    """Calculates the squared L2 norm of a point."""
    return np.sum(point * point)

@njit(cache=True)
def _calculate_squared_distance(p1: np.ndarray, p2: np.ndarray) -> int:
    """Calculates the squared L2 distance between two points."""
    return np.sum((p1 - p2) * (p1 - p2))

@njit(cache=True)
def _check_set_constraint_internal(points_array: np.ndarray):
    """
    Checks if a set of points satisfies the geometric constraint:
    max_i ||p_i||_2^2 <= min_{i != j} ||p_i - p_j||_2^2
    
    Args:
        points_array: A (k, d) numpy array of integer points.
        
    Returns:
        A tuple (is_valid, num_points, max_sq_norm, min_dist_sq).
    """
    k = points_array.shape[0]
    if k == 0:
        return True, 0, 0, np.inf
    if k == 1:
        # A single point is always valid. Max norm is its norm, min_dist_sq is inf.
        return True, 1, _calculate_squared_norm(points_array[0]), np.inf

    sq_norms = np.empty(k, dtype=np.int64)
    for i in range(k):
        sq_norms[i] = _calculate_squared_norm(points_array[i])
    max_sq_norm = np.max(sq_norms)

    min_dist_sq = np.inf
    # Use nested loops for pairwise distances, avoiding diagonal and duplicates
    for i in range(k):
        for j in range(i + 1, k):
            dist_sq = _calculate_squared_distance(points_array[i], points_array[j])
            if dist_sq < min_dist_sq:
                min_dist_sq = dist_sq
    
    is_valid = (max_sq_norm <= min_dist_sq)
    return is_valid, k, max_sq_norm, min_dist_sq

# New generalized candidate point generation function
def generate_all_candidate_points(d: int, max_sq_norm_limit: int) -> list:
    """
    Generates all d-dimensional integer points whose squared L2 norm is <= max_sq_norm_limit.
    This version generates canonical non-negative points first, then expands them to all
    permutations and sign combinations to cover the entire search space.
    Excludes the origin (0,...,0).
    """
    canonical_points_nn = [] # Non-negative, non-increasing order
    
    # Recursive helper to generate non-negative integer vectors (x_1, ..., x_d)
    # such that x_1 >= x_2 >= ... >= x_d >= 0 and sum(x_i^2) <= max_sq_norm_limit
    def _generate_canonical_recursive(index, current_sum_sq, current_coords):
        if index == d:
            if current_sum_sq > 0 and current_sum_sq <= max_sq_norm_limit: # Exclude origin (0,0,...,0)
                canonical_points_nn.append(np.array(current_coords, dtype=np.int64))
            return
        
        # Pruning: if current_sum_sq already exceeds max_sq_norm_limit, no need to proceed
        if current_sum_sq > max_sq_norm_limit:
            return

        # Determine the maximum possible value for the current coordinate
        # Based on remaining sum_sq and previous coordinate value (for non-increasing order)
        max_val = int(np.sqrt(max_sq_norm_limit - current_sum_sq))
        if index > 0:
            max_val = min(max_val, current_coords[index - 1])

        for x_val in range(max_val, -1, -1): # Iterate downwards for efficiency (larger values first)
            current_coords[index] = x_val
            _generate_canonical_recursive(index + 1, current_sum_sq + x_val*x_val, current_coords)

    _generate_canonical_recursive(0, 0, [0]*d)
    
    final_expanded_points = []
    seen_points = set() # Use a set to store tuple representations to avoid duplicates
    
    for p_canonical_np in canonical_points_nn:
        # Get unique permutations of the absolute values
        for perm_tuple in set(itertools.permutations(p_canonical_np)):
            perm_abs_array = np.array(perm_tuple, dtype=np.int64)
            
            # Generate all sign combinations for non-zero elements
            non_zero_indices = np.where(perm_abs_array != 0)[0]
            num_non_zero = len(non_zero_indices)
            
            for signs_tuple in itertools.product([-1, 1], repeat=num_non_zero):
                point = np.copy(perm_abs_array)
                for i, idx in enumerate(non_zero_indices):
                    point[idx] *= signs_tuple[i]
                
                point_tuple = tuple(point)
                if point_tuple not in seen_points:
                    final_expanded_points.append(point)
                    seen_points.add(point_tuple)
                    
    return final_expanded_points


def kissing_number11() -> np.ndarray:
    """
    Constructs a collection of 11-dimensional points with integral coordinates such that their maximum norm
    is smaller than their minimum pairwise distance, aiming to maximize the number of points.

    Returns:
        points: np.ndarray of shape (num_points, 11)
    """
    np.random.seed(42) # Ensure reproducibility
    d = 11

    # Define the maximum squared norm for generating the initial candidate pool.
    # This value balances the comprehensiveness of the search space with computational feasibility.
    # A value of 5 covers points with squared norms up to 5, including types like (±2, ±1, 0,...)
    # and (±1, ..., ±1, 0,...).
    # This results in ~22k candidate points, which is manageable with KDTree.
    max_sq_norm_candidate_pool = 5 
    print(f"Generating candidate points with squared L2 norm <= {max_sq_norm_candidate_pool}...")
    candidate_points = generate_all_candidate_points(d=d, max_sq_norm_limit=max_sq_norm_candidate_pool)
    print(f"Generated {len(candidate_points)} candidate points.")

    if len(candidate_points) == 0:
        return np.empty((0, d), dtype=np.int64)

    candidate_points_np = np.array(candidate_points, dtype=np.int64)
    num_candidates = len(candidate_points_np)

    # Pre-calculate squared norms for all candidates to speed up filtering
    candidate_sq_norms = np.array([_calculate_squared_norm(p) for p in candidate_points_np], dtype=np.int64)

    best_num_points = 0
    best_points = np.empty((0, d), dtype=np.int64)

    # Iterate through potential values for R_sq_final (the max_sq_norm for the *final* selected set).
    # For a valid set S, we need max_p in S ||p||_2^2 <= min_{p1, p2 in S} ||p1 - p2||_2^2.
    # This loop assumes max_p in S ||p||_2^2 = R_sq_final_attempt and builds a graph accordingly.
    # Start from higher values as they often allow for denser packings in this problem variant.
    for R_sq_final_attempt in range(max_sq_norm_candidate_pool, 0, -1):
        print(f"\n--- Attempting to find a set where max_sq_norm <= {R_sq_final_attempt} and min_dist_sq >= {R_sq_final_attempt} ---")
        
        # Filter candidate points: only consider those whose squared norm is <= R_sq_final_attempt.
        eligible_indices = np.where(candidate_sq_norms <= R_sq_final_attempt)[0]
        eligible_points_np = candidate_points_np[eligible_indices]
        
        num_eligible = len(eligible_points_np)
        
        # Optimization: if the number of eligible points is less than or equal to the current best, skip.
        # This is a heuristic that might sometimes miss better solutions if a smaller candidate pool
        # could yield a denser packing, but generally, more candidates are better.
        if num_eligible <= best_num_points: 
             print(f"  Skipping R_sq_final={R_sq_final_attempt} as number of eligible points ({num_eligible}) is not greater than current best ({best_num_points}).")
             continue

        if num_eligible == 0:
            continue

        print(f"  Considering {num_eligible} eligible points for R_sq_final={R_sq_final_attempt}.")

        # Build KDTree for efficient nearest neighbor searches among eligible points.
        eligible_kdtree = KDTree(eligible_points_np)
        
        adj_current = collections.defaultdict(list)
        
        # Determine the query radius for KDTree. We need points p_j such that ||p_i - p_j||_2^2 < R_sq_final_attempt.
        # This means ||p_i - p_j||_2 < sqrt(R_sq_final_attempt).
        # Add a small epsilon for floating-point safety to ensure we capture all points up to the boundary.
        query_radius = np.sqrt(R_sq_final_attempt) * (1 + 1e-9) 
        
        # Construct the adjacency graph for the Maximum Independent Set problem.
        # An edge (i, j) exists if points i and j are "incompatible" (their squared distance is too small).
        for i in range(num_eligible):
            p_i = eligible_points_np[i]
            # Find all points within the critical distance (excluding self, handled below).
            # This returns indices relative to `eligible_points_np`.
            neighbor_indices = eligible_kdtree.query_ball_point(p_i, query_radius)
            
            for j_idx_in_eligible in neighbor_indices:
                if i == j_idx_in_eligible: # Do not connect a node to itself
                    continue
                
                p_j = eligible_points_np[j_idx_in_eligible]
                dist_sq = _calculate_squared_distance(p_i, p_j)
                
                # Add an edge if their squared distance is strictly less than R_sq_final_attempt.
                if dist_sq < R_sq_final_attempt:
                    adj_current[i].append(j_idx_in_eligible)
                    # For an undirected graph, add the reverse edge as well for consistency with MIS heuristic.
                    adj_current[j_idx_in_eligible].append(i) 
        
        # Ensure unique neighbors in adjacency lists for accurate degree calculation.
        for node in adj_current:
            adj_current[node] = list(set(adj_current[node]))

        # --- Greedy Maximum Independent Set (MIS) heuristic ---
        # This heuristic prioritizes selecting nodes with the lowest degree first,
        # as they typically exclude fewer other nodes, potentially leading to a larger independent set.
        degrees = np.array([len(adj_current[i]) for i in range(num_eligible)], dtype=np.int64)
        
        if np.all(degrees == 0): # If no edges exist, all eligible points form a valid independent set.
            current_mis_indices = np.arange(num_eligible)
        else:
            sorted_indices = np.argsort(degrees) # Sort nodes by degree in ascending order.
            is_excluded = np.zeros(num_eligible, dtype=np.bool_) # Tracks nodes that are already selected or are neighbors of selected nodes.
            current_mis_indices = []
            
            for idx in sorted_indices:
                if not is_excluded[idx]:
                    current_mis_indices.append(idx) # Select this node for the MIS.
                    is_excluded[idx] = True # Mark itself as excluded (selected).
                    for neighbor_idx in adj_current[idx]:
                        is_excluded[neighbor_idx] = True # Exclude all its neighbors.
        
        final_set_points_for_R = eligible_points_np[current_mis_indices]
        
        # Final verification of the set obtained. This check should always pass if the graph
        # construction and MIS logic are correct for the chosen R_sq_final_attempt.
        is_valid, num_points_found, max_sq_norm_found, min_dist_sq_found = _check_set_constraint_internal(final_set_points_for_R)
        
        if is_valid:
            print(f"  Valid set found for R_sq_final={R_sq_final_attempt}: {num_points_found} points (max_sq_norm={max_sq_norm_found}, min_dist_sq={min_dist_sq_found})")
            if num_points_found > best_num_points:
                best_num_points = num_points_found
                best_points = final_set_points_for_R
        else:
            # This branch indicates a logic error, as the MIS for a given R_sq_final_attempt
            # should always produce a valid set where max_sq_norm_found <= R_sq_final_attempt <= min_dist_sq_found.
            print(f"  ERROR: MIS heuristic for R_sq_final={R_sq_final_attempt} resulted in an INVALID set. This indicates a bug in the algorithm.")

    print(f"\nOverall best solution found: {best_num_points} points.")
    return best_points

# EVOLVE-BLOCK-END