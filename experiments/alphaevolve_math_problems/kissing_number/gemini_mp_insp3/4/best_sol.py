# EVOLVE-BLOCK-START
import numpy as np
import itertools
from scipy.spatial.distance import cdist

def generate_norm3_points(d):
    """
    Generates all integer points in d dimensions with squared Euclidean norm 3.
    These points must have exactly three non-zero coordinates, each being +/- 1.
    """
    points = []
    # Iterate over all combinations of 3 positions out of d
    for positions in itertools.combinations(range(d), 3):
        # Iterate over all 2^3 sign combinations for these 3 positions
        for signs_tuple in itertools.product([-1, 1], repeat=3):
            p = np.zeros(d, dtype=np.int8)
            for i, pos in enumerate(positions):
                p[pos] = signs_tuple[i]
            points.append(p)
    return np.array(points)

def kissing_number11()->np.ndarray:
    """
    Constructs a set of 11-dimensional points with integral coordinates
    satisfying N_max < D_min, aiming to maximize the number of points.

    This implementation focuses on points with squared Euclidean norm 3.
    For such points, N_max = sqrt(3). The condition N_max < D_min
    implies D_min > sqrt(3), which means D_min^2 >= 4 (since distances
    between integer points are integers or square roots of integers,
    and we need strictly greater than sqrt(3)).

    The candidate points are all vectors in Z^11 with exactly three non-zero
    coordinates, each being +/- 1. There are C(11,3) * 2^3 = 165 * 8 = 1320
    such points.

    An improved greedy selection algorithm is used to find a maximal subset of these
    1320 candidates such that the minimum pairwise squared distance is at least 4.
    The condition D_min^2 >= 4 for points with squared norm 3 implies that
    their pairwise inner product p_i . p_j must be <= 1.
    The algorithm prioritizes adding points that have the fewest conflicts with
    other currently available candidates, to maximize the number of points.
    Ties are broken by the original index of the candidate (lexicographical order) for determinism.
    """
    d = 11
    # Generate all candidate points with squared norm 3
    candidates = generate_norm3_points(d) # This yields 1320 points

    # All these points have squared norm 3, so N_max = sqrt(3) for any subset.
    # The goal is to select a subset where D_min^2 >= 4.
    # This implies that for any two distinct points p_i, p_j in the set,
    # ||p_i - p_j||^2 >= 4.
    # Since ||p_i||^2 = ||p_j||^2 = 3, this means 2*3 - 2 * (p_i . p_j) >= 4
    # => 6 - 2 * (p_i . p_j) >= 4
    # => 2 >= 2 * (p_i . p_j)
    # => 1 >= p_i . p_j.
    # So we need to find a maximal set of candidates such that their pairwise inner products are <= 1.

    # Precompute all pairwise inner products
    # Shape: (num_candidates, num_candidates)
    inner_products = candidates @ candidates.T

    # Create a conflict matrix: conflict_matrix[i, j] is True if candidates[i] and candidates[j] conflict
    # A conflict occurs if inner_products[i, j] > 1 (i.e., inner_products[i, j] == 2, since values are integers)
    # Exclude self-inner product (which is 3)
    conflict_matrix = (inner_products > 1) & (inner_products != 3) # Conflicts if inner product is 2.

    # Initialize the set of available candidate indices
    available_indices = set(range(len(candidates)))
    selected_points_list = []
    
    # Improved greedy selection:
    # In each step, choose the available candidate that has the minimum number of conflicts
    # with other currently available candidates. Remove the chosen candidate and all
    # candidates it conflicts with from the available set.
    while available_indices:
        min_conflicts = float('inf')
        best_candidate_idx = -1
        
        # To ensure determinism, iterate through available candidates sorted by their original index.
        # This acts as a tie-breaker if multiple candidates have the same minimum number of conflicts.
        sorted_available_indices = sorted(list(available_indices))

        for current_idx in sorted_available_indices:
            num_current_conflicts = 0
            for other_idx in sorted_available_indices:
                if current_idx != other_idx and conflict_matrix[current_idx, other_idx]:
                    num_current_conflicts += 1
            
            if num_current_conflicts < min_conflicts:
                min_conflicts = num_current_conflicts
                best_candidate_idx = current_idx
            # If num_current_conflicts == min_conflicts, the candidate with the smaller
            # original index (due to sorted_available_indices) is implicitly chosen,
            # ensuring deterministic behavior.
        
        # Add the best candidate to the selected set
        selected_points_list.append(candidates[best_candidate_idx])
        
        # Identify all indices to remove: the selected candidate itself, and any
        # other available candidates that conflict with the selected one.
        indices_to_remove = {best_candidate_idx}
        for other_idx in available_indices:
            if conflict_matrix[best_candidate_idx, other_idx]:
                indices_to_remove.add(other_idx)
        
        # Update the set of available candidates
        available_indices.difference_update(indices_to_remove)

    final_points = np.array(selected_points_list, dtype=np.int8)
    
    # Final validation (optional, as the construction ensures it for N_max=sqrt(3) and D_min^2 >= 4)
    if final_points.shape[0] == 0:
        return np.array([], dtype=np.int8)

    # N_max check: All points in the final_points set have squared norm 3.
    N_max = np.sqrt(3)

    # D_min check:
    if final_points.shape[0] > 1:
        # Calculate all pairwise squared distances within the final set
        all_distances_sq = cdist(final_points, final_points, metric='sqeuclidean')
        # Extract upper triangle (excluding diagonal) to get unique distances between distinct points
        min_dist_sq = np.min(all_distances_sq[np.triu_indices(final_points.shape[0], k=1)])
        D_min = np.sqrt(min_dist_sq)
    else:
        D_min = np.inf # If 0 or 1 point, min distance is infinite

    # The condition N_max < D_min must be satisfied.
    # For this construction, N_max = sqrt(3) and D_min >= sqrt(4) = 2.
    # So sqrt(3) < 2, which is true.
    if not (N_max < D_min):
        raise ValueError(f"Constraint N_max ({N_max}) < D_min ({D_min}) NOT satisfied for the constructed set.")

    return final_points

# EVOLVE-BLOCK-END