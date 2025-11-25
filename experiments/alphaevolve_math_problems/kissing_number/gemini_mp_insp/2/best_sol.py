# EVOLVE-BLOCK-START
import numpy as np
import itertools
from scipy.spatial import KDTree # New import for efficient neighbor searching

def kissing_number11() -> np.ndarray:
    """
    Constructs a collection of 11-dimensional points with integral coordinates such that their maximum norm
    is less than or equal to their minimum pairwise distance, aiming to maximize the number of points.

    This implementation first generates candidate points from the second shell of the D_11 lattice
    (vectors with squared L2 norm 4). These points individually have an L2 norm of 2.
    The problem constraint requires that if the maximum L2 norm is 2, then the minimum pairwise L2 distance
    must be at least 2 (i.e., squared distance >= 4).

    However, the full set of D_11 shell-4 points contains pairs with an L2 distance of sqrt(2)
    (squared L2 distance 2), which violates the constraint.
    Therefore, a greedy Maximum Independent Set (MIS) algorithm is applied to select a subset of these
    candidate points such that all selected points have L2 norm 2 and are separated by at least L2 distance 2.

    The candidate points consist of two types:
    1. ( +/- 2, 0, ..., 0 )
    2. ( +/- 1, +/- 1, +/- 1, +/- 1, 0, ..., 0 )

    Returns:
        points: np.ndarray of shape (num_points, 11) representing the valid set of points.
    """
    d = 11
    candidate_points_list = []

    # Type A: ( +/- 2, 0, ..., 0 )
    # These are vectors with a single non-zero coordinate of +/- 2.
    # L2 norm is sqrt(2^2) = 2, squared L2 norm = 4.
    for i in range(d):
        p_plus = np.zeros(d, dtype=np.int64)
        p_minus = np.zeros(d, dtype=np.int64)
        p_plus[i] = 2
        p_minus[i] = -2
        candidate_points_list.append(p_plus)
        candidate_points_list.append(p_minus)

    # Type B: ( +/- 1, +/- 1, +/- 1, +/- 1, 0, ..., 0 )
    # These are vectors with four non-zero coordinates of +/- 1.
    # L2 norm is sqrt(1^2 * 4) = 2, squared L2 norm = 4.
    for indices in itertools.combinations(range(d), 4):
        for signs_tuple in itertools.product([-1, 1], repeat=4):
            p = np.zeros(d, dtype=np.int64)
            for i, sign in zip(indices, signs_tuple):
                p[i] = sign
            candidate_points_list.append(p)

    # Convert to numpy array and sort lexicographically for deterministic greedy selection
    candidate_points = np.array(candidate_points_list, dtype=np.int64)
    # np.lexsort sorts rows based on columns from last to first
    candidate_points = candidate_points[np.lexsort(candidate_points.T[::-1])]

    # --- Apply Greedy Maximum Independent Set algorithm ---
    # The constraint is max_norm <= min_distance.
    # All candidate points have L2 norm 2 (squared norm 4).
    # Thus, we need min_pairwise_distance >= 2 (squared distance >= 4).
    # We need to filter out pairs of points (p_i, p_j) where ||p_i - p_j||_2^2 == 2.

    # 1. Build a KDTree for efficient neighbor searching
    kdtree = KDTree(candidate_points)

    # 2. Identify "bad" neighbors (those with squared distance 2)
    # adj_list[i] will contain indices j such that ||candidate_points[i] - candidate_points[j]||_2^2 == 2
    num_candidates = len(candidate_points)
    adj_list = [[] for _ in range(num_candidates)]

    # Search radius for KDTree: sqrt(2) + epsilon.
    # KDTree's query_ball_point finds points within this radius. We then filter for exactly squared distance 2.
    search_radius = np.sqrt(2) + 1e-9 # Add epsilon for float precision robustness

    for i in range(num_candidates):
        p_i = candidate_points[i]
        # query_ball_point returns indices of neighbors within the radius
        neighbors_in_radius_indices = kdtree.query_ball_point(p_i, search_radius)
        
        for j in neighbors_in_radius_indices:
            if i == j: # Skip self-comparison
                continue
            
            p_j = candidate_points[j]
            squared_distance = np.sum((p_i - p_j)**2)
            
            if squared_distance == 2:
                adj_list[i].append(j)
    
    # 3. Greedy Maximum Independent Set (MIS) selection
    # We iterate through the sorted candidate points. If a point has not been "removed"
    # (i.e., neither it nor its neighbors have been selected for the MIS), we select it
    # and mark it and all its "bad" neighbors (those at squared distance 2) as removed.
    
    selected_indices = []
    # `is_removed` tracks points that are either selected or are neighbors of selected points
    is_removed = np.zeros(num_candidates, dtype=bool)

    for i in range(num_candidates):
        if not is_removed[i]:
            # If point 'i' is not removed, it can be added to our independent set
            selected_indices.append(i)
            is_removed[i] = True # Mark point 'i' as removed (because it's selected)
            
            # Mark all its "bad" neighbors as removed to ensure they are not selected
            # (as they would violate the min_distance constraint with point 'i')
            for neighbor_idx in adj_list[i]:
                is_removed[neighbor_idx] = True
    
    final_points = candidate_points[selected_indices]
    return final_points

# EVOLVE-BLOCK-END