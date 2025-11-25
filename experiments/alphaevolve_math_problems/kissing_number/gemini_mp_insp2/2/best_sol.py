# EVOLVE-BLOCK-START
import numpy as np
import itertools
from collections import defaultdict

def generate_candidates_K2(d):
    """
    Generates integer vectors in d dimensions with squared L2 norm 2.
    These are vectors with exactly two non-zero coordinates, both +/-1.
    e.g., (1,1,0,...), (1,-1,0,...), etc.
    """
    candidates = []
    for i in range(d):
        for j in range(i + 1, d):
            for s1 in [-1, 1]:
                for s2 in [-1, 1]:
                    v = np.zeros(d, dtype=np.int64)
                    v[i] = s1
                    v[j] = s2
                    candidates.append(v)
    return np.array(candidates)

def generate_candidates_K4(d):
    """
    Generates integer vectors in d dimensions with squared L2 norm 4.
    These are vectors of type (2,0,...,0) or (1,1,1,1,0,...,0) and their permutations/sign changes.
    """
    candidates = []

    # Type ( +/-2, 0, ..., 0 )
    for i in range(d):
        for s in [-1, 1]:
            v = np.zeros(d, dtype=np.int64)
            v[i] = s * 2
            candidates.append(v)

    # Type ( +/-1, +/-1, +/-1, +/-1, 0, ..., 0 )
    for indices in itertools.combinations(range(d), 4):
        for signs in itertools.product([-1, 1], repeat=4):
            v = np.zeros(d, dtype=np.int64)
            for k, idx in enumerate(indices):
                v[idx] = signs[k]
            candidates.append(v)
            
    # Use np.unique to remove any accidental duplicates (though construction should prevent it)
    # and ensure canonical representation for sorting
    unique_candidates_tuples = sorted([tuple(p) for p in candidates])
    return np.array(unique_candidates_tuples, dtype=np.int64)

def greedy_independent_set(candidates_np, K):
    """
    Greedy algorithm to find a large independent set from candidates.
    A point p is added to the set if it maintains the condition:
    max_norm_sq (which is K for all candidates) <= min_pairwise_dist_sq.
    This means for all q in selected_points, ||p-q||^2 >= K.
    
    Args:
        candidates_np (np.ndarray): An array of candidate points, all with squared L2 norm K.
        K (int): The common squared L2 norm of the candidate points.

    Returns:
        np.ndarray: The selected set of points.
    """
    if candidates_np.shape[0] == 0:
        return np.array([], dtype=np.int64).reshape(0, candidates_np.shape[1])

    # Sort candidates by L1 norm, then lexicographically for determinism.
    # Prioritizing points with smaller L1 norm (e.g., (2,0,0...) over (1,1,1,1,0...))
    # can influence the greedy choice.
    sorted_indices = np.argsort([np.sum(np.abs(p)) for p in candidates_np])
    candidates_sorted = candidates_np[sorted_indices]
    
    # Secondary sort: lexicographical for full determinism if L1 norms are equal
    # This is handled by np.unique and then sorted(tuple(p)) if candidates_np was created from tuples
    # If not, need to do it explicitly
    
    # To ensure full determinism, convert to tuples for sorting if not already
    candidates_tuples = [tuple(p) for p in candidates_sorted]
    candidates_tuples_sorted = sorted(candidates_tuples)
    candidates_final_sorted = np.array(candidates_tuples_sorted, dtype=np.int64)


    selected_points_list = []
    
    # For efficient distance checking, store selected points in a list/array
    # and use a KDTree or similar structure if performance becomes an issue.
    # For 5302 candidates and ~600 selected, N_cand * N_selected * d operations is feasible.
    # 5302 * 600 * 11 ~ 3.5 * 10^7 operations.
    
    for p in candidates_final_sorted:
        is_valid = True
        for q in selected_points_list:
            dist_sq = np.sum((p - q)**2)
            if dist_sq < K:
                is_valid = False
                break
        if is_valid:
            selected_points_list.append(p)
    
    return np.array(selected_points_list, dtype=np.int64)


def kissing_number11()->np.ndarray:
    """
    Constructs a collection of 11-dimensional points with integral coordinates such that their 
    maximum norm is smaller than their minimum pairwise distance, aiming to maximize the number of points.

    Returns:
        points: np.ndarray of shape (num_points,11)
    """
    d = 11
    
    best_num_points = 0
    best_points = np.array([], dtype=np.int64).reshape(0, d) # Initialize with an empty array

    # Strategy 1: K=2
    # All points have squared norm 2. Minimum distance must be >= sqrt(2).
    # The set of all vectors with two +/-1s and rest zeros works.
    # For p=(1,1,0...) and q=(1,0,1...), ||p-q||^2 = ||(0,1,-1,...)||^2 = 2.
    # This satisfies the condition R_max^2 = 2 <= D_min^2 = 2.
    # Number of such points: C(11,2) * 2^2 = 55 * 4 = 220.
    points_K2 = generate_candidates_K2(d)
    if len(points_K2) > best_num_points:
        best_num_points = len(points_K2)
        best_points = points_K2
    print(f"K=2: Found {len(points_K2)} points.")

    # Strategy 2: K=4
    # All points have squared norm 4. Minimum distance must be >= sqrt(4)=2.
    # As discussed in analysis, the set of *all* such points does not satisfy the condition
    # (e.g., (1,1,1,1,0...) and (1,1,1,0,1...) have distance sqrt(2)).
    # We apply a greedy independent set algorithm.
    candidates_K4 = generate_candidates_K4(d)
    print(f"K=4: Generated {len(candidates_K4)} candidate points.")
    
    points_K4 = greedy_independent_set(candidates_K4, K=4)
    print(f"K=4: Selected {len(points_K4)} points using greedy independent set.")

    if len(points_K4) > best_num_points:
        best_num_points = len(points_K4)
        best_points = points_K4

    return best_points

# EVOLVE-BLOCK-END