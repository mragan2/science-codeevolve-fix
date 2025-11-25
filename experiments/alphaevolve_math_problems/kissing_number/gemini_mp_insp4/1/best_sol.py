# EVOLVE-BLOCK-START
import numpy as np
import itertools
from collections import defaultdict
from scipy.spatial import KDTree
from numba import njit # For performance

# JIT-compiled helper functions for efficiency
@njit(cache=True)
def _dist_sq(p1, p2):
    """Calculates squared Euclidean distance between two numpy arrays."""
    return np.sum((p1 - p2)**2)

@njit(cache=True)
def _check_parity(p):
    """Checks if the sum of coordinates is even."""
    return np.sum(p) % 2 == 0

def generate_points_with_norm_sq(d, target_norm_sq):
    """
    Generates all integer points p in Z^d such that ||p||_2^2 == target_norm_sq.
    This function uses a recursive backtracking approach to find the absolute values
    of coordinates, then generates permutations and sign combinations.
    """
    
    abs_coord_sets = set()

    def find_abs_coords_recursive(current_sum_sq, k, current_coords):
        if k == d:
            if current_sum_sq == target_norm_sq:
                abs_coord_sets.add(tuple(sorted(current_coords, reverse=True)))
            return

        max_val_for_dim_sq = target_norm_sq - current_sum_sq
        if max_val_for_dim_sq < 0:
            return
        max_val_for_dim = int(np.sqrt(max_val_for_dim_sq))
        
        for val in range(max_val_for_dim, -1, -1):
            val_sq = val * val
            find_abs_coords_recursive(current_sum_sq + val_sq, k + 1, current_coords + [val])

    find_abs_coords_recursive(0, 0, [])

    result_points_set = set() 
    for abs_coords_tuple in abs_coord_sets:
        for p_tuple in set(itertools.permutations(abs_coords_tuple)):
            non_zero_indices = [i for i, x in enumerate(p_tuple) if x != 0]
            num_actual_non_zero = len(non_zero_indices)
            
            for signs_tuple in itertools.product([-1, 1], repeat=num_actual_non_zero):
                point_list = list(p_tuple)
                for i, sign in zip(non_zero_indices, signs_tuple):
                    point_list[i] *= sign
                result_points_set.add(tuple(point_list))
                
    return [np.array(p, dtype=np.int64) for p in result_points_set]

def kissing_number11()->np.ndarray:
    """
    Constructs a collection of 11-dimensional points with integral coordinates such that their maximum norm
    is smaller than or equal to their minimum pairwise distance, aiming to maximize the number of points.

    Returns:
        points: np.ndarray of shape (num_points,11)
    """
    d = 11
    
    # Target condition: max_norm_sq <= min_dist_sq
    # The benchmark of 593 points suggests that max_norm_sq = 4 might be achievable.
    # If max_norm_sq = 4, then min_dist_sq must be >= 4.
    # This implies all points p in the set must satisfy ||p||_2^2 <= 4.
    # And all pairs of points (p_i, p_j) must satisfy ||p_i - p_j||_2^2 >= 4.

    # Candidate points:
    # 1. All points with squared norm 2 (C2_points). For d=11, these are (±1, ±1, 0, ..., 0).
    #    Number of points: C(11,2) * 2^2 = 55 * 4 = 220. All have even coordinate sum (are in D_11).
    # 2. All points with squared norm 4 (C4_points). For d=11, these are:
    #    - (±2, 0, ..., 0): 11 * 2 = 22 points. All have even coordinate sum.
    #    - (±1, ±1, ±1, ±1, 0, ..., 0): C(11,4) * 2^4 = 330 * 16 = 5280 points. All have even coordinate sum.
    # All these points belong to the D_11 lattice (sum of coordinates is even).
    # This is crucial because it guarantees that the minimum squared distance between any two D_11 points is at least 2.
    # If we included points with odd coordinate sums (e.g., C1 or C3), the minimum distance could be 1, which would violate R_max_sq <= D_min_sq if R_max_sq > 1.

    C2_points = generate_points_with_norm_sq(d, 2)
    C4_points = generate_points_with_norm_sq(d, 4)
    
    # Combine candidate points. No explicit parity filtering needed as C2 and C4 points in 11D
    # naturally have even coordinate sums.
    candidate_points_list = C2_points + C4_points
    
    # Convert to numpy array for KDTree for performance
    candidate_points_array = np.array(candidate_points_list, dtype=np.int64)
    
    if len(candidate_points_array) == 0:
        return np.array([], dtype=np.int64).reshape(0, d)

    # Build KDTree for efficient nearest neighbor queries.
    # We need to find pairs of points with squared distance < 4 (i.e., = 2, since all points are in D_11).
    kdtree = KDTree(candidate_points_array)

    # Build adjacency list for the conflict graph.
    # An edge (u, v) exists if ||u - v||_2^2 < 4.
    # Convert points to tuples for use as dictionary keys (numpy arrays are not hashable).
    candidate_points_tuples = [tuple(p) for p in candidate_points_list]
    adj = defaultdict(set)
    
    # Query for neighbors within radius sqrt(3.999...) to find all points with squared distance 2.
    # The radius is slightly less than sqrt(4) to avoid floating point issues and capture only distance 2.
    for i, p_tuple in enumerate(candidate_points_tuples):
        # query_ball_point returns indices of neighbors within the specified radius.
        indices = kdtree.query_ball_point(candidate_points_array[i], r=np.sqrt(3.999999))
        
        for neighbor_idx in indices:
            if neighbor_idx == i:
                continue # Skip self-comparison
            
            q_tuple = candidate_points_tuples[neighbor_idx]
            # Verify actual squared distance to be absolutely sure (though for integer lattice points,
            # distances below 4 will typically be exactly 2 for D_11 points).
            if _dist_sq(candidate_points_array[i], candidate_points_array[neighbor_idx]) < 4:
                adj[p_tuple].add(q_tuple)
                adj[q_tuple].add(p_tuple) # Graph is undirected

    # Greedy Maximum Independent Set (MIS) algorithm.
    # Sort points by their degree (number of conflicts) in ascending order.
    # This heuristic tends to pick points that are less "problematic", leaving more options for other points.
    
    # Create a list of (degree, point_tuple) pairs
    degrees_and_points = [(len(adj[p]), p) for p in candidate_points_tuples]
    degrees_and_points.sort() # Sorts primarily by degree, then by point_tuple (deterministic tie-breaking)

    solution_set = set()
    removed_points = set() # Keep track of points already in solution or conflicting with solution.

    for _, p_tuple in degrees_and_points:
        if p_tuple not in removed_points:
            # If point p has not been removed, it can be added to the independent set.
            solution_set.add(p_tuple)
            removed_points.add(p_tuple) # Mark p as removed (selected)
            
            # Remove all neighbors of p, as they conflict with p.
            for neighbor_tuple in adj[p_tuple]:
                removed_points.add(neighbor_tuple)
    
    # Convert the solution set back to numpy arrays for the final output.
    final_points = [np.array(p, dtype=np.int64) for p in solution_set]
    
    return np.array(final_points, dtype=np.int64)

# EVOLVE-BLOCK-END