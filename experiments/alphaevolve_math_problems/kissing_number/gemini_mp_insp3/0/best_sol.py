# EVOLVE-BLOCK-START
import numpy as np
import itertools
import math # For sqrt

def kissing_number11()->np.ndarray:
    """
    Constructs a collection of 11-dimensional points with integral coordinates such that their maximum norm is strictly smaller than their minimum pairwise distance, aiming to maximize the number of points.
    
    Returns:
        points: np.ndarray of shape (num_points,11)
    """
    d = 11
    np.random.seed(42) # For reproducibility and determinism

    # Helper to generate integer points with a specific squared norm (excluding origin)
    def gen_points_on_sphere(dim, target_sq_norm):
        points_list = []
        
        # Recursive helper function to build points coordinate by coordinate
        # current_point_list: A list representing the point being built
        # current_sq_norm: The squared Euclidean norm of the point built so far
        # k: The current dimension index being processed (0 to dim-1)
        def _gen(current_point_list, current_sq_norm, k):
            # Base case: if all dimensions are processed
            if k == dim:
                # If the squared norm exactly matches the target, add the point
                if current_sq_norm == target_sq_norm:
                    points_list.append(np.array(current_point_list, dtype=np.int64))
                return
            
            # Pruning 1: If current_sq_norm already exceeds target_sq_norm, no need to continue
            if current_sq_norm > target_sq_norm:
                return

            # Calculate remaining squared norm needed for the rest of the dimensions
            remaining_target_sq_norm = target_sq_norm - current_sq_norm
            
            # Calculate the maximum possible absolute value for the current coordinate
            # (since val*val must be <= remaining_target_sq_norm)
            max_coord_val = int(math.sqrt(remaining_target_sq_norm))
            
            # Iterate through possible integer values for the current coordinate
            # Start from max_coord_val down to -max_coord_val for potential greedy benefits,
            # though sorting later makes this less critical.
            for val in range(max_coord_val, -max_coord_val - 1, -1):
                # Pruning 2: If adding val*val would exceed the target_sq_norm, skip this value
                if current_sq_norm + val*val <= target_sq_norm:
                    current_point_list[k] = val
                    _gen(current_point_list, current_sq_norm + val*val, k + 1)

        _gen([0] * dim, 0, 0) # Start with an all-zero point and 0 squared norm
        return points_list

    # --- Greedy Algorithm for points on a fixed norm sphere ---
    # The initial solution points (1,...,1) and (-1,...,-1) have squared norm = 11.
    # This suggests that the optimal set might consist of points all having squared norm 11.
    # If all points in S have the same squared norm M, then N_max_sq = M.
    # The problem condition N_max < D_min simplifies to M < D_min_sq.
    # This means for any two distinct points p_i, p_j in S, their squared distance ||p_i - p_j||^2 must be strictly greater than M.

    fixed_N_max_sq = 11 # Target squared norm for all points in the set
    
    # Generate all integer points in 11D with squared norm exactly 11
    candidate_points_on_sphere = gen_points_on_sphere(d, fixed_N_max_sq)
    
    # Sort candidates for deterministic greedy selection and potential performance benefits.
    # Sorting by sparsity (number of non-zero coordinates) then lexicographically.
    # This prioritizes 'simpler' points.
    candidate_points_on_sphere.sort(key=lambda p: (np.count_nonzero(p), tuple(p)))

    S = [] # The final set of points (list of numpy arrays)
    points_set = set() # For efficient O(1) lookup of existing points (stores tuple versions)
    
    for new_point in candidate_points_on_sphere:
        point_tuple = tuple(new_point)
        # Skip if this point (or its negative, or a permutation) has already been added.
        # The `gen_points_on_sphere` generates unique points, but the greedy algorithm might add
        # a point's negative later, and we need to ensure it's still valid.
        if point_tuple in points_set:
            continue
        
        # Check if adding new_point conflicts with any existing point in S.
        # A conflict occurs if ||new_point - existing_point||^2 <= fixed_N_max_sq.
        is_conflict = False
        for existing_point in S:
            dist_sq = np.sum((new_point - existing_point)**2)
            if dist_sq <= fixed_N_max_sq: # Condition: D_min_sq > N_max_sq
                is_conflict = True
                break
        
        # If no conflict, add the new_point to the set
        if not is_conflict:
            S.append(new_point)
            points_set.add(point_tuple)
            
    return np.array(S, dtype=np.int64)

# EVOLVE-BLOCK-END