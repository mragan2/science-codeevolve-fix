# EVOLVE-BLOCK-START
import numpy as np
from itertools import combinations, product

def generate_candidates(dim, max_norm_sq):
    """
    Generates all integer points p with 1 <= ||p||^2 <= max_norm_sq.
    This function explicitly constructs points based on integer partitions
    of the squared norm into sums of squares of coordinates (0, +/-1, +/-2, +/-3).
    """
    candidates = []
    
    # Norm 1: (1, 0, ...)
    if max_norm_sq >= 1: 
        for pos in combinations(range(dim), 1):
            # Generate (1,0,...) and (-1,0,...)
            p_pos = np.zeros(dim, dtype=np.int64); p_pos[list(pos)] = 1; candidates.append(p_pos)
            p_neg = np.zeros(dim, dtype=np.int64); p_neg[list(pos)] = -1; candidates.append(p_neg)

    # Norm 2: (1, 1, 0, ...)
    if max_norm_sq >= 2: 
        for pos in combinations(range(dim), 2):
            for signs in product([-1, 1], repeat=2):
                p = np.zeros(dim, dtype=np.int64); p[list(pos)] = signs; candidates.append(p)

    # Norm 3: (1, 1, 1, 0, ...)
    if max_norm_sq >= 3: 
        for pos in combinations(range(dim), 3):
            for signs in product([-1, 1], repeat=3):
                p = np.zeros(dim, dtype=np.int64); p[list(pos)] = signs; candidates.append(p)

    # Norm 4: (2, 0, ...), (1,1,1,1, 0,...)
    if max_norm_sq >= 4: 
        # (2, 0, ...)
        for pos in combinations(range(dim), 1):
            for signs in product([-2, 2], repeat=1):
                p = np.zeros(dim, dtype=np.int64); p[list(pos)] = signs; candidates.append(p)
        # (1,1,1,1, 0,...)
        for pos in combinations(range(dim), 4):
            for signs in product([-1, 1], repeat=4):
                p = np.zeros(dim, dtype=np.int64); p[list(pos)] = signs; candidates.append(p)

    # Norm 5: (2, 1, 0, ...), (1,1,1,1,1, 0,...)
    if max_norm_sq >= 5: 
        # (2, 1, 0, ...)
        for pos_2 in combinations(range(dim), 1):
            for pos_1 in combinations(set(range(dim)) - set(pos_2), 1):
                for signs in product([-2, 2], [-1, 1]):
                    p = np.zeros(dim, dtype=np.int64); p[list(pos_2)] = signs[0]; p[list(pos_1)] = signs[1]; candidates.append(p)
        # (1,1,1,1,1, 0,...)
        for pos in combinations(range(dim), 5):
            for signs in product([-1, 1], repeat=5):
                p = np.zeros(dim, dtype=np.int64); p[list(pos)] = signs; candidates.append(p)

    # Norm 6: (2, 1, 1, 0, ...), (1...1, 0,...)
    if max_norm_sq >= 6: 
        # (2, 1, 1, 0, ...)
        for pos_2 in combinations(range(dim), 1):
            for pos_1s in combinations(set(range(dim)) - set(pos_2), 2):
                for s2 in [-2, 2]:
                    for s1s in product([-1, 1], repeat=2):
                        p = np.zeros(dim, dtype=np.int64); p[list(pos_2)] = s2; p[list(pos_1s)] = s1s; candidates.append(p)
        # (1, ..., 1, 0, ...) (6 ones)
        for pos in combinations(range(dim), 6):
            for signs in product([-1, 1], repeat=6):
                p = np.zeros(dim, dtype=np.int64); p[list(pos)] = signs; candidates.append(p)

    # Norm 7: (2,1,1,1,0,...), (1...1,0,...)
    if max_norm_sq >= 7: 
        # (2,1,1,1,0,...)
        for pos_2 in combinations(range(dim), 1):
            for pos_1s in combinations(set(range(dim)) - set(pos_2), 3):
                for s2 in [-2, 2]:
                    for s1s in product([-1, 1], repeat=3):
                        p = np.zeros(dim, dtype=np.int64); p[list(pos_2)] = s2; p[list(pos_1s)] = s1s; candidates.append(p)
        # (1, ..., 1, 0, ...) (7 ones)
        for pos in combinations(range(dim), 7):
            for signs in product([-1, 1], repeat=7):
                p = np.zeros(dim, dtype=np.int64); p[list(pos)] = signs; candidates.append(p)

    # Norm 8: (2,2,0,...), (2,1,1,1,1,0,...), (1...1,0,...)
    if max_norm_sq >= 8: 
        # (2,2,0,...)
        for pos in combinations(range(dim), 2):
            for signs in product([-2, 2], repeat=2):
                p = np.zeros(dim, dtype=np.int64); p[list(pos)] = signs; candidates.append(p)
        # (2,1,1,1,1,0,...)
        for pos_2 in combinations(range(dim), 1):
            for pos_1s in combinations(set(range(dim)) - set(pos_2), 4):
                for s2 in [-2, 2]:
                    for s1s in product([-1, 1], repeat=4):
                        p = np.zeros(dim, dtype=np.int64); p[list(pos_2)] = s2; p[list(pos_1s)] = s1s; candidates.append(p)
        # (1, ..., 1, 0, ...) (8 ones)
        for pos in combinations(range(dim), 8):
            for signs in product([-1, 1], repeat=8):
                p = np.zeros(dim, dtype=np.int64); p[list(pos)] = signs; candidates.append(p)
    
    # Norm 9: (3,0,...), (2,2,1,0,...), (2,1,1,1,1,1,0,...), (1...1,0,...)
    if max_norm_sq >= 9:
        # (3, 0, ...)
        for pos in combinations(range(dim), 1):
            for signs in product([-3, 3], repeat=1):
                p = np.zeros(dim, dtype=np.int64); p[list(pos)] = signs; candidates.append(p)
        # (2, 2, 1, 0, ...)
        for pos_2_1 in combinations(range(dim), 2): # Positions for the two '2's
            remaining_dims = set(range(dim)) - set(pos_2_1)
            for pos_1 in combinations(remaining_dims, 1): # Position for the '1'
                for s2s in product([-2, 2], repeat=2):
                    for s1 in [-1, 1]:
                        p = np.zeros(dim, dtype=np.int64)
                        p[list(pos_2_1)] = s2s
                        p[list(pos_1)] = s1
                        candidates.append(p)
        # (2, 1, 1, 1, 1, 1, 0, ...) -> one '2', five '1's
        for pos_2 in combinations(range(dim), 1):
            remaining_dims = set(range(dim)) - set(pos_2)
            for pos_1s in combinations(remaining_dims, 5):
                for s2 in [-2, 2]:
                    for s1s in product([-1, 1], repeat=5):
                        p = np.zeros(dim, dtype=np.int64)
                        p[list(pos_2)] = s2
                        p[list(pos_1s)] = s1s
                        candidates.append(p)
        # (1, ..., 1, 0, ...) (9 ones)
        for pos in combinations(range(dim), 9):
            for signs in product([-1, 1], repeat=9):
                p = np.zeros(dim, dtype=np.int64); p[list(pos)] = signs; candidates.append(p)

    return candidates

def find_packing(dim, min_dist_sq):
    """Constructs a packing for a given squared distance threshold."""
    candidates = generate_candidates(dim, min_dist_sq)
    if not candidates:
        return np.array([])

    # Sort candidates by squared norm, descending. This is a crucial heuristic.
    # It prioritizes placing points on the boundary of the feasible norm-sphere.
    candidates.sort(key=lambda p: np.sum(p**2), reverse=True)
    candidates_np = np.array(candidates, dtype=np.int64)
    
    # Use a dynamic list for solution points for potentially better memory usage
    # if num_points is much smaller than len(candidates_np)
    solution_points_list = []

    # The first point is simply the highest-norm candidate if the list is not empty
    if len(candidates_np) > 0:
        solution_points_list.append(candidates_np[0])

    # Vectorized greedy selection
    for i in range(1, len(candidates_np)):
        p = candidates_np[i]
        
        # If no points yet, add the current one (should be handled by initial append)
        if not solution_points_list:
            solution_points_list.append(p)
            continue
            
        # Convert current solution to a NumPy array for vectorized distance check
        current_solution_np = np.array(solution_points_list, dtype=np.int64)
        
        # Check distance against all points found so far in a single vectorized operation
        dist_sq_vec = np.sum((p - current_solution_np)**2, axis=1)
        
        if np.all(dist_sq_vec >= min_dist_sq):
            solution_points_list.append(p)
            
    return np.array(solution_points_list, dtype=np.int64)

def kissing_number11() -> np.ndarray:
    """
    Finds a large set of points S in Z^11 satisfying max(||p||) <= min(||p-q||).

    This is achieved by searching for the best integer squared distance D_sq that
    maximizes the size of a set S where:
    1. For all p in S, ||p||^2 <= D_sq.
    2. For all distinct p, q in S, ||p - q||^2 >= D_sq.

    The search is performed over a promising range of D_sq values. For each D_sq,
    a greedy algorithm constructs the set using a vectorized distance check for
    efficiency. The best result across all tested D_sq values is returned.
    """
    dim = 11
    best_points = np.array([])
    
    # Search over a range of possible squared distances. Higher values are more
    # computationally expensive but may yield denser packings.
    # Prioritizing higher D_sq first as they generally allow for more points.
    # Also including D_sq=4 as it's a common lattice packing parameter and very fast.
    for d_sq in [9, 8, 7, 6, 5, 4]:
        points = find_packing(dim, d_sq)
        if len(points) > len(best_points):
            best_points = points
            
    return best_points

# EVOLVE-BLOCK-END