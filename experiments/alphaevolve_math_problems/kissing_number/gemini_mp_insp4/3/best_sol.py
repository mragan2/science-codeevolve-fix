# EVOLVE-BLOCK-START
import numpy as np
from itertools import combinations, product

def kissing_number11() -> np.ndarray:
    """
    Constructs a collection of 11-dimensional points with integral coordinates
    such that their maximum norm is less than or equal to their minimum pairwise distance,
    aiming to maximize the number of points.

    This implementation uses a greedy algorithm to find a large independent set
    on a graph of candidate points. The candidates are chosen based on a
    structure known to yield good results for this type of problem.

    Strategy:
    1.  Define a universe of candidate points with a fixed structure: vectors
        with exactly 5 non-zero entries, all of which are +/- 2.
    2.  All such points have a constant squared L2 norm of 5 * (2^2) = 20.
    3.  The problem constraint becomes finding a subset of these points where the
        minimum pairwise squared distance is at least 20.
    4.  This is a maximum independent set problem. We use a greedy approach:
        iterate through the candidates in a fixed random order and add a point
        to the solution if it doesn't conflict with any already selected points.
    5.  A fixed random seed (42) is used for reproducibility.

    Returns:
        points: np.ndarray of shape (num_points, 11)
    """
    d = 11
    # Changed candidate type to target sq_norm = 4, using 4 non-zero entries of +/-1.
    # This provides a larger pool (5280 candidates) and a lower maximum norm,
    # which is often conducive to higher packing density and thus more points.
    # The condition for adding a point 'p' given 's' in the solution set is
    # ||p-s||^2 >= sq_norm.
    # For points of the same norm, this simplifies to sq_norm >= 2 * np.dot(p,s).
    # With sq_norm=4, this means np.dot(p,s) <= 2.
    num_nonzero = 4
    value = 1
    sq_norm = num_nonzero * value**2 # This will now be 4 * 1^2 = 4

    # 1. Generate all candidate points
    candidates = []
    positions_iter = combinations(range(d), num_nonzero)
    for positions in positions_iter:
        signs_iter = product([-value, value], repeat=num_nonzero)
        base_vector = np.zeros(d, dtype=np.int64)
        for signs in signs_iter:
            p = base_vector.copy()
            p[list(positions)] = signs
            candidates.append(p)
    
    # 2. Sort candidates by degree in the conflict graph instead of shuffling.
    # This is a common and effective heuristic for the greedy independent set algorithm.
    # A low-degree point conflicts with fewer other points, so picking it first
    # leaves more options available for subsequent selections.

    # Convert list of arrays to a single 2D array for vectorized computation
    candidates_arr = np.array(candidates, dtype=np.int64)

    # Compute the dot products between all pairs of candidates
    dot_products = np.dot(candidates_arr, candidates_arr.T)

    # Two points p, s conflict if sq_norm < 2 * dot(p, s)
    conflict_matrix = (2 * dot_products) > sq_norm

    # A point does not conflict with itself. The diagonal of dot_products is sq_norm,
    # so 2*sq_norm > sq_norm is true. We must set the diagonal to False.
    np.fill_diagonal(conflict_matrix, False)

    # The degree of a candidate is the number of other candidates it conflicts with.
    degrees = np.sum(conflict_matrix, axis=1)

    # Get the indices that would sort the candidates by their degree, ascending.
    # 'stable' kind ensures that ties are broken deterministically, maintaining reproducibility.
    sorted_indices = np.argsort(degrees, kind='stable')

    solution_set = []
    # Use the degree-sorted list for the greedy search.
    candidates_list = [candidates[i] for i in sorted_indices]

    # 3. Perform greedy selection
    for p in candidates_list:
        is_valid_to_add = True
        for s in solution_set:
            # Check squared distance: ||p - s||^2 >= sq_norm
            # Optimized check: ||p-s||^2 = ||p||^2 + ||s||^2 - 2*p.s
            # => sq_norm + sq_norm - 2*p.s >= sq_norm
            # => sq_norm >= 2 * p.s
            if sq_norm < 2 * np.dot(p, s):
                is_valid_to_add = False
                break
        
        if is_valid_to_add:
            solution_set.append(p)
            
    return np.array(solution_set, dtype=np.int64)

# EVOLVE-BLOCK-END