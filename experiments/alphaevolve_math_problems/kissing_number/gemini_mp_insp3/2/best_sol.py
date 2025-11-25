# EVOLVE-BLOCK-START
import numpy as np
from itertools import combinations

def kissing_number11()->np.ndarray:
    """
    Constructs a collection of 11-dimensional points with integral coordinates
    such that their maximum norm is smaller than their minimum pairwise distance,
    aiming to maximize the number of points.

    This implementation constructs a set of 112 points by combining two
    compatible sets of vectors.

    Set 1 (110 points): All permutations of the vector (3, 3, -1, ..., -1)
    and their negatives. For this set:
    - M_1^2 = 2*3^2 + 9*(-1)^2 = 27
    - D_1^2 = 2*(3 - (-1))^2 = 32
    The condition M_1^2 < D_1^2 (27 < 32) is satisfied.

    Set 2 (2 points): The all-ones vector (1, ..., 1) and its negative.
    - M_2^2 = 11*1^2 = 11

    Combined Set: The union of Set 1 and Set 2.
    - M^2 = max(27, 11) = 27
    - D^2 = min(D_1^2, D_2^2, D_cross^2) = min(32, 44, 32) = 32
    The final condition M^2 < D^2 (27 < 32) holds for the combined set.

    Returns:
        points: np.ndarray of shape (112, 11)
    """
    d = 11
    
    # Set 1: Permutations of (3, 3, -1, ..., -1) and their negatives
    points_list_1 = []
    val_a = 3
    num_a = 2
    val_b = -1
    
    for positions_a in combinations(range(d), num_a):
        p = np.full(d, val_b, dtype=np.int64)
        p[list(positions_a)] = val_a
        points_list_1.append(p)

    base_points = np.array(points_list_1, dtype=np.int64)
    set1_points = np.vstack([base_points, -base_points])

    # Set 2: The all-ones vector and its negative
    all_ones = np.ones(d, dtype=np.int64)
    set2_points = np.vstack([all_ones, -all_ones])
    
    # Combine the two sets
    points = np.vstack([set1_points, set2_points])

    return points

# EVOLVE-BLOCK-END