# EVOLVE-BLOCK-START
import numpy as np
from itertools import combinations, product

def kissing_number11()->np.ndarray:
    """
    Constructs a collection of 11-dimensional points with integral coordinates such that their maximum norm is smaller than their minimum pairwise distance, aiming to maximize the number of points. 

    Returns:
        points: np.ndarray of shape (num_points,11)
    """
    d = 11

    # Strategy: Fixed Norm Approach with M=3, as suggested by the prompt.
    # The prompt hinted at 1320 points of type (±1, ±1, ±1, 0, ..., 0).
    # For these points, the squared Euclidean norm ||p||_2^2 = 1^2 + 1^2 + 1^2 = 3.
    # So, R^2 = 3.
    # The core condition is R < D_min, which is equivalent to R^2 < D_min^2.
    # Thus, we need D_min^2 > 3, meaning D_min^2 must be at least 4 (since it's an integer squared distance).

    # Let's derive the equivalent condition on the dot product (p . q) for any two points p, q.
    # For any two distinct points p, q from this set (where ||p||_2^2 = ||q||_2^2 = 3):
    # ||p - q||_2^2 = ||p||_2^2 + ||q||_2^2 - 2 * (p . q)
    # ||p - q||_2^2 = 3 + 3 - 2 * (p . q) = 6 - 2 * (p . q).
    # We require ||p - q||_2^2 >= 4.
    # So, 6 - 2 * (p . q) >= 4.
    # This simplifies to 2 >= 2 * (p . q), or p . q <= 1.

    # Therefore, the task is to find the largest subset of the 1320 candidate points
    # such that for any two distinct points p, q in the subset, their dot product p . q <= 1.
    # This is equivalent to finding a Maximum Independent Set in a graph where vertices are
    # candidate points, and an edge (p,q) exists if p.q == 2 (i.e., they conflict).
    # The prompt explicitly states: "This specific problem is known to yield 593 points."

    # 1. Generate all 1320 candidate points:
    candidate_points_list = []
    for indices in combinations(range(d), 3):  # Choose 3 positions for non-zero coordinates
        for signs in product([-1, 1], repeat=3): # Assign ±1 to these 3 positions
            p = np.zeros(d, dtype=np.int64)
            for i, idx in enumerate(indices):
                p[idx] = signs[i]
            candidate_points_list.append(p)
    
    # Convert list to numpy array for efficient vectorized operations
    candidate_points = np.array(candidate_points_list)
    num_candidates = len(candidate_points)

    # 2. Build the conflict graph (adjacency list)
    # An edge (i, j) exists if candidate_points[i] and candidate_points[j] conflict (p.q == 2)
    adj_list = [[] for _ in range(num_candidates)]
    for i in range(num_candidates):
        for j in range(i + 1, num_candidates): # Check each pair once
            if np.dot(candidate_points[i], candidate_points[j]) == 2:
                adj_list[i].append(j)
                adj_list[j].append(i) # Graph is undirected

    # 3. Implement an improved greedy algorithm (minimum-degree heuristic)
    # This heuristic often performs better for finding large independent sets.
    # It prioritizes adding points that conflict with the fewest other *available* points.
    # Deterministic tie-breaking is applied by choosing the point with the smallest original index.
    
    selected_points_indices = []
    available_mask = np.ones(num_candidates, dtype=bool) # True if point is still available for selection

    while np.any(available_mask):
        # Get indices of currently available points
        available_indices = np.where(available_mask)[0]
        
        min_degree = num_candidates + 1 # Initialize with a value larger than any possible degree
        best_point_idx = -1

        # Calculate degrees for available points within the induced subgraph
        # and find the point with the minimum degree (with deterministic tie-breaking)
        for current_idx in available_indices:
            # Degree of current_idx in the subgraph of available points
            degree_in_subgraph = sum(available_mask[neighbor] for neighbor in adj_list[current_idx])
            
            if degree_in_subgraph < min_degree:
                min_degree = degree_in_subgraph
                best_point_idx = current_idx
            elif degree_in_subgraph == min_degree:
                # Deterministic tie-breaking: choose the point with the smallest original index
                if current_idx < best_point_idx:
                    best_point_idx = current_idx
        
        # Add the selected point to our set
        selected_points_indices.append(best_point_idx)
        
        # Mark the selected point as unavailable
        available_mask[best_point_idx] = False
        
        # Mark all points that conflict with the selected point as unavailable
        # (they cannot be in the independent set with best_point_idx)
        for neighbor_idx in adj_list[best_point_idx]:
            available_mask[neighbor_idx] = False

    # Convert the list of selected point indices back to actual points
    final_points = candidate_points[selected_points_indices]
    
    return final_points

# EVOLVE-BLOCK-END