# EVOLVE-BLOCK-START
import numpy as np
from itertools import combinations, product

def kissing_number11() -> np.ndarray:
    """
    Constructs a large set of 11D points with integer coordinates satisfying the problem's geometric condition.

    The strategy is based on finding a large independent set in a "conflict graph".
    The vertices of the graph are the 1320 points in 11D with exactly three +/-1 coordinates.
    For any such point, the squared norm is 3. The condition M_N^2 < m_D^2 requires m_D^2 >= 4.
    An edge exists between two points if their squared distance is less than 4 (which occurs when
    their dot product is 2). The problem is then to find a maximum independent set in this graph.

    This function implements a minimum-degree greedy algorithm, a powerful heuristic for this NP-hard problem.
    At each step, it selects a point that has the minimum number of conflicts with the remaining
    available points, adds it to the solution, and removes it and its conflicting neighbors.

    Returns:
        points: np.ndarray representing the selected set of points.
    """
    d = 11
    num_nonzero = 3
    
    # 1. Generate all 1320 candidate points and sort for determinism.
    candidate_points_list = []
    for positions in combinations(range(d), num_nonzero):
        for signs in product([-1, 1], repeat=num_nonzero):
            point = np.zeros(d, dtype=np.int64)
            point[list(positions)] = signs
            candidate_points_list.append(tuple(point))
    
    candidate_points_list.sort()
    points = np.array(candidate_points_list, dtype=np.int64)
    N = len(points)

    # 2. Build the conflict graph (adjacency list representation).
    # An edge exists if ||p-q||^2 < 4, which is equivalent to p.dot(q) > 1.
    adj = [[] for _ in range(N)]
    for i in range(N):
        for j in range(i + 1, N):
            # Efficiently check for dot product of 2.
            # This happens iff supports overlap by 2 and signs match on the overlap.
            if np.dot(points[i], points[j]) > 1:
                adj[i].append(j)
                adj[j].append(i)

    # 3. Iteratively build the independent set using the minimum-degree heuristic.
    solution_indices = []
    is_removed = np.zeros(N, dtype=bool)
    degrees = np.array([len(adj[i]) for i in range(N)])
    num_remaining = N

    while num_remaining > 0:
        # a. Find a remaining vertex with the minimum current degree.
        # The np.where clause handles the case where all remaining nodes have degree 0.
        min_deg = -1
        v_min_deg = -1
        
        active_indices = np.where(~is_removed)[0]
        if len(active_indices) == 0:
            break

        min_deg_val = np.min(degrees[active_indices])
        # Pick the first one in sorted order for determinism
        v_min_deg = np.where((~is_removed) & (degrees == min_deg_val))[0][0]

        # b. Add the selected vertex to our solution set.
        solution_indices.append(v_min_deg)
        
        # c. Remove the vertex and all its neighbors.
        # Create a list of vertices to remove to avoid issues with iterating while modifying.
        to_remove = [v_min_deg]
        for neighbor in adj[v_min_deg]:
            if not is_removed[neighbor]:
                to_remove.append(neighbor)
        
        for v_rem in to_remove:
            if not is_removed[v_rem]:
                is_removed[v_rem] = True
                num_remaining -= 1
                # Update the degrees of the neighbors of the removed vertex.
                for neighbor_of_removed in adj[v_rem]:
                    if not is_removed[neighbor_of_removed]:
                        degrees[neighbor_of_removed] -= 1
    
    return points[solution_indices]

# EVOLVE-BLOCK-END