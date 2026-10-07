import numpy as np
from scipy.spatial import Delaunay

def concave_hull(points_array):
    """
    Computes ordered boundary coordinates for alpha = 0 
    using all points via Delaunay Triangulation.
    
    Parameters:
        points_array: List or array of tuples [(x1, y1), (x2, y2), ...]
    Returns:
        Ordered numpy array of shape (N, 2) representing boundary points.
    """
    pts = np.asarray(points_array)
    if len(pts) < 4:
        return None
    tri = Delaunay(pts)
    
    # Extract all edges from triangles
    # Each triangle yields 3 edges: (0,1), (1,2), (2,0)
    simplices = tri.simplices
    edges = np.vstack([
        simplices[:, [0, 1]],
        simplices[:, [1, 2]],
        simplices[:, [2, 0]]
    ])
    
    # Sort edge endpoints to ensure uniqueness check: (u, v) where u < v
    sorted_edges = np.sort(edges, axis=1)
    
    # Find boundary edges (edges that appear exactly once across all triangles)
    edges_unique, counts = np.unique(sorted_edges, axis=0, return_counts=True)
    boundary_edges = edges_unique[counts == 1]
    
    # Order boundary edges sequentially into a closed contour
    adjacency = {}
    for u, v in boundary_edges:
        adjacency.setdefault(u, []).append(v)
        adjacency.setdefault(v, []).append(u)
        
    start_node = boundary_edges[0][0]
    ordered_indices = [start_node]
    curr = start_node
    prev = None
    
    while True:
        neighbors = adjacency[curr]
        next_node = neighbors[0] if neighbors[0] != prev else neighbors[1]
        if next_node == start_node:
            break
        ordered_indices.append(next_node)
        prev, curr = curr, next_node
        
    # Return array of coordinates in order
    return pts[ordered_indices]