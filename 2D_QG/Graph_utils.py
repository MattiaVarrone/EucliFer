from collections import deque
import numpy as np
rng = np.random.default_rng()


class OddFanException(Exception):
    def __init__(self):
        print("n must be even")


def fan_triangulation(n):
    """Generates a fan-shaped triangulation of even size n."""
    if n % 2 == 1:
        raise OddFanException
    return np.array([[(i - 3) % (3 * n), i + 5, i + 4, (i + 6) % (3 * n), i + 2, i + 1] for i in range(0, 3 * n, 6)],
                    dtype=np.int32).flatten()


def is_fpf_involution(adj):
    """Test whether adj defines a fixed-point free involution."""
    for x, a in enumerate(adj):
        if a < 0 or a >= len(adj) or x == a or adj[a] != x:
            return False
    return True


def triangle_neighbours(adj,i):
    """Return the indices of the three neighboring triangles."""
    return [j//3 for j in adj[3*i:3*i+3]]


def connected_components(adj):
    """Calculate the number of connected components of the triangulation."""
    n = len(adj)//3  # the number of triangles
    component = np.full(n, -1, dtype=np.int32)   # array storing the component index of each triangle
    index = 0
    for i in range(n):
        if component[i] == -1:   # new component found, let us explore it
            component[i] = index
            queue = deque([i])   # use an exploration queue for breadth-first search
            while queue:
                for nbr in triangle_neighbours(adj, queue.pop()):
                    if component[nbr] == -1: # the neighboring triangle has not been explored yet
                        component[nbr] = index
                        queue.appendleft(nbr) # add it to the exploration queue
            index += 1
    return index


def next_(i):
    """Return the label of the side following side i in counter-clockwise direction."""
    return i - i % 3 + (i + 1) % 3


def prev_(i):
    """Return the label of the side preceding side i in counter-clockwise direction."""
    return i - i % 3 + (i - 1) % 3


def vertex_list(adj):
    """
    Return the number of vertices and an array `vertex` of the same size as `adj`,
    ↪
    such that `vertex[i]` is the index of the vertex at the start (in ccw order)␣
    ↪
    of the side labeled `i`.
    """
    vertex = np.full(len(adj), -1, dtype=np.int32)  # a side i that have not been visited vertex[i] == -1
    vert_index = 0  #
    for i in range(len(adj)):
        if vertex[i] == -1:
            side = i
            while vertex[side] == -1:  # find all sides that share the same vertex
                vertex[side] = vert_index
                side = next_(adj[side])
            vert_index += 1
    return vert_index, vertex


def vertex_neighbors_list(adj):
    '''Return a list `neighbors` such that `neighbors[v]` is a list of neighbors␣
    ↪of the vertex v.'''
    num_vertices, vertex = vertex_list(adj)
    neighbors = [[] for _ in range(num_vertices)]
    for i,j in enumerate(adj):
        neighbors[vertex[i]].append(vertex[j])
    return neighbors


def number_of_vertices(adj):
    '''Calculate the number of vertices in the triangulation.'''
    return vertex_list(adj)[0]


def is_sphere_triangulation(adj):
    '''Test whether adj defines a triangulation of the 2-sphere.'''
    if not is_fpf_involution(adj) or connected_components(adj) != 1:
        return False
    num_vert = number_of_vertices(adj)
    num_face = len(adj)//3
    num_edge = len(adj)//2
    # verify Euler's formula for the sphere
    return num_vert - num_edge + num_face == 2





