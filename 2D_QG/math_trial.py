import numpy as np

rng = np.random.default_rng()
import matplotlib.pylab as plt

phi_const = 1
phi_range = 1


def S_phi(adj, phi, c):
    S = 0
    for i in range(3):
        adj_c = adj[3 * c + i] // 3
        S += phi_const * (phi[c] - phi[adj_c]) ** 2
    return S


class Manifold:
    # Manifolds build by piecing together 2-simplices (triangles)

    def __init__(self, N):
        self.N = N
        self.adj = fan_triangulation(N)
        self.vert_n, self.vert = vertex_list(self.adj)
        self.phi = np.random.randn(N)

    def flip_edge(self, i):
        if self.adj[i] == next_(i) or self.adj[i] == prev_(i):
            return False
        # flipping an edge that is adjacent to the same triangle on both sides makes no sense

        j = prev_(i)
        k = self.adj[i]
        l = prev_(k)
        n = self.adj[l]
        self.adj[i] = n  # it is important that we first update
        self.adj[n] = i  # these adjacencies, before determining m,
        m = self.adj[j]  # to treat the case j == n appropriately
        self.adj[k] = m
        self.adj[m] = k
        self.adj[j] = l
        self.adj[l] = j
        return True

    def random_flip(self):
        random_side = rng.integers(0, len(self.adj))
        self.flip(random_side)

    def evolve(self, n):
        for _ in range(n):
            self.random_flip()
        self.vert = vertex_list(self.adj)[1]

    def flip(self, i):
        if self.adj[i] == next_(i) or self.adj[i] == prev_(i):
            return False
        # flipping an edge that is adjacent to the same triangle on both sides makes no sense

        adj_new = np.copy(self.adj)

        j = prev_(i)
        k = adj_new[i]
        l = prev_(k)
        n = adj_new[l]
        adj_new[i] = n  # it is important that we first update
        adj_new[n] = i  # these adjacencies, before determining m,
        m = adj_new[j]  # to treat the case j == n appropriately
        adj_new[k] = m
        adj_new[m] = k
        adj_new[j] = l
        adj_new[l] = j

        c1 = i//3
        c2 = k//3
        S_old = S_phi(self.adj, self.phi, c1) + S_phi(self.adj, self.phi, c2)
        S_new = S_phi(adj_new, self.phi, c1) + S_phi(adj_new, self.phi, c2)
        p = np.exp(-0.1*(S_new - S_old))
        if p > np.random.rand():
            self.adj = adj_new

            print('flipped')
        else: print('unchanged')
        print(f'{S_old = }' + f'{ S_new = }' + f'{ p = }')
        print()

        return True

    def vary_phi(self, i):
        c = i//3
        phi_new = self.phi[c] + phi_range*np.random.randn
        S_old = S_phi(self.adj, self.phi, c)
        S_new = S_phi(self.adj, phi_new, c)
        p = np.exp(-(S_new - S_old))

        if p > np.random.rand():
            self.phi = phi_new


def fan_triangulation(n):
    """Generates a fan-shaped triangulation of even size n."""
    return np.array([[(i - 3) % (3 * n), i + 5, i + 4, (i + 6) % (3 * n), i + 2, i + 1] for i in range(0, 3 * n, 6)],
                    dtype=np.int32).flatten()


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

print("we made it")

a = 4