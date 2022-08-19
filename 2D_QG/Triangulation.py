import numpy as np
from Graph_utils import *

rng = np.random.default_rng()

phi_const = 1
phi_range = 1
psi_range = 1
sigma_const = 1
beta = 1


def S_phi(adj, phi, c):
    S = 0
    for i in range(3):
        adj_c = adj[3 * c + i] // 3
        S += phi_const * (phi[c] - phi[adj_c]) ** 2
    return S


def S_psi(adj, phi, c):
    return 0


def S_sigma(adj, sigma, c):
    S = 0
    for i in range(3):
        adj_c = adj[3 * c + i] // 3
        S += - sigma_const * sigma[c] * sigma[adj_c]
    return S


def update_field(i, adj, field, field_range, S, b):
    c = i // 3
    field_new = np.copy(field)
    field_new[c] += field_range * np.random.normal(field[0].shape)
    S_old = S(adj, field, c)
    S_new = S_phi(adj, field_new, c)
    p = np.exp(-b * (S_new - S_old))

    if p > np.random.rand():
        field = field_new
    return field


class Manifold:
    # Simplicial Manifold: created by piecing triangles together with the topology of a sphere

    def __init__(self, N):
        self.N = N
        self.adj = fan_triangulation(N)
        self.vert_n, self.vert = vertex_list(self.adj)
        self.phi = np.zeros(N)
        self.psi = np.zeros((N, 2))
        self.sigma = np.ones(N)

    def random_flip(self, b, strategy=['gravity']):
        if 'gravity' in strategy:
            random_side = rng.integers(0, len(self.adj))
            self.flip(random_side, b)
        if 'scalar' in strategy:
            random_side = rng.integers(0, len(self.adj))
            self.phi = update_field(random_side, self.adj, self.field, phi_range, S_phi, b)
        if 'spinor' in strategy:
            random_side = rng.integers(0, len(self.adj))
            self.psi = update_field(random_side, self.adj, self.psi, psi_range, S_psi, b)
        if 'ising' in strategy:
            random_side = rng.integers(0, len(self.adj))
            self.vary_sigma(random_side, b)

    def sweep(self, n_sweeps, beta=beta, strategy=['gravity']):
        n = n_sweeps * 3 * self.N
        for _ in range(n):
            self.random_flip(beta, strategy)

    def flip(self, i, b):
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

        c1 = i // 3
        c2 = k // 3

        S_old_phi = S_phi(self.adj, self.phi, c1) + S_phi(self.adj, self.phi, c2)
        S_new_phi = S_phi(adj_new, self.phi, c1) + S_phi(adj_new, self.phi, c2)
        S_old_sigma = S_sigma(self.adj, self.sigma, c1) + S_sigma(self.adj, self.sigma, c2)
        S_new_sigma = S_sigma(adj_new, self.sigma, c1) + S_sigma(adj_new, self.sigma, c2)

        dS_phi = S_new_phi - S_old_phi
        dS_sigma = S_new_sigma - S_old_sigma
        dS = dS_phi + dS_sigma

        p = np.exp(-b * dS)
        if p > np.random.rand():
            self.adj = adj_new
        return True

    def vary_phi(self, i, b):
        c = i // 3
        phi_new = np.copy(self.phi)
        phi_new[c] += phi_range * np.random.randn()
        S_old = S_phi(self.adj, self.phi, c)
        S_new = S_phi(self.adj, phi_new, c)
        p = np.exp(-b * (S_new - S_old))

        if p > np.random.rand():
            self.phi = phi_new

        """
            print('flipped')
        else: print('unchanged')
        print(f'{S_old = }' + f'{ S_new = }' + f'{ p = }')
        print()
        """

    def vary_sigma(self, i, b):
        c = i // 3
        sigma_new = np.copy(self.sigma)
        sigma_new[c] *= -1
        S_old = S_phi(self.adj, self.sigma, c)
        S_new = S_phi(self.adj, sigma_new, c)
        p = np.exp(-b * (S_new - S_old))

        if p > np.random.rand():
            self.sigma = sigma_new

    def T_trace(self, beta, n):
        return 0

