import numpy as np
import scipy
from Graph_utils import *

rng = np.random.default_rng()

# beta is the inverse strength of gravity
beta = 1

# scalar field params
phi_const = 1
phi_range = 1

# ising params
sigma_const = 1

# spinor field params
psi_const = 1
psi_range = 0.7
A_range = 1
gamma0 = np.array([[1, 0], [0, -1]])  ### are these gamma matrices in euclidean space?
gamma1 = np.array([[0, 1], [1, 0]])
eps = np.array([[0, -1], [1, 0]])


def S_phi(adj, phi, c):
    S = 0
    for i in range(3):
        adj_c = adj[3 * c + i] // 3
        S += phi_const * (phi[c] - phi[adj_c]) ** 2
    return S


def U(A_i):  ### check how to calculate parallel transporter
    U = scipy.linalg.expm(A_i * eps)  ###complex exponential?
    return U


def S_psi(adj, psi, c, A):
    d_psi_x, d_psi_y = 0, 0
    for i in range(3):
        j = 3 * c + i
        theta = 2 * i * np.pi / 3
        adj_c = adj[j] // 3

        d_psi = (np.matmul(U(A[j]), psi[adj_c]) - psi[c])  ### check how to calculate parallel transporter
        d_psi_x += d_psi * np.cos(theta)
        d_psi_y += d_psi * np.sin(theta)

    D_psi = np.matmul(gamma0, d_psi_x) + np.matmul(gamma1, d_psi_y)
    psi_bar = np.conj(np.matmul(gamma0, psi[c]))  ### check how to calc psi_bar
    S = psi_const * np.abs(np.matmul(psi_bar, D_psi))  ### why is the action complex?
    return S


def S_A(adj, A, i, psi):
    return S_psi(adj, psi, i // 3, A)


def S_sigma(adj, sigma, c):
    S = 0
    for i in range(3):
        adj_c = adj[3 * c + i] // 3
        S += - sigma_const * sigma[c] * sigma[adj_c]
    return S


def update_field(i, adj, field, field_range, action, beta, is_complex=False, add_field=None):
    field_new = np.copy(field)
    field_real = np.random.normal(size=field[0].shape)
    field_imaginary = np.random.normal(size=field[0].shape) * 1j if is_complex else 0
    field_new[i] += field_range * (field_real + field_imaginary)
    if add_field is not None:
        S_old = action(adj, field, i, add_field)
        S_new = action(adj, field_new, i, add_field)
    else:
        S_old = action(adj, field, i)
        S_new = action(adj, field_new, i)

    p = np.exp(-beta * (S_new - S_old))
    if p > np.random.rand():
        field = field_new
    return field


def update_gauge(i, adj, gauge, gauge_range, action, beta, matt_field):
    gauge_new = np.copy(gauge)
    gauge_diff = np.random.normal(size=gauge[0].shape)
    gauge_new[i] += gauge_range * gauge_diff

    c = i//3
    S_old = action(adj, matt_field, c, gauge)
    S_new = action(adj, matt_field, c, gauge_new)

    p = np.exp(-beta * (S_new - S_old))
    if p > np.random.rand():
        gauge = gauge_new
    return gauge


class Manifold:
    # Simplicial Manifold: created by piecing triangles together with the topology of a sphere

    def __init__(self, N):
        self.N = N
        self.adj = fan_triangulation(N)

        self.psi = np.zeros((N, 2), dtype=np.complex_)
        self.A = np.zeros(3 * N)
        self.phi = np.zeros(N)
        self.sigma = np.ones(N)

    def random_flip(self, b, strategy, fast=False):
        if 'gravity' in strategy:
            random_side = rng.integers(0, len(self.adj))
            self.flip(random_side, b, fast=fast)
        if 'scalar' in strategy:
            random_centre = rng.integers(0, self.N)
            self.phi = update_field(random_centre, self.adj, self.phi, phi_range, action=S_phi, beta=b)
        if 'spinor' in strategy:
            random_centre, random_side = rng.integers(0, self.N), rng.integers(0, len(self.adj))
            self.psi = update_field(random_centre, self.adj, self.psi, psi_range,
                                    action=S_psi, beta=b, is_complex=True, add_field=self.A)
            self.A = update_gauge(random_side, self.adj, self.A, A_range, action=S_psi, beta=b, matt_field=self.psi)
        if 'ising' in strategy:
            random_side = rng.integers(0, len(self.adj))
            self.vary_sigma(random_side, b)

    def sweep(self, n_sweeps, beta, strategy, fast=False):
        n = n_sweeps * 3 * self.N
        for _ in range(n):
            self.random_flip(beta, strategy, fast=fast)

    def flip(self, i, b, fast=False):
        if self.adj[i] == next_(i) or self.adj[i] == prev_(i):
            return False
        # flipping an edge that is adjacent to the same triangle on both sides makes no sense

        adj_new = np.copy(self.adj)
        A_new = np.copy(self.A)

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

        if fast:
            dS = 0
        else:
            # gauge links corresponding to adjacent sides must be opposites
            A_new[i] = -self.A[n]
            A_new[k] = -self.A[m]
            A_new[l] = -self.A[i]
            A_new[j] = -self.A[l]

            # centres of triangles involved
            c1 = i // 3
            c2 = k // 3

            # calculate action variation
            S_old_psi = S_psi(self.adj, self.psi, c1, self.A) + S_psi(self.adj, self.psi, c2, self.A)
            S_new_psi = S_psi(adj_new, self.psi, c1, A_new) + S_psi(adj_new, self.psi, c2, A_new)
            S_old_phi = S_phi(self.adj, self.phi, c1) + S_phi(self.adj, self.phi, c2)
            S_new_phi = S_phi(adj_new, self.phi, c1) + S_phi(adj_new, self.phi, c2)
            S_old_sigma = S_sigma(self.adj, self.sigma, c1) + S_sigma(self.adj, self.sigma, c2)
            S_new_sigma = S_sigma(adj_new, self.sigma, c1) + S_sigma(adj_new, self.sigma, c2)

            dS_psi = S_new_psi - S_old_psi
            dS_phi = S_new_phi - S_old_phi
            dS_sigma = S_new_sigma - S_old_sigma
            dS = dS_psi + dS_phi + dS_sigma

        p = np.exp(-b * dS)
        if p > np.random.rand():
            self.adj = adj_new
            self.A = A_new
        return True

    def vary_sigma(self, i, b):
        c = i // 3
        sigma_new = np.copy(self.sigma)
        sigma_new[c] *= -1
        S_old = S_phi(self.adj, self.sigma, c)
        S_new = S_phi(self.adj, sigma_new, c)
        p = np.exp(-b * (S_new - S_old))

        if p > np.random.rand():
            self.sigma = sigma_new
