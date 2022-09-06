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
_mass = 0
### figure out what this is in 2d euclidean space
gamma1 = np.array([[0, 1], [1, 0]])  ### refer to Zuber
gamma2 = np.array([[0, -1j], [1j, 0]])  ### are these the correct gamma matrices in euclidean space?

id = np.identity(2)
eps = np.array([[0, 1], [-1, 0]])


def S_phi(adj, phi, c):
    S = 0
    for i in range(3):
        adj_c = adj[3 * c + i] // 3
        S += phi_const * (phi[c] - phi[adj_c]) ** 2
    return S


def paral_trans(A_i):  ### check how to calculate parallel transporter
    U = np.cos(A_i) * id + np.sin(A_i) * eps
    return U


def S_psi(adj, psi, c, A):
    d_psi_x, d_psi_y = 0, 0

    for i in range(3):
        j = 3 * c + i
        theta = 2 * i * np.pi / 3
        adj_c = adj[j] // 3

        # we need parallel transport to take derivatives
        d_psi = (np.matmul(paral_trans(A[j]), psi[adj_c]) - psi[c])  ### check how to calculate parallel transporter
        d_psi_x += d_psi * np.cos(theta)
        d_psi_y += d_psi * np.sin(theta)

    D_psi = np.matmul(id + gamma1, d_psi_x) / 2 + np.matmul(id + gamma2, d_psi_y) / 2
    psi_bar = np.conj(np.matmul(gamma1, psi[c]))  ### check how to calc psi_bar
    S = - psi_const * np.imag(np.matmul(psi_bar, D_psi)) + _mass * np.real(np.matmul(psi_bar, psi[c]))
    return S


###check action calculation thoroughly     ### check why psi tends to diverge
### check how to obtain real action: take imag() or use hermit conjugate


def S_sigma(adj, sigma, c):
    S = 0
    for i in range(3):
        adj_c = adj[3 * c + i] // 3
        S += - sigma_const * sigma[c] * sigma[adj_c]
    return S


class Manifold:
    # Simplicial Manifold: created by piecing triangles together with the topology of a sphere

    def __init__(self, N):
        self.N = N
        self.adj = fan_triangulation(N)

        self.psi = np.zeros((N, 2), dtype=np.complex_)
        self.A = np.zeros(3 * N)
        self.phi = np.zeros(N)
        self.sigma = np.ones(N)

    def random_update(self, beta, strategy):
        if 'gravity' in strategy:
            random_side = rng.integers(0, len(self.adj))
            self.flip_edge(random_side, beta, strategy=strategy)
        if 'scalar' in strategy:
            random_centre = rng.integers(0, self.N)
            self.phi = self.update_field(random_centre, self.phi, phi_range, action=S_phi, beta=beta)
        if 'spinor' in strategy:
            random_centre, random_side = rng.integers(0, self.N), rng.integers(0, len(self.adj))
            self.psi = self.update_field(random_centre, self.psi, psi_range,
                                         action=S_psi, beta=beta, is_complex=True, add_field=self.A)
            self.A = self.update_gauge(random_side, self.A, A_range, action=S_psi, beta=beta, matt_field=self.psi)
        if 'ising' in strategy:
            random_centre = rng.integers(0, self.N)
            self.update_spin(random_centre, beta)

    def sweep(self, n_sweeps, beta, strategy):
        n = n_sweeps * 3 * self.N
        for _ in range(n):
            self.random_update(beta, strategy)

    def flip_edge(self, i, beta, strategy):
        if self.adj[i] != next_(i) and self.adj[i] != prev_(i):
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

            # Action variation due to gravity is 0
            dS = 0

            # centres of triangles involved
            c1 = i // 3
            c2 = k // 3

            if 'scalar' in strategy:
                S_old = S_phi(self.adj, self.phi, c1) + S_phi(self.adj, self.phi, c2)
                S_new = S_phi(adj_new, self.phi, c1) + S_phi(adj_new, self.phi, c2)
                dS += S_new - S_old
            if 'spinor' in strategy:
                # gauge links corresponding to adjacent sides must be opposites
                A_new[i] = -self.A[n]
                A_new[k] = -self.A[m]
                A_new[l] = -self.A[i]
                A_new[j] = -A_new[l]

                S_old = S_psi(self.adj, self.psi, c1, self.A) + S_psi(self.adj, self.psi, c2, self.A)
                S_new = S_psi(adj_new, self.psi, c1, A_new) + S_psi(adj_new, self.psi, c2, A_new)
                dS += S_new - S_old
            if 'ising' in strategy:
                S_old = S_sigma(self.adj, self.sigma, c1) + S_sigma(self.adj, self.sigma, c2)
                S_new = S_sigma(adj_new, self.sigma, c1) + S_sigma(adj_new, self.sigma, c2)
                dS += S_new - S_old

            p = np.exp(-beta * dS)
            if p > np.random.rand():
                self.adj = adj_new
                self.A = A_new

    def update_field(self, i, field, field_range, action, beta, is_complex=False, add_field=None):
        field_new = np.copy(field)
        field_real = np.random.normal(size=field[0].shape)
        field_imaginary = np.random.normal(size=field[0].shape) * 1j if is_complex else 0
        field_new[i] += field_range * (field_real + field_imaginary)
        if add_field is not None:
            S_old = action(self.adj, field, i, add_field)
            S_new = action(self.adj, field_new, i, add_field)
        else:
            S_old = action(self.adj, field, i)
            S_new = action(self.adj, field_new, i)
        dS = S_new - S_old

        p = np.exp(-beta * dS)
        if p > np.random.rand():
            field = field_new
        return field

    def update_gauge(self, i, gauge, gauge_range, action, beta, matt_field):
        gauge_new = np.copy(gauge)
        gauge_diff = np.random.normal(size=gauge[0].shape)
        gauge_new[i] += gauge_range * gauge_diff  # proposed update
        gauge_new[self.adj[i]] = -gauge_new[i]  # adjacent link must be updated as well

        c = i // 3
        S_old = action(self.adj, matt_field, c, gauge)
        S_new = action(self.adj, matt_field, c, gauge_new)
        dS = S_new - S_old

        p = np.exp(-beta * dS)
        if p > np.random.rand():
            gauge = gauge_new
        return gauge

    def update_spin(self, c, beta):
        sigma_new = np.copy(self.sigma)
        sigma_new[c] *= -1
        S_old = S_phi(self.adj, self.sigma, c)
        S_new = S_phi(self.adj, sigma_new, c)
        dS = S_new - S_old

        p = np.exp(-beta * dS)
        if p > np.random.rand():
            self.sigma = sigma_new

    def S_tot(self, strategy):
        S = 0
        for c in range(self.N):
            if 'scalar' in strategy:
                S += S_phi(self.adj, self.phi, c)
            if 'spinor' in strategy:
                S += S_psi(self.adj, self.psi, c, self.A)
            if 'ising' in strategy:
                S += S_sigma(self.adj, self.sigma, c)
        return S
