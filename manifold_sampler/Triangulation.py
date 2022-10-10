from Action import *
from Graph_utils import *

rng = np.random.default_rng()


# counter-clockwise circling of vertices to compute plaquettes and angle deficiency
def circle_vertex(adj, sign, i):
    alpha = theta[i % 3] - theta[adj[i] % 3] + np.pi
    U = sign[i] * paral_trans(alpha / 2)
    def_triangles = 6 - 1

    j = prev_(adj[i])
    while j != i:
        alpha = theta[j % 3] - theta[adj[j] % 3] + np.pi
        U_1 = sign[j] * paral_trans(alpha / 2)
        U = np.matmul(U, U_1)
        def_triangles -= 1
        j = prev_(adj[j])

    return U, def_triangles


class Manifold:
    # Simplicial Manifold: created by piecing triangles together with the topology of a sphere

    def __init__(self, N, comb=[0, 1, 2]):
        self.N = N
        self.adj = fan_triangulation(N)

        self.phi = np.zeros(N)
        self.sigma = np.ones(N)

        self.A = np.zeros(3 * N)  # variable gauge link
        self.sign = fan_sign(self.adj, comb)  # gauge link sign corresponding to edges
        self.D = Dirac_operator(self.adj, self.sign, A=self.A)

    def random_update(self, beta, strategy):
        if 'gravity' in strategy:
            random_side = rng.integers(0, len(self.adj))
            self.flip_edge(random_side, beta, strategy=strategy)
        if 'scalar' in strategy:
            random_centre = rng.integers(0, self.N)
            self.phi = self.update_field(random_centre, self.phi, phi_range, action=S_phi, beta=beta)
        if 'ising' in strategy:
            random_centre = rng.integers(0, self.N)
            self.update_spin(random_centre, beta)
        if 'spinor_inter' in strategy:
            random_side = rng.integers(0, len(self.adj))
            self.A = self.update_gauge(random_side, self.A, A_range, action=S_psi_inter, beta=beta, matt_field=self.psi)  ### TO BE CHANGED

    def sweep(self, n_sweeps, beta, strategy):
        n = n_sweeps * 3 * self.N
        for _ in range(n):
            self.random_update(beta, strategy)

    def flip_edge(self, i, beta, strategy):
        if self.adj[i] != next_(i) and self.adj[i] != prev_(i):
            # flipping an edge that is adjacent to the same triangle on both sides makes no sense

            adj_new = np.copy(self.adj)
            A_new = np.copy(self.A)
            sign_new = np.copy(self.sign)
            D_new = np.copy(self.D)

            # (check p. 88 at https://hef.ru.nl/~tbudd/mct/mct_book.pdf for a labelling scheme)
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
            if 'ising' in strategy:
                S_old = S_sigma(self.adj, self.sigma, c1) + S_sigma(self.adj, self.sigma, c2)
                S_new = S_sigma(adj_new, self.sigma, c1) + S_sigma(adj_new, self.sigma, c2)
                dS += S_new - S_old
            if 'spinor_free' in strategy:
                # signs of parallel transporters are updated to ensure positive plaquette sign
                # re-initialise signs
                sign_new[i] = -self.sign[n]
                sign_new[k] = -self.sign[m]
                sign_new[m] = -sign_new[k]  # accounts for the case when j == n
                sign_new[l] = -self.sign[i]
                sign_new[j] = -sign_new[l]

                for edge in (i, j, k):
                    U, def_triangles = circle_vertex(adj_new, sign_new, edge)
                    trace = np.trace(U) / 2
                    trace_th = np.cos(def_triangles * np.pi / 6)

                    if np.isclose(trace, 0):  ### problems when trace = 0
                        s = np.sign(U[0, 1] / np.sin(def_triangles * np.pi / 6))  ### maybe should include minus sign
                    else:
                        s = np.sign(trace / trace_th)

                    sign_new[edge] *= s
                    sign_new[adj_new[edge]] = -sign_new[edge]

                D_new = Dirac_operator(adj_new, sign_new, D=D_new, triangles=(c1, c2), A=A_new)

                # action variation
                S_old = S_spinor(self.D)
                S_new = S_spinor(D_new)
                dS += S_new - S_old
            if 'spinor_inter' in strategy:
                # gauge links corresponding to adjacent sides must be opposites
                A_new[i] = -self.A[n]
                A_new[k] = -self.A[m]
                A_new[m] = -A_new[k]   # accounts for the case when j == n
                A_new[l] = -self.A[i]
                A_new[j] = -A_new[l]

                S_old = S_psi_inter(self.adj, self.psi, c1, self.A) + S_psi_inter(self.adj, self.psi, c2, self.A)
                S_new = S_psi_inter(adj_new, self.psi, c1, A_new) + S_psi_inter(adj_new, self.psi, c2, A_new)
                dS += S_new - S_old

            p = np.exp(-beta * dS)
            if p > np.random.rand():
                self.adj = adj_new
                self.A = A_new
                self.sign = sign_new
                self.D = D_new

    def update_field(self, c, field, field_range, action, beta, is_complex=False, add_field=None):
        field_new = np.copy(field)
        field_real = np.random.normal(size=field[0].shape)
        field_imaginary = np.random.normal(size=field[0].shape) * 1j if is_complex else 0
        field_new[c] += field_range * (field_real + field_imaginary)
        if add_field is not None:
            S_old = action(self.adj, field, c, add_field)
            S_new = action(self.adj, field_new, c, add_field)
        else:
            S_old = action(self.adj, field, c)
            S_new = action(self.adj, field_new, c)
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
            if 'spinor_free' in strategy:
                S += S_spinor(self.adj, self.sign)
            if 'spinor_inter' in strategy:
                S += S_psi_inter(self.adj, self.psi, c, self.A)
            if 'ising' in strategy:
                S += S_sigma(self.adj, self.sigma, c)
        return S