import numpy as np

# beta is the inverse strength of gravity
beta = 1

# scalar field params
phi_const = 1
phi_range = 1

# ising params
sigma_const = 1

# spinor field params
K = 0.3746
psi_range = 0.7
A_range = 1
_mass = 1/2

# parallel transport of free fermions
theta = [np.pi, np.pi / 3, 5 * np.pi / 3]

# dirac gamma matrices
gamma1 = np.array([[1, 0], [0, -1]])  ### are these the correct gamma matrices in euclidean space?
gamma2 = np.array([[0, 1], [1, 0]])   ### refer to Zuber

id = np.identity(2)
eps = np.array([[0, 1], [-1, 0]])


def S_phi(adj, phi, c):
    S = 0
    for i in range(3):
        adj_c = adj[3 * c + i] // 3
        S += phi_const * (phi[c] - phi[adj_c]) ** 2
    return S


def S_sigma(adj, sigma, c):
    S = 0
    for i in range(3):
        adj_c = adj[3 * c + i] // 3
        S += - sigma_const * sigma[c] * sigma[adj_c]
    return S


def paral_trans(A):  ### check how to calculate parallel transporter
    U = np.cos(A) * id + np.sin(A) * eps
    return U


def S_psi_inter(adj, psi, c, A):
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
    psi_bar = np.conj(np.matmul(psi[c], eps))  ### check how to calc psi_bar
    S = K * np.imag(np.matmul(psi_bar, D_psi)) + _mass * np.real(np.matmul(psi_bar, psi[c]))
    return S


def S_psi_free(adj, psi, c, sign):
    d_psi_x, d_psi_y = 0, 0

    for i in range(3):
        j = 3 * c + i
        adj_c = adj[j] // 3
        alpha = theta[i] - theta[adj[j] % 3] + np.pi
        U = sign[j] * paral_trans(alpha/2)            # factor of 1/2 accounts for spinor transport
                                                      # (I should store the possible values of U)
        d_psi = (np.matmul(U, psi[adj_c]) - psi[c])
        d_psi_x += d_psi * np.cos(theta[i])
        d_psi_y += d_psi * np.sin(theta[i])

    D_psi = np.matmul(id + gamma1, d_psi_x) / 2 + np.matmul(id + gamma2, d_psi_y) / 2
    psi_bar = np.matmul(eps, psi[c])      ### check how to calc psi_bar
    S = -K * np.matmul(psi_bar, D_psi) + _mass * 2 * psi[c, 0] * psi[c, 1]
    return np.real(S)

###check action calculation thoroughly     ### check why psi tends to diverge
### check how to obtain real action: take imag() or use hermit conjugate