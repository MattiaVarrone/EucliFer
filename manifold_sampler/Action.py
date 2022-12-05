import numpy as np

# beta is the inverse strength of gravity
beta = 1

# scalar field params
phi_const = 1
phi_range = 1

# ising params
sigma_const = 1

# spinor field params
_K = 0.3746166
psi_range = 0.7
A_range = 1
_mass = 1 / 2

# parallel transport of free spinors
theta = [np.pi, np.pi / 3, 5 * np.pi / 3]

# dirac gamma matrices
gamma1 = np.array([[1, 0], [0, -1]])  ### are these the correct gamma matrices in euclidean space?
gamma2 = np.array([[0, 1], [1, 0]])  ### refer to Zuber

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


def S_spinor(D):
    logdet = np.linalg.slogdet(D)
    return 1 / 2 * logdet[1]



def Dirac_operator(adj, sign, D=None, triangles=None, A=None):
    N = len(adj) // 3

    if D is None:
        D = np.zeros(shape=(2 * N, 2 * N))
    D1 = np.copy(D)
    if triangles is None:
        triangles = range(N)

    ### first DELETE ROWS TO BE UPDATED

    if A is None:
        A = np.zeros(3 * N)

    for c in triangles:
        i = 2 * c
        D[i:i + 2].fill(0)  # re-initializes row before making changes
        D1[i:i + 2].fill(0)
        D[i, i + 1] = _mass
        D[i + 1, i] = -_mass
        D1[i, i + 1] = _mass
        D1[i + 1, i] = -_mass

        for k in range(3):
            edge = 3 * c + k
            q = adj[edge] % 3
            adj_c = adj[edge] // 3
            j = adj_c * 2

            sink, sinq, cosk, cosq = np.sin(theta[k] / 2), np.sin(theta[q] / 2), np.cos(theta[k] / 2), np.cos(
                theta[q] / 2)
            H1 = -_K * sign[edge] * np.array([[cosk * cosq, -sinq * cosk], [-sink * cosq, sink * sinq]])

            alpha = theta[k] - theta[adj[edge] % 3] + np.pi
            U = sign[edge] * paral_trans((alpha + A[edge]) / 2)  # factor of 1/2 accounts for spinor transport
            H_0 = 1 / 2 * np.matmul(id + np.cos(theta[k]) * gamma1 - np.sin(theta[k]) * gamma2, U)  ### should we subtract identity?
            H = -_K * np.matmul(eps, H_0)

            for a in range(2):
                for b in range(2):
                    D[i + a, j + b] += H[a, b]
                    D1[i + a, j + b] += H1[a, b]

    det = np.linalg.slogdet(D1)
    if det[0] == -1:
        print("sign(det) = -1")
    return D1
