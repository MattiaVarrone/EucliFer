from Analysis_utils import *

N = 10
n_sweeps = 100

beta = 0.63
l = Manifold(N)

a = np.copy(l.adj)
l.sweep(n_sweeps, beta, ["gravity", "spinor_free"])

D1 = Dirac_operator(l.adj, l.sign)


print(np.linalg.slogdet(D1))
