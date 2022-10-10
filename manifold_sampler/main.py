from Analysis_utils import *

N = 10 ### needs to be multiple of 4
n_sweeps = 1

beta = 0.63
l = Manifold(N)
D = Dirac_operator(l.adj, l.sign)

a = np.copy(l.adj)
l.sweep(n_sweeps, beta, ["gravity", "spinor_free"])

D1 = Dirac_operator(l.adj, l.sign)

"""
@timebudget
def D(n):
    for _ in range(n):
        Dirac_operator(l.adj, l.sign)
    return Dirac_operator(l.adj, l.sign)

@timebudget
def det(n):
    for _ in range(n):
        np.linalg.slogdet(D1)[0]
    return np.linalg.slogdet(D1)[0]

print(np.linalg.slogdet(D(100)))
print(det(100))
"""