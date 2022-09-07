from Analysis_utils import *

N = 12
n_sweeps = 20

beta = 0.63
K = 0
l = Manifold(N)
strategy = ['gravity', 'spinor_free']

l.sweep(n_sweeps, beta, strategy)
l.sweep(n_sweeps, beta, strategy)

print(l.psi)
print(l.sign)