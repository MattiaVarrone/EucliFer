from Analysis_utils import *

N = 20
n_sweeps = 5

beta = 0.63
K = 0
l = Manifold(N)
strategy = ['gravity', 'spinor_free']

l.sweep(n_sweeps, beta, strategy)
