from Analysis_utils import *

N = 12 ### needs to be multiple of 4
n_sweeps = 1

beta = 0.63
K = 0
l = Manifold(N)
strategy = ['gravity', 'spinor_free']
print("first")
l.flip_edge(1, beta, strategy)

print("finished")