from Triangulation import *
from Plots import *

N = 50
iter = 1

beta = 0.3

l = Manifold(N)

l.sweep(iter, beta=beta, strategy=['gravity', 'spinor'])
print(l.sigma+1)
l.sweep(10, beta=beta, strategy=['gravity', 'ising'])
print(l.sigma+1)
l.sweep(200, beta=beta, strategy=['gravity', 'ising'])
print(l.psi+1)


