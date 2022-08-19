from Triangulation import *
from Plots import *

N = 10
iter = 1

l = Manifold(N)

l.sweep(iter, beta=1, strategy=['gravity', 'scalar'])
print(l.phi)
l.sweep(10, beta=1, strategy=['gravity', 'scalar'])
print(l.phi)
l.sweep(50, beta=1, strategy=['gravity', 'scalar'])
print(l.phi)

