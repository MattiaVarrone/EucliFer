from Triangulation import *
from Plots import *

N = 50
iter = 20

beta = 0.5

l = Manifold(N)

l.sweep(iter, beta=beta, strategy=['spinor', 'gravity'])

print(l.psi)
print(l.A)

l.sweep(1, beta=beta, strategy=['spinor', 'gravity'])
