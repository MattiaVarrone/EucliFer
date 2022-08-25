from Triangulation import *
from Plots import *

N = 50
iter = 100

beta = 0.5

l = Manifold(N)

l.sweep(iter, beta=beta, strategy=['spinor', 'gravity'])

print(l.psi)
print(l.A)
