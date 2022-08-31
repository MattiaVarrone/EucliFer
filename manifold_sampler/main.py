from Triangulation import *
from Plots import *
from Analysis_utils import *

N = 12
iter = 200

beta = 0.5

l = Manifold(N)

l.sweep(iter, beta=beta, strategy=[ 'gravity', 'spinor'])

print(l.psi)
print(l.A)
print(l.phi)
print(l.sigma)
