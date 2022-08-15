from Triangulation import *

N = 10
iter = 2000

l = Manifold(N)
l.sweep(iter, beta=1, strategy=['flip', 'ising'])
print(l.sigma)
