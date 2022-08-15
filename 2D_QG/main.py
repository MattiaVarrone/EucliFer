import numpy as np
from Triangulation import *

N = 100
iter = 1000
l = Manifold(N)
l.sweep(iter)

adj = l.adj + 1
next_l = [next_(i)+1 for i in range(3*N)]
prev_l = [prev_(i)+1 for i in range(3*N)]

# this is the object to be used by Mathematica
map = list(zip(next_l, prev_l, adj))


