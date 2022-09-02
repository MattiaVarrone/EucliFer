import numpy as np

from Analysis_utils import *

N = 12
iter = 10

beta = 0.5


masses = np.geomspace(0.1, 100, 3)
"""
for m in masses:
    _mass = m
    l = Manifold(N)
    l.sweep(iter, beta=beta, strategy=['gravity', 'spinor'])
    std = np.std(l.psi)
    print(f'{m = } : {std = }')
"""
_mass = 0
l = Manifold(N)
Ss = []
range = range(10000)

for _ in range:
    l.sweep(1, beta=beta, strategy=['gravity', 'ising'])
    Ss.append(l.S)

plt.plot(range, Ss)
plt.show()



