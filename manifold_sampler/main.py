import numpy as np

from Analysis_utils import *

N = 32
iter = 10

beta = 0.63


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
strategy = ['gravity', 'scalar']

range = range(1000)
Ss = []
Phis = []
Ms = []
Psis = []

fig, ax = plt.subplots(4)
for i in range:
    Ss.append(l.S_tot(strategy)/N)
    Phis.append(np.std(l.phi))
    Ms.append(np.average(l.sigma))
    Psis.append(np.std(l.psi))
    l.sweep(1, beta=beta, strategy=strategy)

ax[0].plot(range, Ss), ax[0].set_title("Action")
ax[1].plot(range, Phis), ax[1].set_title("Phi")
ax[2].plot(range, Ms), ax[2].set_title("Magn")
ax[3].plot(range, Psis), ax[3].set_title("Psi")
plt.show()



