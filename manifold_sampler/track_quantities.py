import numpy as np

from Analysis_utils import *

N = 42
beta = 0.216
strategy = ['gravity', 'ising']

l = Manifold(N)

range = range(100)
Ss = []
Phis = []
Ms = []

#_K = 0

fig, ax = plt.subplots(3)
for _ in range:
    Ss.append(l.S_tot(strategy)/N)
    #Phis.append(np.std(l.phi))
    Ms.append(np.average(l.sigma))
    l.sweep(1, beta=beta, strategy=strategy)

fig.suptitle("Lattice size = " + str(N) + f', {beta = }')
ax[0].plot(range, Ss), ax[0].set_title("Action")
#ax[1].plot(range, Phis), ax[1].set_title("Phi")
ax[2].plot(range, Ms), ax[2].set_title("Magn")
plt.show()