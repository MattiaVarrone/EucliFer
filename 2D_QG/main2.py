from Triangulation import *
from Plots import *

N = 10
iter = 100

beta = 1

l = Manifold(N)

l.sweep(iter, beta=beta, strategy=['gravity'])
print(l.sigma+1)
psi_1 = np.random.normal(size=N)+1j*np.random.normal(size=N)
psi_2 = np.random.normal(size=N)+1j*np.random.normal(size=N)
psi = np.array([psi_1,psi_2]).reshape(N, 2)
A = np.random.uniform(size=3*N)
l.psi = psi
l.A = A
l.sweep(1, beta=beta, strategy=['spinor'])
print(l.psi-psi)


