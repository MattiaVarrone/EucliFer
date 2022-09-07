import sys
import unittest

import scipy

sys.path.append("../")
from manifold_sampler.Triangulation import *

strategy = ['gravity', 'spinor']
beta = 0.5


class TestGauge(unittest.TestCase):

    def test_adjacent(self):
        N = 50
        n_sweeps = 20
        m = Manifold(N)
        for _ in range(2):
            m.sweep(n_sweeps, beta, strategy)
            for i in range(len(m.adj)):
                self.assertEqual(m.A[i], -m.A[m.adj[i]])  ### RARELY DOES NOT WORK

    def test_paral_trans(self):
        A = 2
        np.testing.assert_allclose(scipy.linalg.expm(A * eps), paral_trans(A))


class TestPsi(unittest.TestCase):

    def test_mass_term_real(self):
        N = 50
        n_sweeps = 20
        m = Manifold(N)
        m.sweep(n_sweeps, beta, strategy)
        for c in range(N):
            psi = m.psi[c]
            psi_bar = np.conj(np.matmul(gamma1, psi))
            product = np.matmul(psi_bar, psi)
            self.assertAlmostEqual(np.imag(product), 0)

    def test_plaquette(self):
        N = 16
        m = Manifold(N)

        # check if the fan triangulation has a consistent sign configuration
        for i in range(len(m.adj)):
            U, def_triangles = circle_vertex(m.adj, m.sign, i)
            trace_plaquette = np.trace(U) / 2
            self.assertAlmostEqual(trace_plaquette, np.cos(def_triangles*np.pi/6))

        # check if edge flips preserve the consistency of the sign configuration
        m.sweep(n_sweeps=1, beta=0.6, strategy=['gravity', 'spinor_free'])
        for i in range(len(m.adj)):
            U, def_triangles = circle_vertex(m.adj, m.sign, i)
            trace_plaquette = np.trace(U)/2
            print(np.cos(def_triangles*np.pi/6))
            self.assertAlmostEqual(trace_plaquette, np.cos(def_triangles*np.pi/6))

    def test_gauge_sign(self):
        N = 16
        m = Manifold(N)
        np.testing.assert_allclose(m.sign**2, np.ones(3*N))
        m.sweep(10, 0.5, strategy)
        np.testing.assert_allclose(m.sign**2, np.ones(3*N))

if __name__ == '__main__':
    unittest.main()
