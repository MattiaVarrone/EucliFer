import sys
import unittest

import scipy.linalg

sys.path.append("../")
from manifold_sampler.Triangulation import *

beta = 0.5


class TestGauge(unittest.TestCase):

    def test_adjacent(self):
        N = 16
        n_sweeps = 100
        m = Manifold(N)
        strategy = ['gravity', 'spinor_inter']
        for _ in range(2):
            m.sweep(n_sweeps, beta, strategy)
            for i in range(len(m.adj)):
                self.assertEqual(m.A[i], -m.A[m.adj[i]])  ### RARELY DOES NOT WORK # fixed now

    def test_paral_trans(self):
        A = 2
        np.testing.assert_allclose(scipy.linalg.expm(A * eps), paral_trans(A))


class TestPsi(unittest.TestCase):

    def test_mass_term_real(self):
        N = 16
        n_sweeps = 10
        m = Manifold(N)
        strategy = ['gravity', 'spinor_inter']
        m.sweep(n_sweeps, beta, strategy)
        for c in range(N):
            psi = m.psi[c]
            psi_bar = np.conj(np.matmul(gamma1, psi))
            product = np.matmul(psi_bar, psi)
            self.assertAlmostEqual(np.imag(product), 0)

class TestConnection(unittest.TestCase):

    def test_transport_inverse(self):
        N = 20
        m = Manifold(N)
        strategy = ['gravity', 'spinor_free']
        Us = []
        for i in range(len(m.adj)):
            alpha = theta[i % 3] - theta[m.adj[i] % 3] + np.pi
            U = paral_trans(alpha / 2)
            Us.append(U)
            if m.adj[i] < i:
                I = np.matmul(Us[m.adj[i]], U)
                print(I[0, 0])
                # np.testing.assert_allclose(I, np.eye(2))

    def test_plaquette(self):
        N = 20
        m = Manifold(N)
        strategy = ['gravity', 'spinor_free']
        # check if the fan triangulation has a consistent sign configuration
        for i in range(len(m.adj)):
            U, def_triangles = circle_vertex(m.adj, m.sign, m.signc, i)
            trace_plaquette = np.trace(U) / 2
            self.assertAlmostEqual(trace_plaquette, np.cos(def_triangles * np.pi / 6))

        # check if edge flips preserve the consistency of the sign configuration
        for it in range(10 * N):
            m.random_update(beta=0.6, strategy=strategy)
            for i in range(len(m.adj)):
                U, def_triangles = circle_vertex(m.adj, m.sign, m.signc, i)
                trace_plaquette = np.trace(U) / 2
                self.assertAlmostEqual(trace_plaquette, np.cos(def_triangles * np.pi / 6))
            print(it)


    def test_gauge_sign(self):
        N = 10
        m = Manifold(N)
        strategy = ['gravity', 'spinor_free']

        np.testing.assert_allclose(m.sign ** 2, np.ones(3 * N))
        for edge in m.adj:
            self.assertEqual(m.sign[edge], -m.sign[m.adj[edge]])

        for _ in range(30 * N):
            m.random_update(0.5, strategy)
            for edge in m.adj:
                self.assertEqual(m.sign[edge], -m.sign[m.adj[edge]])
            np.testing.assert_allclose(m.sign ** 2, np.ones(3 * N))
            np.testing.assert_allclose(m.signc ** 2, np.ones(N))


if __name__ == '__main__':
    unittest.main()
