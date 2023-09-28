import sys
import unittest

import numpy as np
import scipy.linalg
from timebudget import timebudget

sys.path.append("../")
from manifold_sampler.Triangulation import *

beta = 0.5


class TestGauge(unittest.TestCase):
    @timebudget
    def test_adjacent(self):
        N = 22
        n_sweeps = 10
        m = Manifold(N)
        strategy = ['gravity', 'spinor_free']
        for _ in range(2):
            m.sweep(n_sweeps, beta, strategy)
            for i in range(len(m.adj)):
                self.assertEqual(m.A[i], -m.A[m.adj[i]])

    def test_paral_trans(self):
        A = 2
        np.testing.assert_allclose(scipy.linalg.expm(A * eps), paral_trans(A))


class TestConnection(unittest.TestCase):

    def test_transport_inverse(self):
        N = 22
        m = Manifold(N)
        strategy = ['gravity', 'spinor_free']
        As, Us = [], []
        for i in range(len(m.adj)):
            alpha = theta[i % 3] - theta[m.adj[i] % 3] + np.pi
            A = paral_trans(alpha / 2)
            U = m.sign[i] * A
            As.append(A), Us.append(U)
            if m.adj[i] < i:
                I_ = np.matmul(As[m.adj[i]], A)
                I = np.matmul(Us[m.adj[i]], U)
                np.testing.assert_allclose(I_, -np.eye(2), atol=1e-7)
                np.testing.assert_allclose(I, np.eye(2), atol=1e-7)

    def test_plaquette(self):
        N = 22
        m = Manifold(N)
        strategy = ['gravity', 'spinor_free']
        # check if the fan triangulation has a consistent sign configuration
        for i in range(len(m.adj)):
            U, def_triangles = circle_vertex(m.adj, m.sign, i)
            trace_plaquette = np.trace(U) / 2
            self.assertAlmostEqual(trace_plaquette, np.cos(def_triangles * np.pi / 6))

        # check if edge flips preserve the consistency of the sign configuration
        for _ in range(10 * N):
            m.random_update(beta=0.6, strategy=strategy)
            for i in range(len(m.adj)):
                U, def_triangles = circle_vertex(m.adj, m.sign, i)
                trace_plaquette = np.trace(U) / 2
                self.assertAlmostEqual(trace_plaquette, np.cos(def_triangles * np.pi / 6))

    def test_gauge_sign(self):
        N = 22
        m = Manifold(N)
        strategy = ['gravity', 'spinor_free']

        np.testing.assert_allclose(m.sign ** 2, np.ones(3 * N))
        for edge in m.adj:
            self.assertEqual(m.sign[edge], -m.sign[m.adj[edge]])

        for _ in range(22 * N):
            m.random_update(0.5, strategy)
            for edge in m.adj:
                self.assertEqual(m.sign[edge], -m.sign[m.adj[edge]])
            np.testing.assert_allclose(m.sign ** 2, np.ones(3 * N))


class TestAction(unittest.TestCase):
    @timebudget
    def test_dirac_determinant_sign(self):
        N = 22
        m = Manifold(N)
        strategy = ['gravity', 'spinor_free']

        # check if the fan triangulation has a consistent dirac determinant sign
        det_sign = np.linalg.slogdet(m.D)[0]
        self.assertEqual(det_sign, 1)

        # check if edge flips preserve the consistency of the dirac determinant sign
        for it in range(10 * N):
            m.random_update(beta=0.6, strategy=strategy)
            det_sign = np.linalg.slogdet(m.D)[0]
            self.assertEqual(1, det_sign)

    @timebudget
    def test_dirac_operator_antisymm(self):
        N = 22
        m = Manifold(N)
        strategy = ['gravity', 'spinor_free']

        # check if the fan triangulation has a consistent dirac determinant sign
        D_sym = (m.D + m.D.T)/2
        np.testing.assert_allclose(D_sym, np.zeros_like(D_sym), atol=1e-7)

        # check if edge flips preserve the consistency of the dirac determinant sign
        for it in range(10 * N):
            m.random_update(beta=0.6, strategy=strategy)
            D_sym = (m.D + m.D.T) / 2
            np.testing.assert_allclose(D_sym, np.zeros_like(D_sym), atol=1e-7)


if __name__ == '__main__':
    unittest.main()
