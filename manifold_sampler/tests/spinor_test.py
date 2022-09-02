import sys
import unittest

import numpy as np
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


if __name__ == '__main__':
    unittest.main()
