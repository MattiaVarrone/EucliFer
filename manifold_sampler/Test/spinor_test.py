import sys
import unittest

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
                self.assertEqual(m.A[i], -m.A[m.adj[i]])    ### RARELY DOES NOT WORK

    def test_paral_trans(self):
        A = 2
        np.testing.assert_allclose(scipy.linalg.expm(A * eps), paral_trans(A))




if __name__ == '__main__':
    unittest.main()
