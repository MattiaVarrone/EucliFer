import sys
import unittest

sys.path.append("../")
from manifold_sampler.Triangulation import Manifold

strategy = ['gravity', 'spinor']
beta = 0.5


class TestGauge(unittest.TestCase):

    def test_adjacent(self):
        N = 50
        n_sweeps = 20
        m = Manifold(N)
        m.sweep(n_sweeps, beta, strategy)
        for i in range(len(m.adj)):
            self.assertEqual(m.A[i], -m.A[m.adj[i]])


if __name__ == '__main__':
    unittest.main()
