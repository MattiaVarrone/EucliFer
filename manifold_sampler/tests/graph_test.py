import sys
import unittest

sys.path.append("../")
from manifold_sampler.Triangulation import *
from manifold_sampler.Graph_utils import *


class TestGraph(unittest.TestCase):

    def test_is_sphere(self):
        N = 16
        beta = 0.6
        iter = 20
        strategy = ['gravity', 'ising', 'scalar', 'spinor_free']
        m = Manifold(N)

        self.assertEqual(is_sphere_triangulation(m.adj), True)
        m.sweep(iter, beta, strategy)
        self.assertEqual(is_sphere_triangulation(m.adj), True)


if __name__ == '__main__':
    unittest.main()
