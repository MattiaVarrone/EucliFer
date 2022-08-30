import unittest
import sys

sys.path.append("../2D_QG")
from a2D_QG.Triangulation import *


class TestGauge(unittest.TestCase):


    def test_adjecent(self, n=100):
        self.assertEqual(6, 6, "should be 6")


if __name__ == '__main__':
    unittest.main()