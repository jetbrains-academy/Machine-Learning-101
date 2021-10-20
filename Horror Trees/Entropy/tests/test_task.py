import unittest
import numpy as np
from calculate_entropy import entropy



class TestCase(unittest.TestCase):
    def test_equally_divided(self):
        a = np.array([1, 2, 1, 2, 1, 2])
        ent = entropy(a)
        self.assertEqual(1, ent, "Entropy for the equally divided array should be 1")

    def test_homogeneous(self):
        a = np.array([1, 1, 1, 1, 1, 1])
        ent = entropy(a)
        self.assertEqual(0, ent, "Entropy for the homogeneous array should be 0")

    def test_random(self):
        a = np.array([1, 2, 1, 1, 1, 1])
        ent = entropy(a)
        self.assertEqual(0.65, round(ent, 2), "Wrong entropy computed")

