import unittest

import numpy as np

from divide import Predicate


class TestCase(unittest.TestCase):
    def test_max_gain(self):
        predicate = Predicate(0, 2)
        X = np.array([[1, 1, 1],
                      [2, 2, 2],
                      [3, 3, 3],
                      [1, 2, 3]])
        y = np.array([1, 2, 3, 4])

        gain = predicate.information_gain(X, y)
        self.assertEqual(1, gain)

    def test_min_gain(self):
        predicate = Predicate(0, 0)
        X = np.array([[1, 1, 1],
                      [2, 2, 2],
                      [3, 3, 3],
                      [1, 2, 3]])
        y = np.array([1, 2, 3, 4])

        gain = predicate.information_gain(X, y)
        self.assertEqual(0, gain)

    def test_gain(self):
        predicate = Predicate(0, 3)
        X = np.array([[1, 1, 1],
                      [1, 1, 1],
                      [2, 2, 2],
                      [3, 3, 3],
                      [3, 3, 3],
                      [1, 2, 3]])
        y = np.array([1, 2, 3, 4, 5,6])

        gain = predicate.information_gain(X, y)
        self.assertEqual(0.92, round(gain, 2))
