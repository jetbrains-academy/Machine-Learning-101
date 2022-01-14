import unittest

import numpy as np

from gradient_descent import GradientDescent


class TestCase(unittest.TestCase):
    def test_weights(self):
        gd = GradientDescent(alpha=0.1)
        X = np.array([[1, 2, 3, 4, 5],
                      [1, 2, 3, 4, 5],
                      [0, 0, 0, 0, 0],
                      [0, 0, 0, 0, 0]])
        y = np.array([1, 1, 2, 2])
        gd.fit(X, y)
        self.assertEquals(5, len(gd.weights))

    def test_fit(self):
        gd = GradientDescent(alpha=0.1)
        X = np.array([[1, 2, 3, 4, 5], [1, 2, 3, 4, 5]])
        y = np.array([1, 2])
        result = gd.fit(X, y)
        self.assertNotEqual(0, len(result))

    def test_predict(self):
        gd = GradientDescent(alpha=0.1)
        X = np.array([[1, 2, 3, 4, 5],
                      [1, 2, 3, 4, 5],
                      [0, 0, 0, 0, 0],
                      [0, 0, 0, 0, 0]])
        y = np.array([1, 1, 2, 2])
        gd.fit(X, y)
        self.assertEquals(1, gd.predict(np.array([1, 2, 3, 4, 5])))
        self.assertEquals(0, gd.predict(np.array([0, 0, 0, 0, 0])))
