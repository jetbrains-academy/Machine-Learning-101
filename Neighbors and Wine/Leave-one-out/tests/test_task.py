import unittest

from distances import euclidean_dist
from crossvalidation import loocv

import numpy as np


class TestCase(unittest.TestCase):
    def test_loo(self):
        X_train = np.array([
            [0, 4],
            [0, 7],
            [3, 9],
            [7, 2],
            [4, 8],
            [3, 7],
            [4, 0],
            [4, 4]
        ])
        y_train = np.array([0, 1, 1, 0, 0, 1, 0, 0])

        euclidean_opt = loocv(X_train, y_train, euclidean_dist)
        self.assertEqual(3, euclidean_opt)

    def test_loo_1_neighbor(self):
        X_train = np.array([
            [255, 255, 255],
            [255, 255, 255],
            [255, 255, 255],
            [128, 128, 128],
            [128, 128, 128],
            [128, 128, 128],
        ])
        y_train = np.array([1, 1, 1, 0, 0, 0])

        euclidean_opt = loocv(X_train, y_train, euclidean_dist)
        self.assertEqual(1, euclidean_opt)
