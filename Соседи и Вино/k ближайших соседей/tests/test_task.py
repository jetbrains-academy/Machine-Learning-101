import numpy as np
import unittest

from numpy.ma.testutils import assert_array_equal

from metric_classification import knn
from distances import euclidean_dist


class TestCase(unittest.TestCase):
    def test_length(self):
        X_train = np.array([
            [255, 255, 255],
            [0, 0, 0],
            [128, 128, 128],
            [255, 0, 0],
            [0, 255, 0],
            [0, 0, 255]
        ])
        y_train = np.array([0, 1, 2, 3, 4, 5])

        X_test = np.array([
            [0, 255, 0],
            [255, 0, 0],
            [0, 0, 255],
            [255, 0, 255],
            [100, 0, 50],
            [100, 100, 100],
            [50, 50, 50],
            [200, 200, 200],
            [10, 20, 30],
            [100, 10, 200],
            [32, 0, 255],
            [128, 255, 64]
        ])
        y_predicted = knn(X_train, y_train, X_test, 1, euclidean_dist)
        self.assertEqual(len(y_predicted), 12, "You should assign label for each object in the X_train")

    def test_knn_1_neighbor(self):
        X_train = np.array([
            [255, 255, 255],
            [0, 0, 0],
            [128, 128, 128],
            [255, 0, 0],
            [0, 255, 0],
            [0, 0, 255]
        ])
        y_train = np.array([0, 1, 2, 3, 4, 5])

        X_test = np.array([
            [0, 255, 0],
            [255, 0, 0],
            [0, 0, 255],
            [255, 0, 255],
            [100, 0, 50],
            [100, 100, 100],
            [50, 50, 50],
            [200, 200, 200],
            [10, 20, 30],
            [100, 10, 200],
            [32, 0, 255],
            [128, 255, 64]
        ])
        y_test = np.array([4, 3, 5, 2, 1, 2, 1, 0, 1, 5, 5, 2])
        y_predicted = knn(X_train, y_train, X_test, 1, euclidean_dist)
        assert_array_equal(y_predicted, y_test)

    def test_knn_4_neighbor(self):
        X_train = np.array([
            [255, 255, 255],
            [0, 0, 0],
            [128, 128, 128],
            [255, 0, 0],
            [0, 255, 0],
            [0, 0, 255]
        ])
        y_train = np.array([0, 1, 2, 3, 4, 5])

        X_test = np.array([
            [0, 255, 0],
            [255, 0, 0],
            [0, 0, 255],
            [255, 0, 255],
            [100, 0, 50],
            [100, 100, 100],
            [50, 50, 50],
            [200, 200, 200],
            [10, 20, 30],
            [100, 10, 200],
            [32, 0, 255],
            [128, 255, 64]
        ])
        y_test = np.array([0, 0, 1, 0, 1, 1, 1, 0, 1, 1, 1, 0])
        y_predicted = knn(X_train, y_train, X_test, 4, euclidean_dist)
        assert_array_equal(y_predicted, y_test)
