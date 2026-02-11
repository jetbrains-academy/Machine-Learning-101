import unittest
import numpy as np
from numpy.testing import assert_array_equal

from clustering import k_means, init_clusters
from distances import euclidean_distance


class TestCase(unittest.TestCase):
    def test_kmeans_sizes(self):
        X = np.array([[0, 0], [0, 1], [0, 1]])
        labels, centers = k_means(X, 2, euclidean_distance)
        self.assertEqual(2, len(centers))
        self.assertEqual(3, len(labels))

    def test_kmeans_results(self):
        X = np.array([[0, 0], [0.5, 0], [0.5, 1], [1, 1]])
        expected_labels = [0, 0, 1, 1]
        expected_centers = np.array([[0, 0], [0, 1]])

        # init_clusters = lambda x, y: np.array([[0, 0], [1, 1]])
        for i in range(10):
            classification, clusters = k_means(X, 2, euclidean_distance)
            if np.array_equal(classification, [0, 0, 0, 0]):
                continue
            else:
                break
        assert_array_equal(classification, expected_labels)
        assert_array_equal(clusters, expected_centers)
