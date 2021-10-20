import unittest
import numpy as np
from numpy.testing import assert_array_equal

from plotting import centroid_histogram


class TestCase(unittest.TestCase):
    def test(self):
        labels = np.array([1, 2, 3, 4, 5, 1, 2, 3, 4, 1, 2, 3, 1, 2, 1])
        expected = np.array([5, 4, 3, 2, 1])
        histogram = centroid_histogram(labels)
        assert_array_equal(expected, histogram)
