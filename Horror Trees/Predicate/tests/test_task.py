import numpy as np
import unittest

from numpy import array_equal
from numpy.ma.testutils import assert_array_equal

from divide import Predicate


class TestCase(unittest.TestCase):
    def test_nominal_int(self):
        predicate = Predicate(0, 2)
        X = np.array([[1, 1, 1],
                      [2, 2, 2],
                      [3, 3, 3],
                      [1, 2, 3]])
        y = np.array([1, 2, 3, 4])

        X1, y1, X2, y2 = predicate.divide(X, y)

        if array_equal(np.array([[2, 2, 2], [3, 3, 3]]), X1):
            assert_array_equal(np.array([2, 3]), y1, err_msg="Incorrect split for int feature")
            assert_array_equal(np.array([[1, 1, 1], [1, 2, 3]]), X2, err_msg="Incorrect split for int feature")
            assert_array_equal(np.array([1, 4]), y2, err_msg="Incorrect split for int feature")
        else:
            assert_array_equal(np.array([[1, 1, 1], [1, 2, 3]]), X1, err_msg="Incorrect split for int feature")
            assert_array_equal(np.array([1, 4]), y1, err_msg="Incorrect split for int feature")
            assert_array_equal(np.array([[2, 2, 2], [3, 3, 3]]), X2, err_msg="Incorrect split for int feature")
            assert_array_equal(np.array([2, 3]), y2, err_msg="Incorrect split for int feature")

    def test_nominal_float(self):
        predicate = Predicate(0, 2.1)
        X = np.array([[1., 1., 1.],
                      [2., 2., 2.],
                      [3., 3., 3.],
                      [1., 2., 3.]])
        y = np.array([1, 2, 3, 4])

        X1, y1, X2, y2 = predicate.divide(X, y)

        if array_equal(np.array([[3, 3, 3]]), X1):
            assert_array_equal(np.array([3]), y1, err_msg="Incorrect split for float feature")
            assert_array_equal(np.array([[1, 1, 1], [2, 2, 2], [1, 2, 3]]), X2,
                               err_msg="Incorrect split for float feature")
            assert_array_equal(np.array([1, 2, 4]), y2, err_msg="Incorrect split for float feature")
        else:
            assert_array_equal(np.array([[1, 1, 1], [2, 2, 2], [1, 2, 3]]), X1,
                               err_msg="Incorrect split for float feature")
            assert_array_equal(np.array([1, 2, 4]), y1, err_msg="Incorrect split for float feature")
            assert_array_equal(np.array([[3, 3, 3]]), X2, err_msg="Incorrect split for float feature")
            assert_array_equal(np.array([3]), y2, err_msg="Incorrect split for float feature")

    def test_quantitative(self):
        predicate = Predicate(3, 'clear')
        X = np.array([[1, 1, 1, 'clear'],
                      [2, 2, 2, 'clear'],
                      [3, 3, 3, 'green'],
                      [1, 2, 3, 'black']])
        y = np.array([1, 2, 3, 4])

        X1, y1, X2, y2 = predicate.divide(X, y)

        if array_equal(np.array([[1, 1, 1, 'clear'], [2, 2, 2, 'clear']]), X1):
            assert_array_equal(np.array([1, 2]), y1, err_msg="Incorrect split for quantitative feature")
            assert_array_equal(np.array([[3, 3, 3, 'green'], [1, 2, 3, 'black']]), X2,
                               err_msg="Incorrect split for quantitative feature")
            assert_array_equal(np.array([3, 4]), y2, err_msg="Incorrect split for quantitative feature")
        else:
            assert_array_equal(np.array([[3, 3, 3, 'green'], [1, 2, 3, 'black']]), X1,
                               err_msg="Incorrect split for quantitative feature")
            assert_array_equal(np.array([3, 4]), y1, err_msg="Incorrect split for quantitative feature")
            assert_array_equal(np.array([[1, 1, 1, 'clear'], [2, 2, 2, 'clear']]), X2,
                               err_msg="Incorrect split for quantitative feature")
            assert_array_equal(np.array([1, 2]), y2, err_msg="Incorrect split for quantitative feature")
