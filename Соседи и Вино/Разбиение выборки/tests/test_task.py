from numpy.ma.testutils import assert_array_equal, fail_if_array_equal

from task import train_test_split
import unittest
import numpy as np

# The solution checked in tests looks like this
# def train_test_split(X, y, ratio=0.8):
#     indices = np.random.permutation(X.shape[0])
#     train_len = int(X.shape[0] * ratio)
#     return X[indices[:train_len]], y[indices[:train_len]], X[indices[train_len:]], y[indices[train_len:]]

class TestCase(unittest.TestCase):
    def test_equal_split(self):
        X = np.arange(100).reshape((10, 10))
        y = np.arange(10)
        X_train, y_train, X_test, y_test = train_test_split(X, y, ratio=.5)

        self.assertEqual(len(y_test), len(y_train), "Check your function with ratio=0.5")

    def test_arrays_mapping(self):
        X = np.arange(100).reshape((10, 10))
        y = np.arange(10)
        X_train, y_train, X_test, y_test = train_test_split(X, y, ratio=.5)

        assert_array_equal(X_train[:, 0], y_train * 10, "X_train doesn't match y_train values")
        assert_array_equal(X_test[:, 0], y_test * 10, "X_test doesn't match y_test values")

    def test_arrays_lengths(self):
        X = np.arange(100).reshape((10, 10))
        y = np.arange(10)
        X_train, y_train, X_test, y_test = train_test_split(X, y, ratio=.7)

        assert_array_equal(X_train.shape[0], len(y_train), "X_train length doesn't match y_train length")
        assert_array_equal(X_test.shape[0], len(y_test), "X_test length doesn't match y_test length")

    def test_ratio(self):
        X = np.arange(100).reshape((10, 10))
        y = np.arange(10)
        ratio = .8
        X_train, y_train, X_test, y_test = train_test_split(X, y, ratio=ratio)
        self.assertEqual(len(X_train) / (len(X_test) + len(X_train)), ratio,
                         "len(X_train) / (len(X_test) + len(X_train)) != ratio")
        self.assertEqual(len(y_train) / (len(y_test) + len(y_train)), ratio,
                         "len(y_train) / (len(y_test) + len(y_train)) != ratio")

    def test_randomize(self):
        X = np.arange(100).reshape((10, 10))
        y = np.arange(10)
        ratio = .8
        X_train, y_train, X_test, y_test = train_test_split(X, y, ratio=ratio)
        X_train1, y_train1, X_test1, y_test1 = train_test_split(X, y, ratio=ratio)
        fail_if_array_equal(X_train, X_train1, "train_test_split should split arrays into random train and test subsets")
        fail_if_array_equal(X_test, X_test1, "train_test_split should split arrays into random train and test subsets")
        fail_if_array_equal(y_train, y_train1, "train_test_split should split arrays into random train and test subsets")
        fail_if_array_equal(y_test, y_test1, "train_test_split should split arrays into random train and test subsets")

    def test_no_randomize_initial_arrays(self):
        X = np.arange(100).reshape((10, 10))
        y = np.arange(10)
        ratio = .8
        train_test_split(X, y, ratio=ratio)
        assert_array_equal(X, np.arange(100).reshape((10, 10)), "Do not reorder initial X array")
        assert_array_equal(y, np.arange(10), "Do not reorder initial y array")