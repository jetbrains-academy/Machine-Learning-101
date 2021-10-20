import unittest

from task import *
from evaluate import accuracy
from network import NN


class TestCase(unittest.TestCase):
    def test_none(self):
        X, y = read_data('iris.csv')
        X_train, y_train, X_test, y_test = train_test_split(X, y, 0.7)
        nn = NN(len(X[0]), 5, 1)
        self.assertIsNotNone(accuracy(nn, X_test, y_test), msg="your function returns nothing")

    def test_type(self):
        X, y = read_data('iris.csv')
        X_train, y_train, X_test, y_test = train_test_split(X, y, 0.7)
        nn = NN(len(X[0]), 5, 1)
        self.assertIsInstance(accuracy(nn, X_test, y_test), float, msg="your function returns a wrong type")

    def test_interval(self):
        X, y = read_data('iris.csv')
        X_train, y_train, X_test, y_test = train_test_split(X, y, 0.7)
        nn = NN(len(X[0]), 5, 1)
        self.assertTrue(0 <= accuracy(nn, X_test, y_test) <= 1, msg="accuracy should be within the [0, 1] interval")
