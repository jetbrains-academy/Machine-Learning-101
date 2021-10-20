import unittest

import numpy as np

from tree import DecisionTree


class TestCase(unittest.TestCase):
    def test_has_method(self):
        self.assertTrue(hasattr(DecisionTree, "predict"), "Implement method `predict`")
        # Not sure this makes any sense since we provide method signature^

    # Maybe a test that checks that classify_subtree() returns something and if it doesn't - print "Implement methods `classify_subtree`"

    def test_root(self):
        X = np.array([[1, 2, 3],
                      [2, 2, 2],
                      [1, 4, 5]])
        y = np.array([1, 2, 1])

        tree = DecisionTree().build(X, y)
        self.assertEqual(1, tree.predict(X[0]))
