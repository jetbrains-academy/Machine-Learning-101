import unittest
import numpy as np
from tree import DecisionTree
from task import Node


class TestCase(unittest.TestCase):
    def test_root(self):
        X = np.array([[1, 2, 3],
                      [2, 2, 2],
                      [1, 4, 5]])
        y = np.array([1, 2, 1])

        tree = DecisionTree().build(X, y)
        self.assertEqual(1, tree.root.value)
        self.assertEqual(0, tree.root.column)

    def test_nodes(self):
        X = np.array([[1, 2, 3],
                      [4, 5, 6],
                      [7, 8, 9]])
        y = np.array([1, 2, 3])

        tree = DecisionTree().build(X, y)
        self.assertEqual(3, count_nodes(tree.root))

    # a test to check if get_best_predicate() method returns anything, if not - print "implement get_best_predicate() method"
    # a test to check if get_best_predicate() returns an instance of Predicate, if not - print "implement the method ... which returns an instance of Predicate"

    # a test to check if build_subtree() returns something, if not - print "implement the method get_best_predicate()"


def count_nodes(node):
    counter = 0
    if isinstance(node, Node):
        counter += count_nodes(node.false_branch)
        counter += count_nodes(node.true_branch)
    else:
        counter += 1
    return counter
