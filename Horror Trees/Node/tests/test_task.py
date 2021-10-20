import unittest
from task import Node

# Tested code should look something like:
#     def __init__(self, column=-1, value=None, true_branch=None, false_branch=None):
#         self.column = column
#         self.value = value
#         self.true_branch = true_branch
#         self.false_branch = false_branch


class TestCase(unittest.TestCase):
    def test_true_branch(self):
        node = Node(1, 2, [1,2], [3,4])
        self.assertTrue(hasattr(node, "true_branch"), "Store true_branch in the true_branch field")

    def test_false_branch(self):
        node = Node(1, 2, [1,2], [3,4])
        self.assertTrue(hasattr(node, "false_branch"), "Store false_branch in the true_branch field")

    # maybe a test that checks that new column / value value is assigned

    def test_fields(self):
        node = Node(1, 2, [1,2], [3,4])
        self.assertEqual(4, len(node.__dict__), "You should store all values passed to the Node object as field")
