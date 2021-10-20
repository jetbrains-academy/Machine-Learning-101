import unittest

import numpy as np

from metrics import precision_recall


class TestCase(unittest.TestCase):
    def test_perfect_precision_recall(self):
        y_test = np.array([1, 1, 1, 0, 1, 0, 1, 0, 1, 1, 1, 0])
        y_predicted = np.array([1, 1, 1, 0, 1, 0, 1, 0, 1, 1, 1, 0])
        result = precision_recall(y_predicted, y_test)
        for _, precision, recall in result:
            self.assertEqual(1, precision)
            self.assertEqual(1, recall)

    def test_zero_precision_recall(self):
        y_test = np.array([1, 1, 1, 0, 1, 0, 1, 0, 1, 1, 1, 0])
        y_predicted = np.array([0, 0, 0, 1, 0, 1, 0, 1, 0, 0, 0, 1])
        result = precision_recall(y_predicted, y_test)
        for _, precision, recall in result:
            self.assertEqual(0, precision)
            self.assertEqual(0, recall)

    def test_full_recall(self):
        y_predicted = np.array([1, 1, 1, 1, 1, 1, 1, 1, 1, 1])
        y_test = np.array([0, 0, 1, 0, 1, 0, 1, 0, 1, 1])
        result = precision_recall(y_predicted, y_test)

        class_0_precision_recall = result[0]
        self.assertEqual(0, class_0_precision_recall[1])
        self.assertEqual(0, class_0_precision_recall[2])

        class_1_precision_recall = result[1]
        self.assertEqual(0.5, class_1_precision_recall[1])
        self.assertEqual(1, class_1_precision_recall[2])

    def test_classes(self):
        y_predicted = np.array([1, 1, 1, 1, 1, 1, 1, 1, 1, 1])
        y_test = np.array([0, 0, 1, 0, 1, 0, 1, 0, 1, 1])
        result = precision_recall(y_predicted, y_test)
        self.assertEqual(2, len(result))
