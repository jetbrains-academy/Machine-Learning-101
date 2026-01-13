import unittest
from task import read_data


# The reference implementation:
# def read_data(fname):
#     data = np.genfromtxt(fname, delimiter=',')
#     X, y = data[:, :-1], data[:, -1]
#     X = (X - X.mean(axis=0)) / X.std(axis=0)
#     X = np.concatenate((-np.ones(len(X)).reshape(-1, 1), X), axis=1)
#     y = -(y * 2 - 1)
#     return X, y

class TestCase(unittest.TestCase):
    def test_X(self):
        X, y = read_data("pima-indians-diabetes.csv")
        self.assertEqual((768, 9), X.shape, "Wrong train data length")

    def test_y(self):
        X, y = read_data("pima-indians-diabetes.csv")
        self.assertEqual(768, len(y), "Wrong train data length")

    def test_y_value(self):
        X, y = read_data("pima-indians-diabetes.csv")
        self.assertTrue(((y == -1) | (y == 1)).all(), "y array should contain only -1 and 1 values")
