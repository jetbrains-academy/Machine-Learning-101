import unittest
from activation import sigmoid


class TestCase(unittest.TestCase):
    def test_sigmoid(self):
        self.assertEqual(0.01, round(sigmoid(-5), 2))
        self.assertEqual(0.05, round(sigmoid(-3), 2))
        self.assertEqual(0.27, round(sigmoid(-1), 2))
        self.assertEqual(0.73, round(sigmoid(1), 2))
        self.assertEqual(0.95, round(sigmoid(3), 2))

    def test_relu(self):
        self.assertNotEqual(0, sigmoid(-1), msg="You should use sigmoid activation function, not ReLU!")
        self.assertNotEqual(1, sigmoid(1), msg="You should use sigmoid activation function, not ReLU!")
        self.assertNotEqual(100, sigmoid(100), msg="You should use sigmoid activation function, not ReLU!")

    def test_tanh(self):
        self.assertNotEqual(0, sigmoid(0), msg="You should use sigmoid activation function!")
        self.assertNotEqual(0.9951, round(sigmoid(3), 4), msg="You should use sigmoid activation function, not tanh!")