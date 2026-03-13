import numpy as np
import unittest

from safe_assert import safe_assert_array_equal

from task import read_image

class TestCase(unittest.TestCase):
    def test_read_image(self):
        image = read_image("./tests/star.png")
        expected_star = np.loadtxt("./tests/star.txt")
        safe_assert_array_equal(expected_star, image, "Incorrect image data")

    def test_shape(self):
        image = read_image("./superman-batman.png")
        self.assertEqual((786432, 3), image.shape, "Incorrect image shape")
