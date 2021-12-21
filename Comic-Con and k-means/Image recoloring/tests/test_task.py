import unittest
import numpy as np

from processing import recolor


class TestCase(unittest.TestCase):

    def test_colors_num(self):
        image = np.array([[255, 255, 255], [0, 0, 0], [0, 2, 0], [255, 255, 254]])

        for i in range(11):
            if i == 10:
                self.fail()
            try:
                recolored_image = recolor(image, 2)
                expected_image = np.array([[255, 255, 254], [0, 1, 0], [0, 1, 0], [255, 255, 254]])
                self.assertEqual(2, len(np.unique(recolored_image, axis=0)))
                np.testing.assert_array_equal(recolored_image, expected_image)
                break
            except IndexError:
                continue

    def test_colors_num_2(self):
        image = np.array([[255, 255, 255], [0, 0, 0], [0, 2, 0], [255, 255, 254]])
        recolored_image = recolor(image, 3)
        self.assertEqual(3, len(np.unique(recolored_image, axis=0)))
