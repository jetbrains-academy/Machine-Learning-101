import unittest

import numpy as np

from bayes import NaiveBayes


class TestCase(unittest.TestCase):
    def test_score(self):
        nb = NaiveBayes()
        X = np.array(['A great game', 'The election was over', 'Very clean match', 'A clean but forgettable game',
                          'It was a close election'])
        y = np.array(['Sports', 'Not sports', 'Sports', 'Sports', 'Not sports'])
        nb.fit(X, y)
        self.assertEqual(1, nb.score(X, y), msg="The score is incorrect!")

    def test_game(self):
        nb = NaiveBayes()
        X = np.array(['A great game', 'The election was over', 'Very clean match', 'A clean but forgettable game',
                          'It was a close election'])
        y = np.array(['Sports', 'Not sports', 'Sports', 'Sports', 'Not sports'])
        nb.fit(X, y)
        self.assertEqual(["Sports"], nb.predict(np.array(["game"])), msg="Your predictions seem off!")

    def test_election(self):
        nb = NaiveBayes()
        X = np.array(['A great game', 'The election was over', 'Very clean match', 'A clean but forgettable game',
                          'It was a close election'])
        y = np.array(['Sports', 'Not sports', 'Sports', 'Sports', 'Not sports'])
        nb.fit(X, y)
        self.assertEqual(["Not sports"], nb.predict(np.array(["election"])), msg="Your predictions seem off!")
