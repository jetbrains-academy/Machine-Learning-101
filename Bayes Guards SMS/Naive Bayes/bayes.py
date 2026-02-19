import numpy as np
from vectorize import *

class NaiveBayes:

    def fit(self, X, y):
        # This allows us to get the uniques classes from the array of all class labels
        self.unique_classes = np.unique(y)

        # Get the dictionary of unique words and the vectorized representation of the input data using vectorize()
        self.dictionary, X = vectorize(X)
        self.dict_size = len(self.dictionary)

        # Create three arrays of required dimensions to store class prior probabilities, total word
        # counts per class, and relative word frequencies for each class, respectively:
        self.classes_prior = np.zeros(len(self.unique_classes), dtype=np.float64)
        self.classes_words_count = np.zeros(len(self.unique_classes), dtype=np.float64)
        self.likelihood = np.full((len(self.unique_classes), self.dict_size + 1), 0, dtype=np.float64)

        for i, clazz in enumerate(self.unique_classes):
            # Create a mask to filter data based on the class currently being processed
            y_i_mask = y == clazz
            # Count the sum of bools in the mask array to get the number of class occurrences
            # in the whole training set
            y_i_sum = np.sum(y_i_mask)
            # Class prior probability is its fraction in the training set
            self.classes_prior[i] = y_i_sum / len(y)
            # Count the number of words in each class
            self.classes_words_count[i] = np.sum(X[y_i_mask])
            # Calculate the likelihood numerator by counting how many times each unique word
            # is encountered in all messages of this class
            self.likelihood[i, :-1] += np.sum(X[y_i_mask], 0)

            # Get the denominator for this class and finalize the calculation of the
            # likelihood of each word for this class
            denominator = self.classes_words_count[i]
            self.likelihood[i] = self.likelihood[i] / denominator