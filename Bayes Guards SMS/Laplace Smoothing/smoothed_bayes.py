import numpy as np
from vectorize import *

class SmoothedNaiveBayes:
    def __init__(self, alpha=1):
        self.alpha = alpha

    def fit(self, X, y):
        self.unique_classes = np.unique(y)

        self.dictionary, X = vectorize(X)
        self.dict_size = len(self.dictionary)

        # Create three arrays of required dimensions to store class prior probabilities, total number
        # of words in each class and relative word frequencies for each class, respectively:
        self.classes_prior = np.zeros(len(self.unique_classes), dtype=np.float64)
        self.classes_words_count = np.zeros(len(self.unique_classes), dtype=np.float64)
        self.likelihood = np.full((len(self.unique_classes), self.dict_size + 1), self.alpha, dtype=np.float64)

        for i, clazz in enumerate(self.unique_classes):
            y_i_mask = y == clazz
            y_i_sum = np.sum(y_i_mask)
            self.classes_prior[i] = y_i_sum / len(y)
            self.classes_words_count[i] = np.sum(X[y_i_mask])
            self.likelihood[i, :-1] += np.sum(X[y_i_mask], 0)

            denominator = self.classes_words_count[i] + self.alpha * self.dict_size
            self.likelihood[i] = self.likelihood[i] / denominator

    def predict(self, X):
        pass
        # result = []
        # # Transform each message within the input array into a vector of words
        # X = split_by_words(X)
        #
        # # In each message find the unique words and create an array of zeros of the appropriate size
        # for message in X:
        #     unique = np.unique(message)
        #     index_array = # TODO
        #
        #     # Look for each unique word in the dictionary, and add its index to the array we just created,
        #     # If its not there - add the index equal to the length of the dictionary
        #     for i, word in enumerate(unique):
        #         word_index = # TODO
        #         index_array[i] = word_index
        #
        #     # Slice the likelihood array to leave only the words for words that are in the current message
        #     # and apply logarithm to calculate log likelihood
        #     log_likelihood = # TODO
        #     # Refer to the formula and hints in the text to calculate the posterior probability
        #     # For each class and select the class with the largest probability, append it to the result
        #     posterior = # TODO
        #     predicted = # TODO
        #     result.append(predicted)
        # return result

    # This method should run the algorithm on the test set, compare the obtained classification
    # results with the real class labels, and return the proportion of correctly classified objects.
    def score(self, X, y):
        # TODO
        pass