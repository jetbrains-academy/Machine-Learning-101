import numpy as np
from vectorize import *


class NaiveBayes:
    # a predefined method needed for 'Laplace Smoothing' that initializes the
    # smoothing alpha parameter, by default it's 1
    def __init__(self, alpha=1):
        self.alpha = alpha

    def fit(self, X, y):
        # This allows us to get the uniques classes from the array of all class labels
        self.unique_classes = np.unique(y)

        # get the unique dictionary and input data representation array using vectorize()
        self.dictionary, X = vectorize(X)
        self.dict_size = len(self.dictionary)

        # create three arrays of required dimensions to store class prior probabilities, total number
        # of words in each class and relative word frequencies for each class, respectively:
        self.classes_prior = np.zeros(len(self.unique_classes), dtype=np.float64)
        self.classes_words_count = np.zeros(len(self.unique_classes), dtype=np.float64)
        self.likelihood = np.full((len(self.unique_classes), self.dict_size + 1), 0, dtype=np.float64)

        for i, clazz in enumerate(self.unique_classes):
            # create a mask to filter data based on which class in processed right now
            y_i_mask = # TODO
            # count the sum of bools in the mask array to get the number of class occurrences
            # in the whole training set
            y_i_sum = np.sum(y_i_mask)
            # class prior probability is its fraction in the training set
            self.classes_prior[i] = # TODO
            # count the number of words in each class
            self.classes_words_count[i] = # TODO
            # calculate the likelihood nominator by counting how many times each unique word
            # is encountered in all messages of this class
            self.likelihood[i, :-1] += # TODO

            # get the denominator for this class and finalize the calculation of the
            # likelihood of this word for this class
            denominator = # TODO
            self.likelihood[i] = self.likelihood[i] / denominator

    # the following methods are needed for the task 'Predict'
    def predict(self, X):
        pass
        # result = []
        # # transform each message within the input array into a vector of words
        # X = split_by_words(X)
        #
        # # in each message find the unique words and create an array of zeros of the appropriate size
        # for message in X:
        #     unique = np.unique(message)
        #     index_array = # TODO
        #
        #     # look for each unique word in the dictionary, and add its index to the array we just created,
        #     # if its not there - add the index equal to the length of the dictionary
        #     for i, word in enumerate(unique):
        #         word_index = # TODO
        #         index_array[i] = word_index
        #
        #     # slice the likelihood array to leave only the words for words that are in the current message
        #     # and apply logarithm to calculate log likelihood
        #     log_likelihood = # TODO
        #     # refer to the formula and hints in the text to calculate the posterior probability
        #     # for each class and select the class with the largest probability, append it to the result
        #     posterior = # TODO
        #     predicted = # TODO
        #     result.append(predicted)
        # return result


    # This method should run the algorithm on the test set, compare the obtained classification
    # results with the real class labels, and return the proportion of correctly classified objects.
    def score(self, X, y):
        pass
        # return # TODO