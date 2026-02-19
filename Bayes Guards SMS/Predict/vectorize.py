import numpy as np
import string


# This function creates an array of lines, where lines are transformed into lists of words
# without spaces and punctuation symbols.
def split_by_words(X):
    return np.char.split(np.char.translate(np.char.lower(X), str.maketrans('', '', string.punctuation)))


def vectorize(X):
    # Get the number of input messages
    X_len = len(X)
    # Convert each message into a vector of words
    X_split = split_by_words(X)

    # Get a 1D array of unique words
    uniques = np.unique(np.hstack(X_split))
    # Create an index dictionary and fill it with unique words and their indices
    index_dict = {}
    for index, word in enumerate(uniques):
        index_dict[word] = index

    # Create an array of zeros with dimensions corresponding
    # to input data size and index_dict length
    vectorization = np.zeros((X_len, len(index_dict)), dtype=np.int64)
    # The i-th row and j-th column contains the count
    # of the j-th word within the i-th message
    for index, message in enumerate(X_split):
        unique, count = np.unique(message, return_counts=True)
        for i, word in enumerate(unique):
            word_index = index_dict[word]
            vectorization[index, word_index] = count[i]

    return index_dict, vectorization