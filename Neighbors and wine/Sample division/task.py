import numpy as np


def train_test_split(X, y, ratio=0.8):
    indices = np.random.permutation(X.shape[0])
    train_len = int(X.shape[0] * ratio)
    # A slice with first train_len elements of X will be
    # the training sample for objects.
    X_train
    # A slice with the rest of X will be
    # the testing sample for objects.
    X_test
    # A slice with first train_len of y will be
    # the training sample for classes.
    y_train
    # A slice with the rest of y will be
    # the testing sample for classes.
    y_test
    return X_train, y_train, X_test, y_test


if __name__ == '__main__':
    # Here we load data from the csv file.
    wines = np.genfromtxt('wine.csv', delimiter=',')
    # Here the data is split into objects and classes
    # to process them separately. Note that an object and
    # its corresponding class will have the same index.
    X, y = wines[:, 1:], np.array(wines[:, 0], dtype=np.int32)
    # Here we call our function to look at the result.
    X_train, y_train, X_test, y_test = train_test_split(X, y, 0.6)
    # It is convenient to add visualization (a console output at least)
    # for each of the functions you add in a new step. Like this:
    print("X_train: ", "\n")
    print(X_train)
    print("y_train: ")
    print(y_train)
    print("X_test", "\n")
    print(X_test)
    print("y_test", "\n")
    print(y_test)
