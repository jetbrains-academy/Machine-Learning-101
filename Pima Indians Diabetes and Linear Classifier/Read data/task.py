import numpy as np


# The function will load data from a text file with a relative path
# 'fname'
# and return it as a pair of arrays - one for different features
# and the other one for a diabetes presence
def read_data(fname):
    # genfromtxt method loads data from a text file, using a delimeter
    # to distinguish the columns
    data = np.genfromtxt(fname, delimiter=',')
    # The data is split into the all but the last column for X and
    # the last one for y respectively
    X, y = data[:, :-1], data[:, -1]
    # The features are rescaled:
    # X is standardized - all values are centered around mean
    # with a unit standard deviation. It means if we will calculate mean
    # and standard deviation of standard scores it will be 0 and 1 respectively.
    # This procedure is recommended if the data follows normal distribution
    X = # Standardize it
    # A column with '-1' values is added to the left of the X array
    # It is a pseudo-feature that will allow us to build initial vectors
    # for all of the cases. It will be essential later
    X = # Add a column of -1 to the left of the X
    # y is normalized - centered around 0 with a unit of 1
    y =  # {0, 1} -> {1, -1}
    return X, y


if __name__ == '__main__':
    X, y = read_data("pima-indians-diabetes.csv")
    print(f'Features in X array:', X)
    print(f'Diabetes: ', y)
