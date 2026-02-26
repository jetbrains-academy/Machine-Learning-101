from calculate_entropy import entropy


# Here we define the Predicate class to store predicates – values in
# specific columns used to split the dataset. The class
# includes a divide method to partition the data
# and an information_gain method to calculate the gain achieved by a given split.
class Predicate:
    def __init__(self, column, value):
        self.column = column
        self.value = value

    def divide(self, X, y):
        # Check if the value is numeric and create a boolean filter array
        # based on the "greater than or equal to" condition.
        if  isinstance(self.value, int) or isinstance(self.value, float):
            mask =  X[:, self.column] >= self.value
            # If the value is not numeric (int or float),
            # create the array based on the "equal to" condition.
        else:
            mask = X[:, self.column] == self.value
            # Return the results in the following order: X1, y1, X2, y2.
        return X[mask], y[mask], X[~mask], y[~mask]

    # This method is to be implemented in the task "Information Gain".
    def information_gain(self, X, y):
        pass
        # Use the divide method to split the sample.
        # X1, y1, X2, y2 = # TODO
        # Calculate the fraction of X1 in the whole dataset.
        # p = # TODO
        # Use the entropy function you wrote earlier and the formula
        # from the task to calculate the information gain.
        # gain = # TODO
        # Return the gain