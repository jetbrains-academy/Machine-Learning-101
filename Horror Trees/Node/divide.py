from calculate_entropy import entropy

# Here we defined the Predicate class to store predicates – values in
# particular columns that are used to split our dataset. The class
# includes the divide method, which splits the dataset by the given predicate,
# and the information_gain method, which calculates the information gain for a given split.
class Predicate:
    def __init__(self, column, value):
        self.column = column
        self.value = value

    def divide(self, X, y):
        # Check if the value is numeric and create a boolean filter array
        # based on the "greater than or equal to" condition.
        if # TODO :
            mask = # TODO
        # If the value is not numeric (int or float), create the array based on the
        # "equal to" condition.
        else:
            mask = # TODO
        # Return the results in the following order: X1, y1, X2, y2.
        return # TODO

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
        # return gain