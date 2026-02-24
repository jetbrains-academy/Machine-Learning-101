import numpy as np
from divide import Predicate
from node import Node


# Here, we define the DecisionTree class to store the tree data. It includes
# the build method, which assigns the resulting subtree to the root attribute.
# The get_best_predicate static method evaluates all possible predicates for a given trait, returning
# the one that maximizes information gain. The result is used in the
# build_subtree method to split the dataset by the best predicate, recursively
# building the left and right subtrees. The predict method accepts an object and returns its class label.
# The predict method calls the recursive classify_subtree function, which accepts an object and a subtree to
# perform the classification and return the result.
class DecisionTree:
    def build(self, X, y):
        self.root = self.build_subtree(X, y)
        return self

    def build_subtree(self, X, y):
        # Get the most informative predicate using the get_best_predicate method
        predicate = DecisionTree.get_best_predicate(X, y)

        # if such predicate is found:
        if predicate:
            # Split the sample using the Predicate class's divide method
            X1, y1, X2, y2 = predicate.divide(X, y)
            # Build subtrees recursively:
            true_branch = self.build_subtree(X1, y1)
            false_branch = self.build_subtree(X2, y2)
            # Return the tree as an instance of Node
            return Node(column=predicate.column, value=predicate.value,
                        true_branch=true_branch, false_branch=false_branch)
        # If no suitable predicate is found, return the most common class label
        else:
            unique_y = np.unique(y, return_counts=True)
            return unique_y[np.argmax(unique_y[1])][0]

    def get_best_predicate(X, y):
        best_predicate = None
        best_gain = 0.0
        column_count = len(X[0])

        # Iterate over columns to find the value that forms the best predicate
        for column in range(0, column_count):
            # Get the unique values within the current column
            column_values = np.unique(X[:, column])
            # Iterate over the unique values in the column, using
            # the Predicate class to calculate the information gain for each value
            for value in column_values:
                predicate = Predicate(column, value)
                gain = predicate.information_gain(X, y)
                if gain > best_gain:
                    # Assign new values
                    best_predicate = predicate
                    best_gain = gain

        return best_predicate

    # The following methods are to be implemented in the task "Predict"
    def predict(self, x):
        return self.classify_subtree(x, self.root)

    def classify_subtree(self, x, sub_tree):
        pass
        # Return sub_tree if it is a leaf node (a class label) rather than a Node instance
        # if # TODO
            # Return sub_tree
        # else:
            # Take the value of object x in the sub_tree column
            # v = # TODO
            # Check if it is a numeric value
            # if isinstance(v, int) or isinstance(v, float):
                # If value v fits the numeric condition at the given node, proceed
                # to the true_branch, if not - to the false_branch
                # if v >= # TODO
                    # branch = # TODO
                # else:
                    # branch = # TODO
            # If v is a non-numeric value, compare it for equality with the node's condition,
            # and proceed with the same recursive logic as described in the numeric case
            # else:
                # if v == # TODO
                    # branch = # TODO
                # else:
                    # branch = # TODO
            # Repeat this process recursively for the new branch
            # return # TODO

    # Below, we define a __repr__ method to provide a human-readable
    # representation of Decision Tree instances
    def __repr__(self):
        return f'Decision Tree: \n{self.root};\n'