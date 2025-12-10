import numpy as np
from divide import Predicate
from node import Node

class DecisionTree:
    def build(self, X, y):
        self.root = self.build_subtree(X, y)
        return self

    def build_subtree(self, X, y):
        predicate = DecisionTree.get_best_predicate(X, y)
        if predicate:
            X1, y1, X2, y2 = predicate.divide(X, y)
            true_branch = self.build_subtree(X1, y1)
            false_branch = self.build_subtree(X2, y2)
            return Node(column=predicate.column, value=predicate.value,
                        true_branch=true_branch, false_branch=false_branch)
        else:
            unique_y = np.unique(y, return_counts=True)
            return unique_y[np.argmax(unique_y[1])][0]
        pass

    def get_best_predicate(X, y):
        best_predicate = None
        best_gain = 0.0
        column_count = len(X[0])

        for column in range(0, column_count):
            column_values = np.unique(X[:, column])

            for value in column_values:
                predicate = Predicate(column, value)
                gain = predicate.information_gain(X, y)
                if gain > best_gain:
                    best_predicate = predicate
                    best_gain = gain

        return best_predicate

    def __repr__(self):
        return f'Decision Tree: \n{self.root};\n'