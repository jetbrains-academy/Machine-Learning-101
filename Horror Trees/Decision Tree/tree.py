import numpy as np
from divide import Predicate
from node import Node

class DecisionTree:
    def build(self, X, y):
        self.root = # TODO: Build the tree using build_subtree
        return # self

    def build_subtree(self, X, y):
        """ TODO: Split by predicate, recursively build subtrees,
        and return a Node or the majority class """
        pass

    def get_best_predicate(X, y):
        """ TODO: Build predicates for the attribute, compute information
        gain, and return the optimal one """
        pass

    def predict(self, x):
        return self.classify_subtree(x, self.root)

    def classify_subtree(self, x, sub_tree):
        """ TODO: Recursively follow the correct branch until a leaf
        node is reached and return the class label """
        pass

    def __repr__(self):
        return f'Decision Tree: \n{self.root};\n'