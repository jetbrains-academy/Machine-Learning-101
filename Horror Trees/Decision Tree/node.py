# Here, we define the Node class to store tree nodes with the following attributes:
# the predicate, consisting of a column index (the feature)
# and the threshold value used to partition the data;
# the true and false branches.
class Node:
    def __init__(self, column=-1, value=None, true_branch=None, false_branch=None):
        self.column = column
        self.value = value
        self.true_branch = true_branch
        self.false_branch = false_branch

    # Below, we define the __repr__ method to provide a human-readable representation of Node instances.
    def __repr__(self):
        return f'column: {self.column};\n' \
               f'value: {self.value};\n' \
               f'true branch: {self.true_branch};\n' \
               f'false branch: {self.false_branch}'
