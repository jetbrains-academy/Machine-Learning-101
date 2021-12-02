class Node:
    def __init__(self, column=-1, value=None, true_branch=None, false_branch=None):
        self.column = column
        self.value = value
        self.true_branch = true_branch
        self.false_branch = false_branch

    def __repr__(self):
        return f'column: {self.column};\n' \
               f'value: {self.value};\n' \
               f'true branch: {self.true_branch};\n' \
               f'false branch: {self.false_branch}'
