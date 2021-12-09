To classify an object, we need to go through the tree from the root to the terminal nodes;
in each node, we need to make a decision as regards which subtree we should go to next. The decision
is based on the comparison of the object parameters with the predicate values in
the tree node.

### Task

In the `tree.py` file, the `predict` method of the `DecisionTree` class has been implemented; it takes an object`x` and returns its class. 
You need to implement a recursive method `classify_subtree`, which will be called from `predict` 
and which will take the `х` object and return the classification result (the object class). First, uncomment 
all lines except text comments (be attentive!) and delete the pass operator.
In `classify_subtree`, you need to:

1. Check whether `sub_tree` is an instance of the `Node` class and if yes, return the
   current value of `sub_tree`, as in such a case, it is a class label.
2. Compare the characteristic value from the column according to which a condition is set in the given node with the threshold value.
3. Depending on the result, choose the tree branch along which you will proceed (`true_branch` or `false_branch`).
4. Repeat these actions recursively until the result will be a class label (a leaf node).

<div class="hint">Follow the instructions in the code comments! :)</div>

To see the results of your code, add the following lines
to the `main` block in `task.py` and run it:
```python
print(f'Class prediction for object X[0]: {tree.predict(X[0])}\n')
```
Variables required for the correct work of this code were introduced in the previous steps; in case you haven't worked with
`task.py` yet, pay attention to them.
