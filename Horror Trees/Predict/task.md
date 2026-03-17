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
    
<div class="hint">

If sub_tree is not a Node, return it — it already represents the class label.

```python
if not isinstance(sub_tree, Node):
    return sub_tree
```
</div>

2. Compare the characteristic value from the column according to which a condition is set in the given node with the threshold value.

<div class="hint">

Each node checks one feature of the object being classified. The node stores which feature to check in `sub_tree.column`.
You need to take that value from `x`.

```python
v = x[sub_tree.column]
```

</div>

3. Depending on the result, choose the tree branch along which you will proceed (`true_branch` or `false_branch`).

<div class="hint" title="Compare numeric features">

For numeric features, the node applies a threshold (e.g., `age >= 30`), so compare the value with the threshold to choose the branch.

```python
if isinstance(v, int) or isinstance(v, float):
    if v >= sub_tree.value:
        branch = sub_tree.true_branch
    else:
        branch = sub_tree.false_branch
```

</div>

<div class="hint" title="Compare categorical features">

For categorical features, the node checks equality (e.g., `color == "red"`).
Choose the branch based on whether the value matches.

```python
else:
    if v == sub_tree.value:
        branch = sub_tree.true_branch
    else:
        branch = sub_tree.false_branch
```

</div>

4. Repeat these actions recursively until the result will be a class label (a leaf node).

<div class="hint">

After choosing a branch, you are not done — there may still be another node.
Apply the same logic again by calling the function on the new branch until you reach a leaf (class label).

```python
return self.classify_subtree(x, branch)
```
</div>

To see the results of your code, add the following lines
to the `main` block in `task.py` and run it:
```python
print(f'Class prediction for object X[0]: {tree.predict(X[0])}\n')
```
The variables required for this code to function were introduced in previous steps. If you haven't worked with
`task.py` yet, ensure they are properly defined.
