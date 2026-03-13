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


To see the results of your code, add the following lines
to the `main` block in `task.py` and run it:
```python
print(f'Class prediction for object X[0]: {tree.predict(X[0])}\n')
```
The variables required for this code to function were introduced in previous steps. If you haven't worked with
`task.py` yet, ensure they are properly defined.

<div class="hint">

A decision tree predicts a class by **moving from the root to a leaf**.  
If the current element is already a class label (not a `Node`), the traversal is finished.

```python
if not isinstance(sub_tree, Node):
    return sub_tree
```

</div>

<div class="hint">

Each node checks one feature of the object being classified.  
The node stores which feature to check in `sub_tree.column`.
So you need to take that value from `x`.

```python
v = x[sub_tree.column]
```

</div>

<div class="hint">

Some features are numeric (for example, age or income).  
In this case the node represents a **threshold condition**, such as:

`age >= 30`

So if the feature value is numeric, compare it with the node’s threshold to decide which branch to follow.

```python
if isinstance(v, int) or isinstance(v, float):
    if v >= sub_tree.value:
        branch = sub_tree.true_branch
    else:
        branch = sub_tree.false_branch
```

</div>

<div class="hint">

Other features may be categorical (for example, color or country).
In this case the node checks **equality** with the stored value.

For example: `color == "red"`

Choose the branch depending on whether the value matches the condition.

```python
else:
    if v == sub_tree.value:
        branch = sub_tree.true_branch
    else:
        branch = sub_tree.false_branch
```

</div>

<div class="hint">

After choosing the next branch, the classification is **not finished yet**.  
You have only moved **one step down the tree**.

The selected branch may still be another node that asks the next question.
So you need to **repeat the same process** for the new subtree.

The easiest way to repeat the same logic is to **call the same function again** with the new branch.

```python
return self.classify_subtree(x, branch)
```
The recursion continues until a leaf node (class label) is reached, which is then returned.
</div>