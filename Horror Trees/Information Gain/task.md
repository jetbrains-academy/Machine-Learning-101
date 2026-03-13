### Information Gain

In information theory, [Information Gain](https://en.wikipedia.org/wiki/Information_gain_in_decision_trees) is
the amount of information about a random value received while observing another random value. 
It is based on the decrease of entropy after dividing a sample according to a certain characteristic, 
and it allows assessing the division quality. In other words, 
the expected information gain is the change of information entropy H from the previous state to a state
that takes certain information (a condition) as given.
To build a decision tree is to find a characteristic (i.e., the homogenous branches) that 
provides the largest amount of information.

$$IGain = H(parent_y) - H(children_{y|x}) $$

We subtract entropy `Y` for the condition `X` from entropy `Y` to calculate the indeterminacy decrease of
`Y`, provided that there is some additional knowledge `X` about `Y`.



### Task

Implement the `information_gain` method, which 
takes a sample, divides it into two independent sub-samples, and calculates the information gain.
To divide the sample, use the `divide` method written
in the previous step. 

<div class="hint">

To calculate information gain, you can use the above formula in the following way:

`gain = entropy(y) - p * entropy(y1) - (1 - p) * entropy(y2)`

where:
- `y` is all object labels (parent);
- `y1` is one of the two label sets (child) received after division;
- `y2` is the second one of the two label sets (child) received after division;
- `p` is the proportion of the objects of the first child set among all objects of the whole dataset (consequently, `1 - p` is the proportion of the second one).
</div>

To see the results of your code, you can add the following line to the
`main` block in `task.py` and run it:

```python
    print(f'Information Gain: {predicate.information_gain(X, y)}\n')     
```
The variables required for this code to function were introduced in previous steps. If you haven't worked with
`task.py` yet, ensure they are properly defined.

<div class="hint">

Just use `divide` method of the `Predicate` class to divide the sample into two subsets.

`X1, y1, X2, y2 = self.divide(X, y)`

</div>

<div class="hint">
After the split, each subset contributes to the final entropy proportionally to its size. Therefore we compute the fraction of 
samples that went to the first subset.

Use `float(...)` to ensure that the division produces a floating-point number (a probability between 0 and 1) rather than an integer.

`p = float(len(X1)) / len(X)`

</div>
