####Information Gain

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

In the file `divide.py`, in the `information_gain` method of the `Predicate` class, delete the `pass` operator, 
uncomment all lines with `# TODO` and the line with `return`. 

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
Variables required for the correct work of this code were introduced in the previous steps; in case you haven't worked with 
`task.py` yet, pay attention to them.