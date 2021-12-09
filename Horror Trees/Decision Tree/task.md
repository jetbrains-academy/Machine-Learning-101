### ID3

The basic algorithm for building decision trees is called [ID3](https://en.wikipedia.org/wiki/ID3_algorithm). ID3 
recursively goes through all branches that are not leaves until it classifies all the data.

1. The dataset is divided according to different characteristics. 
   For each branch, entropy is calculated. The received values are proportionally summarized as the total division entropy. After subtracting it from the entropy before the division, we get the information gain, i.e., the decrease of entropy.
    
2. The characteristic with the largest information gain becomes the decisive node, and the dataset is divided on the basis of objects' possessing or not possessing that characteristic. The process is repeated for each branch.
   
3. A branch with the `0` entropy is a leaf.
 Branches with the entropy greater than `0` require further division.



### Task


In the `tree.py` file, implement a recursive algorithm for building a decision tree in the
`build` method of the `DecisionTree` class. You will need to implement two additional methods:
`build_subtree` and `get_best_predicate`.
1. In the `get_best_predicate` method, build all possible predicates for a specific characteristic.
   To do that, you need to find the unique values of the given characteristic.
2. Assess the information gain of all possible predicates for all characteristics on the basis of entropy.
3. Choose the predicate that provides the best division in terms of information gain.
   The `get_best_predicate` method should return this predicate as an instance of the
   `Predicate` class.
4. In the `build_sutree` method, divide the sample according to the chosen predicate using the
   `divide` method of the `Predicate` class; recursively build the right and the left subtrees. The `build_sutree`
   method returns an instance of the `Node` class in case the best predicate was found.
   If not, the method returns the most frequently occurring class label.
5. The `build` method returns `self`.
 
At this stage, don't mind the rest of class methods – you will need to implement them
in the next task. 

To see the results of your code, add the following lines
to `task.py` and run it:
1. Required import:
 ```python
from tree import DecisionTree  
```
2. Add the lines for result output to the `main` block.
```python
tree = DecisionTree().build(X, y) 
print(f'{tree}\n')
```
Variables required for the correct work of this code were introduced in the previous steps; in case you haven't worked with
`task.py` yet, pay attention to them.