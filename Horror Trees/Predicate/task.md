### Predicate


To build a decision tree, in each step, we will need to divide the sample from a tree node
into two independent sub-samples and calculate the entropy of each of them. 
For convenience, we will create a separate class `Predicate` in the `divide.py`; it will store the values
of the characteristic and the threshold value used in division.



### Task

Implement the method `divide`, which allows dividing the sample into two independent
sub-samples according to a certain characteristic. The function should take a dataset 
(`X`) and class labels (`y`). Mind that the dataset contains two types of characteristics –
nominal and quantitative ones. First, the method needs to check if the characteristic is 
quantitative, and if yes, the threshold condition should be an inequality. For nominal characteristics,
the number of predicates equals the number of unique characteristic values, so the threshold condition for
sample division is an equality check for the characteristic. The method should return a dataset and class
labels divided according to this characteristic. 


<div class="hint">

Create a filter array `mask` by comparing the elements in the required column with the threshold condition
and use it to divide the sample. </div>

<div class="hint">

To get the second part of the sample, the one that did not pass the threshold condition, 
you can apply [bit-wise inversion](https://numpy.org/doc/stable/reference/generated/numpy.invert.html) to the created array
for data filtering. When working with arrays, the `~` operator may be used instead of np.invert for the sake of brevity.</div>

At this stage, do not mind the other method of the class – you will need to
implement it in the next task.

In order to see the results of your code, you can add
the following lines to `task.py` and run it:
1. Required imports:
 ```python
        import numpy as np
        from divide import Predicate
```
2. A mockup dataset for checking the work of the `divide` method and outputting the results should be added to the block `main`.
```python
        predicate = Predicate(3, 'clear')           
        X = np.array([[1, 1, 1, 'clear'],
                    [2, 2, 2, 'clear'],
                    [3, 3, 3, 'green'],
                    [1, 2, 3, 'black']])
        y = np.array([1, 2, 3, 4])

        X1, y1, X2, y2 = predicate.divide(X, y)
        print(f'Division result: '
            f'\nFirst group labels: {y1} '
            f'\nFirst group objects: {X1} '
            f'\nSecond group labels: {y2} '
            f'\nSecond group objects: {X2}\n')
```