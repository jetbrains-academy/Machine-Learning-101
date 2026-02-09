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
for data filtering. When working with arrays, the `~` operator may be used instead of `np.invert` for the sake of brevity.</div>

To see the results of your code, you can run `task.py`.
