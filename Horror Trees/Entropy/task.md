A decision tree is built from top to bottom starting from the root node. The data are divided into subsets
containing similar characteristics (homogeneous subsets).
The algorithm uses the concept of [entropy](https://en.wikipedia.org/wiki/Entropy_(information_theory)) 
to determine the homogeneity of a sample.

The value of entropy may be greater than or equal to 0. This value is calculated by the following formula:


$$H = - \sum\limits_{i=1}^{C} p_i \log_2 p_i$$

where $p_i$ is the possible outcome probability (proportion) of an object or a class $i$ in the sample and $С$ is the number of classes.

An interesting example of information entropy application may be found in research involving biodiversity assessment.
Using the Shannon formula (the above equation), where $p_i$ is the proportional abundance
of the $i$-th species in a group of $С$ different species, we can calculate [Shannon's diversity index](https://en.wikipedia.org/wiki/Diversity_index).



### Task

In the file `calculate_entropy.py`, implement the function `entropy`, which calculates the entropy for a certain subset of objects. 
The input of the function is an array of object labels `y`.

First, the function should calculate the occurrence of each class (label) in the whole subset.
Here, you might want to use the [numpy.unique](https://numpy.org/doc/stable/reference/generated/numpy.unique.html) function, which 
can return the sorted unique elements of a subset and the number of times each of these elements occurs in the original set.
Then, you need to calculate the proportion of each class in the whole set and subsequently
use the above formula to calculate the entropy.


In order to see the results of your code, you can add the following lines to
`task.py` and run it:

1. Required import:
```python
        from calculate_entropy import entropy
```
2. Lines in the `main` block for printing the dataset entropy calculation results:
```python
        X, y, columns = read_data("halloween.csv")
        print(f'dataset entropy: {entropy(y)}\n')
```
In the file, the block `if __name__ == '__main__':` should be put **after** all the functions and variables it uses!