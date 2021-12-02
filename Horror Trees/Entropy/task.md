A decision tree is built from top to bottom starting from the root node. The data are divided into subsets
containing similar characteristics (homogeneous subsets).
The algorithm uses the concept of [entropy](https://ru.wikipedia.org/wiki/%D0%98%D0%BD%D1%84%D0%BE%D1%80%D0%BC%D0%B0%D1%86%D0%B8%D0%BE%D0%BD%D0%BD%D0%B0%D1%8F_%D1%8D%D0%BD%D1%82%D1%80%D0%BE%D0%BF%D0%B8%D1%8F) 
to determine the homogeneity of a sample.

The value of entropy may be greater than or equal to 0. This value is calculated by the following formula:


$$H = - \sum\limits_{i=1}^{C} p_i \log_2 p_i$$

where $p_i$ is the possible outcome probability (proportion) of an object or a class $i$ in the sample and $С$ is the number of classes.

An interesting example of information entropy application may be found in biodiversity assessment scholarly research. 
Using the Shannon formula (the above equation), where $p_i$ is the proportional abundance
of the $i$-th species in a group of $С$ different species, we can calculate [Shannon's diversity index](https://ru.wikipedia.org/wiki/%D0%9C%D0%B5%D1%80%D0%B0_%D1%80%D0%B0%D0%B7%D0%BD%D0%BE%D0%BE%D0%B1%D1%80%D0%B0%D0%B7%D0%B8%D1%8F).



### Task

In the file `calculate_entropy.py`, realize the function `entropy`, which calculates the entropy for a certain subset of objects. 
The input of the function is an array of object labels `y`.

First, the function should calculate the occurrence of each class (label) in the whole subset.
Here, you might want to use the [numpy.unique](https://numpy.org/doc/stable/reference/generated/numpy.unique.html) function, which 
can return the sorted unique elements of a subset and the number of times each of these elements occurs in the original set.
Then, you need to calculate the proportion of each class in the whole set and subsequently
use the above formula to calculate the entropy.


In order to see the results of your code, you can add the following lines to
`task.py` and launch it:

1. Required imports:
```python
        from calculate_entropy import entropy
        import pandas as pd
```
2. Lines in the `if __name__ == '__main__':` block for printing the dataset entropy calculation results:
```python
        X, y, columns = read_data("halloween.csv")
        print(f'dataset entropy: {entropy(y)}\n')
```
In the file, the block `if __name__ == '__main__':` should be put **after** all the functions and variables it uses!