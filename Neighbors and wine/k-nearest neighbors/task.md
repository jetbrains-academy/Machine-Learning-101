The k-nearest neighbors method is based on the assumption that similar objects are always located close to each other. Here we're talking about tha "location" of features in the measured trait coordinate space: imagine coordinate axes with alcohol percentage, acidity, and color saturation values. We assume that the object being classified belongs to the same class as the training sample objects closest to it.

The proximity of objects (or, rather, their **similarity**) is determined by a certain metric; hence, the algorithm is a [similarity-based](https://en.wikipedia.org/wiki/Similarity_learning) classifier.

For example, the **zinfandel** and **primitivo** varieties of grapes used for wine making have a common ancestor and these wines, according to many metrics, belong to the same class. Consequently, wine produces often mark these varieties as interchangeable on wine package.

<details>The assumption of similarity between such objects is called a <b>compactness hypothesis</b>. It says that similar objects more often belong to the same class than to different classes.</details>

The k-nearest neighbors classifier does not imply a separate training procedure: we immediately proceed to classifying. Classifying an object involves the following steps:

- Calculate the distance between the object and all other objects in the sample;
- Sort all objects according to the distance $\rho$ from the classified object in ascending order (from least to greatest);
  $$
  \rho(u,x_1)\leq\rho(u,x_2)\leq...\leq\rho(u,x_l)$$
  where $x_i$ is the $i$-th neighbor of the object $u$.
- Select the first $K$ objects from the sorted list;
- Return the most frequent class label among these $K$ objects. The label here is the class identifier (in our case, the wine variety identifier).

$$
a(u, X^l) = \arg \max\limits_{y\in Y} \sum\limits_{y_i=y} w(i,u)
$$
where:
$y_i$ is the class of the $i$-th neighbor of $u$,

$w(i,u) = [i\leq k]$ are the classes of $i$ nearest neighbors of $u$, 

$a(u,X^l)$ is the most frequent one among them.


To calculate the distance between objects, you can use one of the two functions: [Euclidean distance](https://en.wikipedia.org/wiki/Euclidean_distance) or [taxicab geometry](https://en.wikipedia.org/wiki/Taxicab_geometry). Our solution uses the Euclidean distance, but you can also try using the taxicab distance to check how the classification result changes. Besides, there are other distance functions, too; the choice of the function depends on the particular task. You can find them in the `distances.py` file.

When choosing the metric, you need to try to maximize the sum of distances between objects from different classes and minimize the distances between objects within the same class. Then, different classes will be located far from each other and similar classes – next to each other.

Thus, for example, strong wines with high acidity will be assigned to the same class, while low-alcohol wines with a less distinct taste – to a different one.
### Task

Implement a function which will predict class identifiers based on existing examples. The `knn` function must take:
- the training sample `X_train`, `y_train`;
- the classifiable sample `X_test`;
- the number of neighbors `k`;
- the distance function `dist`.

The function result will be a vector`y_test`, which stores the classes assigned to each element of `X_test` respectively.

```python
def knn(X_train, y_train, X_test, k, dist):
    return #class for each x in the X_test
```

The function template may be found in the`metric_classification.py` file.

In this task, you may use the following functions from the **NumPy** library:
- [numpy.argpartition](https://docs.scipy.org/doc/numpy/reference/generated/numpy.argpartition.html): it takes an array and a `k` index and returns a new array, where the `k` position is occupied by the element from the `k` position in the sorted original array; the lesser values will be to the left of `k` and the greater values – to the right of it. This function will help you select the nearest neighbors in an array sorted according to the distance.
- [numpy.bincount](https://docs.scipy.org/doc/numpy/reference/generated/numpy.bincount.html): it returns the number of different elements in an array of non-negative numbers. This function will help you find out the number of classes among the neighbors.
- [numpy.argmax](https://docs.scipy.org/doc/numpy/reference/generated/numpy.argmax.html): it returns the indices of the elements with the greatest value. This function will help you find out the most frequent class identifiers.
