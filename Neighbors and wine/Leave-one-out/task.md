### Choosing K value

When `k = 1`, the nearest neighbors algorithm won't be resistant to [noise outliers](https://en.wikipedia.org/wiki/Outlier): it makes classification mistakes not only on the outlier objects
but also on the nearby objects of other classes.

**Noise outliers** are extreme values in the sample, lying far beyond other observations. In our context, these are objects mistakenly assigned to a certain class. What accounts for noise outliers may be errors in the equipment work, human errors while data processing, unique characteristics of some wines, etc.

On the other hand, if `k` equals the size of the sample, the algorithm is excessively stable and degenerates into a constant. In such a case, the classification will depend not on the similarity of wines but on the number of certain wines in the sample. Thus, our sample in the **wine.csv** file contains 59 wines of the first class, 71 wine of the second class, and 48 wines of the third class. If we analyze the most frequently occurring class among 177 nearest neighbors, it will be class 2 for each of the wines.

Consequently, extreme values of `k` need to be avoided.

In practice, the optimum value of `k` is determined by the **leave-one-out** cross-validation technique:
${LOO(k, X^l) = \sum\limits_{i=1}^l [a(x_i; X^l \backslash \\{x_i\\}, k) \neq y_i] \rightarrow \min\limits_k}$.

Here, each portion of the sample $X^l \backslash \\{x_i\\}$ is used as a training sample and $\\{x_i\\}$ itself as the testing sample. If an error occurs while training and the predicted class does not equal
$\\{y_i\\}$, the sum of errors 
for the current $k$ grows.

$k$ with the minimum sum of errors is considered optimal, and we choose the smallest $k$ among all optimal values.

<details>
Let's remember the formula of identifying a class according to $k$ neighbors:
$$
\rho(u,x_1)\leq\rho(u,x_2)\leq...\leq\rho(u,x_l)$$
$x_i$ is the $i$-th neighbor of the object $u$

$y_i$ is the class of the $i$-th neighbor of the object $u$
$$
a(u, X^l) = \arg \max\limits_{y\in Y} \sum\limits_{y_i=y} w(i,u)
$$
$w(i,u) = [i\leq k]$ are the classes of the $i$ nearest neighbors of $u$

$a(u, X^l)$ is the prevalent class among them.
</details>

Having excluded one object from the sample and trained the algorithm on the rest of the objects, we can test the algorithm on the excluded object. The optimal `k` will be the smallest value that provides the maximum number of classes correctly identified in this test.

<details>
If we don't exclude the classified object from the training sample, it will always be its own nearest neighbor, and the minimum value of the $LOO(k)$ function will be received with $k=1$. 
</details>

### Task

Implement a function for choosing the optimal `k` with [leave-one-out cross-validation](https://en.wikipedia.org/wiki/Cross-validation_(statistics)#Leave-one-out_cross-validation).
The function needs to take the training sample and the distance function:

      def loocv(X_train, y_train, dist):
          # ...
          return opt_k

The function template is in the `crossvalidation.py` file. 

Assess the precision and recall of the classifier with the optimal `k` and any two distance functions.

To do that, import `loocv`, `precision_recall`, `print_precision_recall`, `euclidian_dist` and `taxicab_dist` in `task.py` and combine all our functions in `main`:
```python
if __name__ == '__main__':
    wines = np.genfromtxt('wine.csv', delimiter=',')

    X, y = wines[:, 1:], np.array(wines[:, 0], dtype=np.int32)
    X_train, y_train, X_test, y_test = train_test_split(X, y, 0.6)
    y_euclidean_predicted = knn(X_train, y_train, X_test, 5, euclidean_dist)
    print_precision_recall(precision_recall(y_euclidean_predicted, y_test))

    euclidean_opt = loocv(X_train, y_train, euclidean_dist)
    taxicab_opt = loocv(X_train, y_train, taxicab_dist)

    print("optimal euclidian k = " + str(euclidean_opt))
    print("optimal taxicab k = " + str(taxicab_opt))

    y_euclidean_predicted = knn(X_train, y_train, X_test, euclidean_opt, euclidean_dist)
    print_precision_recall(precision_recall(y_euclidean_predicted, y_test))

    y_taxicab_predicted = knn(X_train, y_train, X_test, taxicab_opt, euclidean_dist)
    print_precision_recall(precision_recall(y_taxicab_predicted, y_test))
```
