### K-means

In this lesson, we will discuss the clustering task – breaking the sample into non-overlapping sets of objects that are similar to each other but different from the objects of other sets. In our task, the objects will be the image pixels, and their characteristics – the values of color components. The set centers will be defined by the colors we've substituted for the original colors. For example, different shades of red – from bright red to burgundy – may be combined into one averaged red, which will be the center of the cluster.


The k-means clustering algorithm belongs to unsupervised learning algorithms, which are used when we do not know in advance into what clusters we will divide our objects. We do not know in advance in what colors our image will be transformed, and the task of the algorithm is to figure out these colors. In our case, the algorithm has to divide the sample into
`k = 8` groups.

Essentially, the algorithm iteratively assigns each sample pixel to one of `k` groups on the grounds of certain characteristics (in our case, it is the original colors). Pixels with similar characteristics form a cluster.

The algorithm utilizes iterative result improvement. The input is the number of clusters and a data set containing the characteristics of each pixel (in our case, the values of red, blue, and green channels). Clusters are characterized by their centers, aka centroids. First, the algorithm defines the `k` centroids randomly or by arbitrarily choosing them from the input data. A [centroid](https://en.wikipedia.org/wiki/Centroid) is the assumed geometric center of a cluster. Here, we are talking about coordinates in the value space of a characteristic, so the common centroid for the black (0, 0, 0) and white (255, 255, 255) colors will be the grey spot (127, 127, 127). At the initial stage, we don't need it to be close to the real center (although, it might improve the work of the algorithm). With each iteration, we will make the centroids approach their real values.

Next, we will iteratively repeat two following steps:

1. The point distribution step: we assign each point to a cluster with the closest centroid. Or, more formally:

$$c_i = \underset{{c \in 1\dots k}}{\arg\min}  \rho(x_i, \mu_c)$$

where
- $c_i$ is the center of a cluster assigned to the $x_i$ point;
- $\rho(x_i, \mu_c)$ is the distance between the $x_i$ point and the cluster center $\mu_c$;
- $\mu_{c}$ is the cluster center.

2. The centroid updating step: the centroids are recalculated. For each cluster, the new centroid is the average of all its points.

$$ {\mu_{c} = \frac{\sum\limits_{j=1,\dots, n} [c_i = c] x_i^j}{\sum\limits_{c_i = c} 1} } $$


The algorithm repeats steps 1 and 2 until one of the stopping conditions is met: either in the latest iteration none of the points changed its cluster, or the minimum sum of distances was received, or the number of iterations reached a predefined limit. The stopping condition here is the moment when we consider the work of the algorithm completed. Without such a condition, it will continue iterating infinitely.
Such conditions guarantee that the algorithm converges. The stricter the conditions, the longer the algorithm will proceed with calculations. It's a good idea to vary the conditions: if the productivity is insufficient, you can limit the number of iterations. Sometimes, it's worthwhile reconsidering the initial conditions of the task (e.g., we can try building 4 clusters instead of 8).

You need to take into account that if we've chosen totally wrong centroids or an excessively high `k` number, step 1 can result in a cluster no point can be assigned to. Such cases must be processed separately in stage 2 in order to avoid errors in calculating he average. Several approaches are possible: for example, defining the centroid for an empty cluster, we can assign it a random point or a point most distanced from the center of the largest cluster available.

To calculate distances, here we use the [Euclidean distance](https://en.wikipedia.org/wiki/Euclidean_distance); however, depending on the task, different methods may be used. For example, in a text clustering task, we could use the [Levenshtein distance](https://en.wikipedia.org/wiki/Levenshtein_distance). You can find the function for calculating the Euclidean distance in the `distances.py` file.



### Task

Implement the function `k_means(X, n_clusters, distance_metric)`, which takes an $X$ matrix of the order `(n_samples, n_features)`, the number of clusters we want to break the image into, and a metric.

The result of the function is a pair of vectors: the vector of size `n_samples`, where the $i$-th cell contains the number of the cluster corresponding to the $i$-th pixel, and the vector of size `(n_clusters)` with cluster centers.

You can find the function template in the `clustering.py` file. The function `init_clusters`, which creates the initial centroids for the given data set, is also there.

While doing the task, you might need the [numpy.sum](https://numpy.org/doc/1.18/reference/generated/numpy.sum.html) function, which calculates the sum of array elements.

To see the results of your code, add the following lines in `task.py` and run it:
1. Necessary imports:
 ```python
from distances import euclidean_distance
from clustering import k_means  
```
2. Add the lines for printing the results in the `main` block instead of `print(image)`:
```python
(centroids, labels) = k_means(image, 4, euclidean_distance)
print("Cluster centers:")
for label in labels:
    print(label)
```

