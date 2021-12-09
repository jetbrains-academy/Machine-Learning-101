Is it true that all wines are just the same? Wines differ in the shades of taste and smell and in strength, and all these characteristics may be measured.

The `k`-nearest neighbors algorithm allows classifying wines on the grounds of their measured characteristics. If you create more classes and collect additional data on their characteristics, you can use the algorithm to analytically predict whether you will like a certain wine (whether it belongs to a class of wines you liked before).

In this lesson, we've discussed the principles and parameters of an algorithm utilizing the `k`-nearest neighbors method. You can find further information about the method in the following resources:
- [The article on wikipedia](https://en.wikipedia.org/wiki/K-nearest_neighbors_algorithm) contains some additional mathematical formulas.
- [The video on the Simplilearn channel](https://www.youtube.com/watch?v=4HKqjENq9OU&ab_channel=Simplilearn) describes and comments on the process of the algorithm development.
- [This article](https://towardsdatascience.com/k-nearest-neighbors-knn-algorithm-23832490e3f4) discusses examples of applications of the algorithm, among other things.

The `k`-nearest neighbors algorithm is implemented in the [scikit-learn](https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KNeighborsClassifier.html) library. When using it, you need to indicate a number of parameters (the number of neighbors `k`, the weight of each neighbor, the distance function, etc.); you can find the information on the parameters in the algorithm description. As a rule, complex parameters are optional and take default values.

Running the algorithm with different `k` values, you can see that the processing time grows proportionally to this parameter: there is a linear dependence of the [algorithm complexity](https://en.wikipedia.org/wiki/Time_complexity) on `k` (provided that our data may be easily divided into clusters). There are also faster versions of the algorithm, you can find them listed [here](https://en.wikipedia.org/wiki/K-means_clustering#Variations).
