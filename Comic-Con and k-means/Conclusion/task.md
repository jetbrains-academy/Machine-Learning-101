The speed of our algorithm significantly depends on the number of clusters. Try launching the program for 4 colors and then for 16 colors – you will see how the execution time changes.

Another specific feature of the method is its sensitivity to the initial choice of cluster centers: the random selection of centroids does not always guarantee a satisfactory result of clustering. If the centroids were badly located, you might see artefacts. In our case, if you launch the algorithm with `k = 4` several times, you will, sooner or later, receive an image with some parts in unexpected colors. Such a problem may be handled by means of consistently improving the method of initial cluster selection; in the end, you can choose the most suitable method.


The algorithm we've described is realised in the [scikit-learn](https://scikit-learn.org/stable/modules/generated/sklearn.cluster.KMeans.html) library. When using the method from scikit-learn, you need to set various parameters we've covered in this lesson: the number of clusters, the number of test launches with different initial clustering, the acceptable error, and the maximum number of iterations.

The **k-means** algorithm is further developed in the [k-means++](https://ru.wikipedia.org/wiki/K-means%2B%2B) algorithm. The difference between the two lies in the initial cluster selection stage.

Another modification is the [X-means](https://en.wikipedia.org/wiki/Determining_the_number_of_clusters_in_a_data_set#:~:text=In%20statistics%20and%20data%20mining,criterion%20(BIC)%20is%20reached) algorithm. This method takes a range of clusters, launches the algorithm for each value, and finds the best result according to the chosen quality metric.

The **k-means** algorithm is also used in more complex tasks of machine learning: for example, in the [Kohonen neural network](http://www.machinelearning.ru/wiki/index.php?title=%D0%9D%D0%B5%D0%B9%D1%80%D0%BE%D0%BD%D0%BD%D0%B0%D1%8F_%D1%81%D0%B5%D1%82%D1%8C_%D0%9A%D0%BE%D1%85%D0%BE%D0%BD%D0%B5%D0%BD%D0%B0).


> <i>This course is currently in the Alpha version. You can help us improve it by answering questions after each task in the following
> <a href="https://docs.google.com/forms/d/e/1FAIpQLSfQxVgSmjAXNyxqoF5O4XxXSYWvHv2UcLsqfyt0MtE6u9820A/viewform?usp=sf_link">form</a>.
> Thanks! :) </i>