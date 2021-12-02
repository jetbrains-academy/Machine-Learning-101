Is it true that all wines are just the same? Wines differ in the shades of taste and smell and in strength, and all these characteristics may be measured.

The `k`-nearest neighbours algorithm allows classifying wines on the grounds of their measured characteristics. If you create more classes and collect additional data on their characteristics, you can use the algorithm to analytically predict whether you will like a certain wine (whether it belongs to a class of wines you liked before).

In this lesson, we've discussed the principles and parameters of an algorithm utilizing the `k`-nearest neighbours method. You can find further information about the method in the following resources:
- [An article on machinelearning.ru](http://www.machinelearning.ru/wiki/index.php?title=%D0%9C%D0%B5%D1%82%D0%BE%D0%B4_%D0%B1%D0%BB%D0%B8%D0%B6%D0%B0%D0%B9%D1%88%D0%B8%D1%85_%D1%81%D0%BE%D1%81%D0%B5%D0%B4%D0%B5%D0%B9) contains some additional mathematical formulas.
- [A video on the Simplilearn channel](https://www.youtube.com/watch?v=4HKqjENq9OU) describes and comments on the process of the algorithm development.
- [A video from the HSE course "Introduction to machine learning"](https://www.coursera.org/lecture/vvedenie-mashinnoe-obuchenie/mietod-blizhaishikh-sosiediei-jCkvu) discusses examples of using the algorithm, among other things.

The `k`-nearest neighbours algorithm is realized in the [scikit-learn](https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KNeighborsClassifier.html) library. When using it, you need to indicate a number of parameters (the number of neighbours `k`, the weight of each neighbour, the distance function, etc.); you can find the information on the parameters in the algorithm description. As a rule, complex parameters are optional and take default values.

Launching the algorithm with different `k` values, you can see that the processing time grows proportionally to this parameter: there is a linear dependence of the [algorithm complexity](https://ru.wikipedia.org/wiki/%D0%92%D1%80%D0%B5%D0%BC%D0%B5%D0%BD%D0%BD%D0%B0%D1%8F_%D1%81%D0%BB%D0%BE%D0%B6%D0%BD%D0%BE%D1%81%D1%82%D1%8C_%D0%B0%D0%BB%D0%B3%D0%BE%D1%80%D0%B8%D1%82%D0%BC%D0%B0) on `k` (provided that our data may be easily divided into clusters). There are also faster versions of the algorithm, you can find them listed [here.](https://en.wikipedia.org/wiki/K-means_clustering#Variations).

> <i>This course is currently in the Alpha version. You can help us improve it by answering questions after each task in the following
> <a href="https://docs.google.com/forms/d/e/1FAIpQLSfix9bjakXkVGr7c0ErZWzzIdGUUAGwASokBj8CB0ql0s5HWA/viewform?usp=sf_link">form</a>.
> Thanks! :) </i>