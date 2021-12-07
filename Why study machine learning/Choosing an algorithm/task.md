There are several types of machine learning tasks:

### [Supervised learning](http://www.machinelearning.ru/wiki/index.php?title=%D0%9E%D0%B1%D1%83%D1%87%D0%B5%D0%BD%D0%B8%D0%B5_%D1%81_%D1%83%D1%87%D0%B8%D1%82%D0%B5%D0%BB%D0%B5%D0%BC)
Such tasks contain a set of objects (stimuli) and responses to them (reactions). For example, a response might be an object's belonging to a certain class of objects. However, the dependence between the stimuli and the responses is unknown. To find it from the empirical data ([examples](https://ru.wikipedia.org/wiki/%D0%9E%D0%B1%D1%83%D1%87%D0%B5%D0%BD%D0%B8%D0%B5_%D0%BD%D0%B0_%D0%BF%D1%80%D0%B8%D0%BC%D0%B5%D1%80%D0%B0%D1%85) – stimuli with a known response), an algorithm is trained, which can provide a sufficiently accurate answer for any object.

The set of precedents used for algorithm training is called a [training sample](http://www.machinelearning.ru/wiki/index.php?title=%D0%9E%D0%B1%D1%83%D1%87%D0%B0%D1%8E%D1%89%D0%B0%D1%8F_%D0%B2%D1%8B%D0%B1%D0%BE%D1%80%D0%BA%D0%B0).

#### [Classification](http://www.machinelearning.ru/wiki/index.php?title=%D0%9A%D0%BB%D0%B0%D1%81%D1%81%D0%B8%D1%84%D0%B8%D0%BA%D0%B0%D1%86%D0%B8%D1%8F)
In this kind of tasks, there is a set of objects (situations) divided into classes. The training sample is a finite set of objects belonging to known classes. An example of such a task is medical diagnostics: on the grounds of formally described medical analyses, we can assume the presence of a certain disease.

You can find examples of solving such tasks in the following lessons: **Neighbors and Wine**, **Horror Trees**, **Pima indians diabetes and linear classifier**, and **Iris Network**.

#### [Regression](http://www.machinelearning.ru/wiki/index.php?title=%D0%A0%D0%B5%D0%B3%D1%80%D0%B5%D1%81%D1%81%D0%B8%D1%8F)
This task involves building a model of measurable data and studying their properties. The data include pairs of values: a dependable variable and an independent variable. The result is a model explaining the dependence of the response variable on the explanatory variable. An example is a model predicting the dependence of real estate prices on time.

#### [Learning to rank](http://neerc.ifmo.ru/wiki/index.php?title=%D0%A0%D0%B0%D0%BD%D0%B6%D0%B8%D1%80%D0%BE%D0%B2%D0%B0%D0%BD%D0%B8%D0%B5)
In these tasks, it is necessary to sort the values of responses received for a set of objects. The solution may involve a classification or a regression task. This type of problems may occur in an information search or in text analysis.

#### Structured learning
It is an umbrella term for tasks of supervised machine learning tasks which involve predicting complex structural objects rather than distinct or real scalar values. An example is the tree of syntax analysis in natural language processing.

### [Unsupervised learning](http://www.machinelearning.ru/wiki/index.php?title=%D0%9E%D0%B1%D1%83%D1%87%D0%B5%D0%BD%D0%B8%D0%B5_%D0%B1%D0%B5%D0%B7_%D1%83%D1%87%D0%B8%D1%82%D0%B5%D0%BB%D1%8F)
In this class of tasks, we know only the description of a set of objects (the learing sample), and based on that, we need to find out the dependencies between the objects.

#### [Data clustering](http://www.machinelearning.ru/wiki/index.php?title=%D0%9A%D0%BB%D0%B0%D1%81%D1%82%D0%B5%D1%80%D0%B8%D0%B7%D0%B0%D1%86%D0%B8%D1%8F)
The task involves grouping objects into clusters using the information of pairwise similarity between the objects.
An example of such a task is considered in the **Comic-Con and k-means** lesson.

#### Finding association rules
The given data represent feature descriptions. It is necessary to find such sets of features and such feature values most frequently (non-randomly) occur in the object feature descriptions.

### [Semi-supervised learning](http://www.machinelearning.ru/wiki/index.php?title=%D0%A7%D0%B0%D1%81%D1%82%D0%B8%D1%87%D0%BD%D0%BE%D0%B5_%D0%BE%D0%B1%D1%83%D1%87%D0%B5%D0%BD%D0%B8%D0%B5)
This type is in between supervised and unsupervised learning tasks. It involves both labeled and unlabeled data. It happens if the learning sample contains answers only to some precedents. An example is automatic rubrication of a large number of texts in the case when some of those have already been assigned to certain rubrics.

An example of solving such tasks is considered in the **Bayes guards SMS** lesson.

### [Active learning](https://en.wikipedia.org/wiki/Active_learning_(machine_learning)))
A distinctive feature of this type is that the learning algorithm can itself choose the next training subset with a desired answer: for example, an object with a predefined class. This approach may be used in the cases with huge volumes of data where defining classes for each object manually is time-consuming. One of examples of using active learning is image analysis in astronomy – mapping objects on planet surfaces, classifying stars or galaxies, etc. – when the identification of objects in abundant images requires costly professional work.


> <i>This course is currently in the Alpha version. You can help us improve it by answering questions after each task in the following
> <a href="https://docs.google.com/forms/d/e/1FAIpQLSetRXApA3fyEOLne7Ag5lkYrP7nuXfVP_M8bUNJGcDEQjP9kg/viewform?usp=sf_link">form</a>.
> Thanks! :) </i>