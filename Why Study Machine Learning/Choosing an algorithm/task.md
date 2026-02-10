## Choosing an Algorithm

There are several types of machine learning tasks:

### [Supervised learning](https://en.wikipedia.org/wiki/Supervised_learning)
Such tasks contain a set of objects (stimuli) and responses to them (reactions). For example, a response might be an object's belonging to a certain class of objects. However, the dependence between the stimuli and the responses is unknown. To find it from the empirical data ([instances](https://en.wikipedia.org/wiki/Instance-based_learning) – stimuli with a known response), an algorithm is trained, which can provide a sufficiently accurate answer for any object.

The set of instances used for algorithm training is called a [training set](https://en.wikipedia.org/wiki/Training,_validation,_and_test_sets).

#### [Classification](https://en.wikipedia.org/wiki/Statistical_classification)
In this kind of tasks, there is a set of objects (situations) divided into classes. The training sample is a finite set of objects belonging to known classes. An example of such a task is medical diagnostics: on the grounds of formally described medical analyses, we can assume the presence of a certain disease.

You can find examples of solving such tasks in the following lessons: **Neighbors and Wine**, **Horror Trees**, **Pima indians diabetes and linear classifier**, and **Iris Network**.

#### [Regression](https://en.wikipedia.org/wiki/Regression_analysis)
This task involves building a model of measurable data and studying their properties. The data include pairs of values: a dependable variable and an independent variable. The result is a model explaining the dependence of the response variable on the explanatory variable. An example is a model predicting the dependence of real estate prices on time.

#### [Learning to rank](https://en.wikipedia.org/wiki/Learning_to_rank)
In these tasks, it is necessary to sort the values of responses received for a set of objects. The solution may involve a classification or a regression task. This type of problems may occur in an information search or in text analysis.

#### [Structured learning](https://en.wikipedia.org/wiki/Structured_prediction)
It is an umbrella term for tasks of supervised machine learning tasks which involve predicting complex structural objects rather than distinct or real scalar values. An example is the tree of syntax analysis in natural language processing.

### [Unsupervised learning](https://en.wikipedia.org/wiki/Unsupervised_learning)
In this class of tasks, we know only the description of a set of objects (the learning sample), and based on that, we need to find out the dependencies between the objects.

#### [Data clustering](https://en.wikipedia.org/wiki/Cluster_analysis)
The task involves grouping objects into clusters using the information of pairwise similarity between the objects.
An example of such a task is considered in the **Comic-Con and k-means** lesson.

#### [Association rule learning](https://en.wikipedia.org/wiki/Association_rule_learning)
The given data represent feature descriptions. It is necessary to find such sets of features and such feature values that most frequently (non-randomly) occur in the object feature descriptions.

### [Semi-supervised learning](https://en.wikipedia.org/wiki/Semi-supervised_learning)
This type is in between supervised and unsupervised learning tasks. It involves both labeled and unlabeled data. It happens if the learning sample contains answers only to some precedents. An example is automatic rubrication of a large number of texts in the case when some of those have already been assigned to certain rubrics.

An example of solving such tasks is considered in the **Bayes guards SMS** lesson.

### [Active learning](https://en.wikipedia.org/wiki/Active_learning_(machine_learning))
A distinctive feature of this type is that the learning algorithm can itself choose the next training subset with a desired answer: for example, an object with a predefined class. This approach may be used in the cases with huge volumes of data where defining classes for each object manually is time-consuming. One of examples of using active learning is image analysis in astronomy – mapping objects on planet surfaces, classifying stars or galaxies, etc. – when the identification of objects in abundant images requires costly professional work.
