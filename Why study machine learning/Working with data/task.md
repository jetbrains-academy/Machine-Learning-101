### Defining the problem
Before you start using machine learning methods, it is crucial to decide what type of problems your analytical task represents and thus choose the right solution algorithm. It depends on the research goals and available data.

### Working with data
When approaching a task, it's important to realize what data on the research object are available: for example, numbers and text, patients' medical records, coordinates and itineraries, etc. In certain tasks, you may encounter data which is more difficult to process – audio tracks, images, functions, and numerical sequences.
Having outlined the available data, you need to choose the object features the model will be based on.
There are several basic types of [features](http://www.machinelearning.ru/wiki/index.php?title=%D0%9F%D1%80%D0%B8%D0%B7%D0%BD%D0%B0%D0%BA%D0%BE%D0%B2%D0%BE%D0%B5_%D0%BE%D0%BF%D0%B8%D1%81%D0%B0%D0%BD%D0%B8%D0%B5) described in the data and used in task solving. The examples below are taken from the field of medical diagnostics:

- **Binary**. Such features can take one of two possible values. For example, the presence of pain in the back or the presence of a certain genetic variant.
- **Nominal**. For this feature type, the range of possible values lies within a certain finite set. For example, the types of pain: lancinating, acute, or dull.
- **Ordinal**. These features may be ordered on a certain scale and compared with each other. For example, the assessment of a patient's condition: satisfactory, moderately severe, critical.
- **Quantitative**. These features are expressed by real numbers. They include such characteristics as temperature, pulse, and blood pressure.

You may also need to pre-process the data so that it can be conveniently described by this or that feature. In this course, the data was already pre-processed; you can find a review of existing techniques in [this video](https://www.coursera.org/lecture/vvedenie-mashinnoe-obuchenie/priedobrabotka-dannykh-okkCL).

Depending on the result, you may need to select the features or combine several of them into one ([dimensionality reduction](https://ru.wikipedia.org/wiki/%D0%A1%D0%BD%D0%B8%D0%B6%D0%B5%D0%BD%D0%B8%D0%B5_%D1%80%D0%B0%D0%B7%D0%BC%D0%B5%D1%80%D0%BD%D0%BE%D1%81%D1%82%D0%B8)) for a better formalization of the problem.

### Solution quality assessment
Defining a problem, you also need to choose a metric for assessing the quality of the solution. For example, in **regression** problems, it may be standard deviation ([MSE](https://ru.wikipedia.org/wiki/%D0%A1%D1%80%D0%B5%D0%B4%D0%BD%D0%B5%D0%BA%D0%B2%D0%B0%D0%B4%D1%80%D0%B0%D1%82%D0%B8%D1%87%D0%B5%D1%81%D0%BA%D0%BE%D0%B5_%D0%BE%D1%82%D0%BA%D0%BB%D0%BE%D0%BD%D0%B5%D0%BD%D0%B8%D0%B5)), in classification problems – the proportion of right answers, and in clustering the final metric always depends on the specific task.

Besides, at this stage you need to decide what data will be used for testing. For example, in the case of supervised learning, you can divide your data set into [training and testing](http://www.machinelearning.ru/wiki/index.php?title=%D0%9E%D0%B1%D1%83%D1%87%D0%B0%D1%8E%D1%89%D0%B0%D1%8F_%D0%B2%D1%8B%D0%B1%D0%BE%D1%80%D0%BA%D0%B0) parts. The **training set** is the data that will be used to fit the parameters and optimize the model. The **testing set** is the data set on which you will evaluate the trained model.


### Possible bias
The data in your sets may be biased. You need to take it into account while building the model and  while interpreting the result. You can find the possible types of bias [here](https://developers.google.com/machine-learning/glossary#bias_ethics).


### Training limitations
When building a model, you need to assess possible training limitations:
- insufficient number of features in the set;
- insufficient training set;
- feature noise.

### Defining the task type 
After formalizing the data as a model and setting the goal, you need to decide upon the type of the task.
