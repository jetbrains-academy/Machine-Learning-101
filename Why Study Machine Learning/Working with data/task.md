## Working with data

### Defining the problem
Before you start using machine learning methods, it is crucial to decide what type of problems your analytical task represents and thus choose the right solution algorithm. It depends on the research goals and available data.

### Working with data
When approaching a task, it's important to realize what data on the research object are available: for example, numbers and text, patients' medical records, coordinates and itineraries, etc. In certain tasks, you may encounter data which is more difficult to process – audio tracks, images, functions, and numerical sequences.
Having outlined the available data, you need to choose the object features the model will be based on.
There are several basic types of [features](https://en.wikipedia.org/wiki/Feature_(machine_learning)) described in the data and used in task solving. The examples below are taken from the field of medical diagnostics:

- **Binary**. Such features can take one of two possible values. For example, the presence of pain in the back or the presence of a certain genetic variant.
- **Nominal**. For this feature type, the range of possible values lies within a certain finite set. For example, the types of pain: lancinating, acute, or dull.
- **Ordinal**. These features may be ordered on a certain scale and compared with each other. For example, the assessment of a patient's condition: satisfactory, moderately severe, critical.
- **Quantitative**. These features are expressed by real numbers. They include such characteristics as temperature, pulse, and blood pressure.

You may also need to pre-process the data so that it can be conveniently described by this or that feature. In this course, the data was already pre-processed; you can find a review of some basic techniques in [this article](https://towardsdatascience.com/introduction-to-data-preprocessing-in-machine-learning-a9fa83a5dc9d).

Depending on the result, you may need to select the features or combine several of them into one ([dimensionality reduction](https://en.wikipedia.org/wiki/Dimensionality_reduction)) for a better formalization of the problem.

### Solution quality assessment
Defining a problem, you also need to choose a metric for assessing the quality of the solution. For example, in **regression** problems, it may be mean squared error ([MSE](https://en.wikipedia.org/wiki/Mean_squared_error)), in classification problems – the proportion of right answers, and in clustering the final metric always depends on the specific task.

Besides, at this stage you need to decide what data will be used for testing. For example, in the case of supervised learning, you can divide your data set into [training and testing](https://en.wikipedia.org/wiki/Training,_validation,_and_test_sets) parts. The **training set** is the data that will be used to fit the parameters and optimize the model. The **testing set** is the data set on which you will evaluate the trained model.


### Possible bias
The data in your sets may be biased. You need to take it into account while building the model and  while interpreting the result. You can find the possible types of bias [here](https://developers.google.com/machine-learning/glossary#bias_ethics).


### Training limitations
When building a model, you need to assess possible training limitations:
- insufficient number of features in the set;
- insufficient training set;
- feature noise.

### Defining the task type 
After formalizing the data as a model and setting the goal, you need to decide upon the type of the task.
