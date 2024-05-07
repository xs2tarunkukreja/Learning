# Supervised Learning
When we have input/output pair.
Main effort is to build dataset.

# Classification and Regression
Two types of Supervised Learning

## Classification
Goal: predict a class label from a predefined list of possibilities.
Binary Classification (Yes or No) and Multiclassification

## Regression
Goal: Predict a continuous number or a floating-point number.
Predict value is an amount or a number.

# Generalization, Overfitting and Underfitting
If a model able to make accurate prediction on unseen data, we say it is able to generalize from training set to test set.

Overfitting occurs when you fit a model too closely to the particularities of the training set. but the model is not able to generalize to new data set. Too Complex.

Underfitting - So simple - You are not able to capture all aspects. Model fail on training data.

## Relationship of model complexity to Data Size
The larger variety of data points your dataset contains, the more complex a model you can use without overfitting. Usually, collecting more data points will yield more variety, so larger datasets allow building more complex models. However, simply duplicating the same data points or collecting
very similar data will not help.

# Supervised ML Algorithms
Strength; Weakness; What Kind of data they can be best applied to.
Parameters and Options.

## Some Sample Dataset
X, y = mglearn.datasets.make_forge() - For classifier
X, y = mglearn.datasets.make_wave(n_samples=40) - For regression
from sklearn.datasets import load_breast_cancer - benign or malignant
from sklearn.datasets import load_boston - Boston Housing Dataset to predict Home Value.

## k-Nearest Neighbour
Model consist of storing the training dataset. It find the nearest k neighbour from training data for new data point.

### k-Neigbour classification
k = 1  means nearest neighbour define the output.
k means vote system. maximum vote win.

from sklearn.neighbors import KNeighborsClassifier
model = KNeighborsClassifier(n_neighbors=3)

Draw a graph to find the k values -
Y = Accuracy
X = k
plot 2 lines.. One for Test data and One for training data.

Choose where both lines point are near in term of accuracy.

### k-neighbour regression
from sklearn.neighbors import KNeighborsRegressor
model = KNeighborsRegressor(n_neighbors=3)

R2 score, also known as the coefficient of determination, is a measure of goodness of a prediction for a regression model, and yields a score between 0 and 1. A value of 1 corresponds to a perfect prediction.

k = 1, 3, 9. 
calculate score for train as well as test data. based on both we should select.

### Strength, Weakness and Parameters
Parameters - 2 - k and how you measure distance between data points.
    By default, euclidean distance is used.

Strength -
    Easy to understand and ofter gives reasonable performance without a lot of adjustment.
    Fast to train

Weakness -
    If the training dataset size is huge then prediction takes time.
    Not perform well if feature list is huge. (>=100)
    Not perform well with sparse dataset. Lots of 0 most of time.

It is mostly not used.

## Linear Model
Widely used in practice. It predict using a linear function of the input features.

### Regression
ŷ = w[0] * x[0] + w[1] * x[1] + ... + w[p] * x[p] + b
w and b parameters are learned.
ŷ = prediction
x are features.

For one feature
ŷ = w * x + b
w is a slope and b is y-axis offset.

Alternate, you can think of predict response as being a weighted sum of features.

Prediction = line, plane, hyperplane.

Using a straight line to make predictions seems very restrictive. It looks like all the fine details of the data are lost. In a sense, this is true.

But looking at one-dimensional data gives a somewhat skewed perspective. For datasets with many features, linear models can be very powerful.

Linear Regression models are different based on how w and b value is learn.

#### Linear Regression (ordinary least square)
w and b is find by minimizing the mean squared error. sum of square difference of ŷ and y.

Parameters - No - So, no control on complexity.

from sklearn.linear_model import LinearRegression
w = slope = weight = coefficient = model.coef_ = Numpy array.
b = intercept = model.intercept_

_ in end means derived values, not a parameter in scikit-learn.

model.score = R^2

higher chances of overfitting

#### Ridge Regression
all w should be near to 0. So, all feature have little effect on outcome. this constraint is called regularization. It avoid overfitting.

from sklearn.linear_model import Ridge

Parameter alpha - It value depends on dataset.
    Increasing its value force coefficient to more toward zero, which decrease performance in training set but might help generalization (which is good...)

low alpha means Ridge = LinearRegression

#### Lasso
An alternative to Ridge for regularizing linear regression is Lasso.
It also restrict w to near to 0 but in different way.
    L1 Regularization - Some coeff become exact 0. Automatic feature selection.

from sklearn.linear_model import Lasso
model = Lasso().fit(X_train, y_train)

It perform bad for training as well as test data. Underfitting.
lasso001 = Lasso(alpha=0.01, max_iter=100000).fit(X_train, y_train)
// Decrease alpha and increase max_iter.

Now better than Ridge.
np.sum(lasso001.coef_ != 0) - Number of features actual used.

Too low alpha means overfitting. So, become Linear Regression.

Note - In practice, Ridge is first choice. If lots of features and we expect few are important than Lasso.

### Linear Model for Classification
ŷ = w[0] * x[0] + w[1] * x[1] + ... + w[p] * x[p] + b > 0
We threshold predict value to 0.

If function < 0 then ŷ = -1 Else ŷ = 1

Linear classifier separate 2 classes by line, plane, hyper plane.
Different algorithm choose different way to measure what "fitting the training set well".

Two most famouse one - inear_model.LogisticRegression and linear support vector machines
(linear SVMs), implemented in svm.LinearSVC.
Both apply L2 Regularization as Ridge.

Trade-off parameter that determine the strength of regularization is called C.
    Higher value of C corresponding to less regularization.
    High C, model try to fit training set as best as possible.
    Low C, w close to 0.

lr_l1 = LogisticRegression(C=C, penalty="l1").fit(X_train, y_train) // Here we are using L1 Regularization. Now feature set will be reduced. By default L2.

### Linear Model for multiclass classification
Most are binary classification. Use one-v/s-rest approach for multiclass classification.
classifier/model with highest score win.

#### Strength, Weakness and Parameters
Main parameters - regularization parameters - alpha in regession and C in classification.
other decision - L1 Regularization or L2 Regularization.

If you think only few feature are important than choose L1.

Strength
    Fast to train and fast to predict.
    Scale very large dataset and work well with sparse data.
    Easy to understand.

Weekness
    If your dataset is highly correlated features then coefficient hard to interpret.

Note - 
    Dataset is large - solver=sag in logistic regression and ridge. SGDClassifier & SGDRegressor Classes are option.
    In lower dimension space, other model might yield better generalization performance. 
    It is good when number of feature set are large.

## Naive Bayes Classifier
It is fast to train but generalization performance is slightly worse than linear classifiers.

Reason for fast - They learn parameters by looking at each feature individually and collect simple per-class statistic from each feature.

Three types in scikit-learn - GaussianNB; BernoulliNB; MultinomialNB

GaussianNB can be applied on any continuous data.
BernoulliNB assume binary data. Used in Text Data Classification.
MultinomialNB assume count data. Used in Text Data Classification.

BernoulliNB - Count how often every feature of each class is non-zero.
MultinomialNB takes into account the average value of each feature for each class.
GaussianNB takes into account the average value as well as standard deviation of each feature for each class.

Formula for MultinomialNB and GaussaianNB is same as Linear but have differnet meaning.

### Strength, Weakness and Parameters
MultinomialNB and BernoulliNB have single parameter i.e. alpha. 
    Large alpha, more smoothing, less complex model.
    It not impact performance but accuracy.
    
    This both work on sparse data.

MulinomialNB perform better than BinaryNB.

Strength -
    They are very fast to train and to predict, 
    The training procedure is easy to understand. 
    The models work very well with high-dimensional sparse data and are relatively robust to the parameters. 
    Naive Bayes models are great baseline models and are often used on very large datasets, where training even a linear model might take too long.

## Decision Tree
Widely used model for classification and regression tasks.
    Learn if/else.

Here data is like you ask if/else question like Is animal have feather?
                            - Yes - Hawk
            Yes - Can fly? 
                            - No - Penguine
Has Feather 
                            - No - Bear
            No - Has Fine ?
                            - Yes - Dolphine

Leaf conatin answers.

### Building Decision Tree
2 half-moon shapes - each class have 75 data points.
In ML settings these questions are called tests.
For continuous data, question are like "Is feature i larget than value a?"
Algorithm search over all possible tests and find the one most informative about the target value.

Recursive process - It is repeated until each region in the partition only contains a single target value(A single class or a single regression value). Leaf contain one target is called Pure.

### Controlling Complexity of Decision Tree
All leaves are pure then it may be overfitting.
Avoid Strategy - 2 -
    Pre-pruning - Stopping tree creation early.
    Post-pruning or pruning - Build complete tree then remove nodes.
Critria to stop -
    Maximum depth of tree
    Maximum number of leaves.
    Minimum number of point in node for split.

scikit-learn only implements pre-pruning, not post-pruning.

from sklearn.tree import DecisionTreeClassifier
tree = DecisionTreeClassifier(random_state=0)

tree = DecisionTreeClassifier(max_depth=4, random_state=0)

Visualize Tree -
    from sklearn.tree import export_graphviz
    export_graphviz(tree, out_file="tree.dot", class_names=["malignant", "benign"],
                    feature_names=cancer.feature_names, impurity=False, filled=True)

    import graphviz
    with open("tree.dot") as f:
    dot_graph = f.read()
    graphviz.Source(dot_graph)

Feature importance in trees -
    Matrix that can help to understand the working of Tree  - Feature Importance - Which rate how important each feature is for decision a tree makes.
    tree.
        feature_importances_
    Create a bar chart for all feature with its value.
    Always positive.

DecisionTreeRegressor
Note - The DecisionTreeRegressor (and all other tree-based regression models) is not able to extrapolate, or make predictions outside of the range of the training data.

### Strength, Weakness and Parameters
Parameters -
    Pre-pruning parameter - max_depth; max_leaf_nodes; min_sample_leaf.

Advantages
    Resulting model can be visualized and understand by non-expert.
    Algorithm are completely invariant to scale of the data.
    No normalization or standardization of feature is needed for decision tree algorithm.
Weakness 
    Even it can be overfitted.

Note: Ensemble Decision Tree is used in practice.

## Ensemble of Decision Tree
Ensemble are method that combine multiple ML Algorithms.

### Random Forest
Solve overfitting problem.
Collection of Decision Tree.
Concept - One Decision Tree Overfit in one area.. another in another area. Reduce the overfitting by averaging this.

By 2 ways in which trees are randomized
    Select different data points.
    Selecting feature in each split test.

#### Building Random Forest
Select number of tree - n_estimators in RandomForestRegressor or Random ForestClassifier

Now prepare bootstrap sample from sample data sets. In bootstrap - few data points from sample data + few are repeated. So, count remain same. Example - [a,b,c,d] => [b,b,c,a]

Now create tree. Instead of selecting best test, in each node, algorithm select a subset of features and select best test based on those subset features.
    max_features parameter.

high max_feature means trees will be quite similar.

#### Result
Regression - Average of all trees.
Classification - A soft voting.

#### Code
from sklearn.ensemble import RandomForestClassifier
forest = RandomForestClassifier(n_estimators=5, random_state=2)

Often default parameters work fine.

#### Strength, Weakness and Parameters
Strength
    Most widely used algorithm in regression and classifier.
    Perform well without tuning.
Weakness
    Timeconsuming to build on large dataset.
    Don't perform well on very high dimensional, sparse data and text.
    more memory and time as compare to linear.  

Parameters 
    n_jobs - Use number of cores, || processing. -1 means use all cores.
    n_estimator - Large is better.
    max_features - smaller reduce overfitting. Ideal = sqrt(n_features) for classification; log2(n_features) for regression.
    max_depth 

### Gradient Boosted Decision Tree
It build tree in serial manner, where each tree tries to correct the mistake of last tree.
No randomization and use strong pre-pruning.
Very shallow tree of depth 1 to 5. Small memory and fast prediction.
Logic - combine weak learner or shallow tree. each tree make prediction on part of data. So, more tree improve.

Note: Widely used in industry.

Here parameters are sensitive. We need to set them correctly.

learning_rate - how stronly next tree try to correct previous tree.

// By default, 100 trees of maximum depth 3 and a learning rate of 0.1 are used
from sklearn.ensemble import GradientBoostingClassifier
gbrt = GradientBoostingClassifier(random_state=0)

Overfitting reduce - By apply strong pre-pruning by limiting maximum depth or lower learning_rate.

#### Strength, Weakness and Parameters
Strength
    Most powerful and widely used.

Drawback
    Careful tuning of parameters are required.
    Not work with high Dimension and sparse data.

Parameters
    n_estimators - depends on memory and time you have.
    learning_rate

## Kernelized Support Vector Machine
Kernelized Support Vector Machine is an extension of SVM for more complex model.

### Linear models and nonlinear features
suppose data distribution is such that it can't be categorized by a line for feature1 and feature2.
now add one more dimension - f1^2

Now linear SVM model is not linear any more. It is not a line.. it is ellipse.

### Kernel Trick
Adding nonlinear features to the representation of data can make linear model more powerful. But when feature count is 100 or more then it is difficult to choose nonlinear feature.

it works by directly computing the distance (more precisely, the scalar products) of the data points for the expanded feature representation, without ever actually computing the expansion.

There are two ways to map your data into a higher-dimensional space that are commonly used with support vector machines: 
the polynomial kernel, which computes all possible polynomials up to a certain degree of the original features (like feature1 ** 2 * feature2 ** 5); 
the radial basis function (RBF) kernel, also known as the Gaussian kernel. The Gaussian kernel is a bit harder to explain, as it corresponds to an infinite-dimensional feature space. One way to explain the Gaussian kernel is that
it considers all possible polynomials of all degrees, but the importance of the features decreases for higher degrees.

### Understanding SVM
During training, the SVM learns how important each of the training data points is to represent the decision boundary between the two classes. Typically only a subset of the training points matter for defining the decision boundary: the ones that lie on the border between the classes. These are called support vectors and give the support vector machine its name.

To make a prediction for a new point, the distance to each of the support vectors is measured. A classification decision is made based on the distances to the support vector, and the importance of the support vectors that was learned during training (stored in the dual_coef_ attribute of SVC).

The distance between data points is measured by the Gaussian kernel:
krbf(x1, x2) = exp (ɣǁx1 - x2ǁ2)
Here, x1 and x2 are data points, ǁ x1 - x2 ǁ denotes Euclidean distance, and ɣ (gamma)
is a parameter that controls the width of the Gaussian kernel.

### Tunning SVM Parameters
C - regularization parameter
gamma - controls the width of the Gaussian kernel. It determines the scale of what it
means for points to be close together.

### overfitting - How to solve?
Preprocessing data - Rescale features to bring them on one scale.
    Use MinMaxScaler

### Strength, Weakness and parameters
Strength -
    Complex boundaries
    Work well with low as well as high dimension.

Weakness -
    Don't work with large data sample.
    Preprocessing and tuning of parameter need to be done carefully.
    SVM model are hard to inspect.

We should try if all parameters are in same scale.


## Neural Network - Deep Learning
NeuralNetwork.md

# Uncertainty Estimate from Classifiers
scikit-learn, classifier also provide uncertainty estimate of predictions.
    decision_function and predict_proba

gbrt = GradientBoostingClassifier(random_state=0)
gbrt.fit(X_train, y_train_named)

gbrt.decision_function(X_test)
gbrt.predict_proba(X_test)

## Uncertainity in MultiClass Classification
Same functions. 