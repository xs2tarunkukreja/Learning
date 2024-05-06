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

Note - Dataset is large - solver=sag in logistic regression and ridge. SGDClassifier & SGDRegressor Classes are option.

## Naive Bayes Classifier

## Decision Tree

## Ensemble of Decision Tree

## Kernelized Support Vector Machine

## Neural Network - Deep Learning



# Uncertainty Estimate from Classifiers