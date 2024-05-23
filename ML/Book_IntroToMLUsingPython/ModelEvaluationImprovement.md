# Model Evaluation and Improvement
Evaluating Models and Selecting Parameters.

Common way: Evaluate Score on Test Data
model.score(X_test, y_test)

# Cross Validation
Cross-validation is a statistical method of evaluating generalization performance that is more stable and thorough than using a split into a training and a test set.

In crossvalidation, the data is instead split repeatedly and multiple models are trained.

## k-fold cross validation
One way of Cross Validation.

k = 5 means data is partitioned into 5 parts (mostly equal) called fold. 
First Model - 1 as Test; (2,3,4,5) as Train Data.
2nd Model - 2 as Test; (1,3,4,5) as Train Data.
Upto 5th model.

In end we collect 5 accuracy values.

## Cross Validation in scikit-learn
from sklearn.model_selection import cross_val_score
from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression

iris = load_iris()
logreg = LogisticRegression()
scores = cross_val_score(logreg, iris.data, iris.target) // By default, 3 fold. cv parameter is used to change.

Using the mean cross-validation we can conclude that we expect the model to be around 96% accurate on average.
Range 90% to 100%

## Benefits
train_test_split is luck based. Lucky if all easy in test data. Unlucky if hard to classify in test data.
But cross validation actually evaluate the model.

We get best and worst performance as well.

More data is use to train.. so the model get trained better.

Disadvantage - Computational Cost

## Stratified K-Fold Cross Validation and Other Strategies
Splitting the dataset into k folds by starting with the first one-k-th part of the data might not always be a good idea. Example Iris Data - 0-1/3 is class 0; 1/3-2/3 is class 1; 2/3-1 is class 2.

Use stratified k-fold cross-validation - It split the data such that the proportions between classes are the same in each fold as they are in the whole dataset

### More Control Over Cross Validation
cv parameter to control the number of folds.

from sklearn.model_selection import KFold
kfold = KFold(n_splits=5) // It gives 0, 0, 0 score.
cross_val_score(logreg, iris.data, iris.target, cv=kfold)

kfold = KFold(n_splits=3, shuffle=True, random_state=0)

## Leave-one-out cross-validation
Another frequently used cross-validation method is leave-one-out.
For each split, you pick a single data point as Test Set.

It is very time consuming.

from sklearn.model_selection import LeaveOneOut
loo = LeaveOneOut()
scores = cross_val_score(logreg, iris.data, iris.target, cv=loo)

## Shuffle-split cross-validation
each split samples train_size many points for the training set and test_size many (disjoint) point for the test set. This splitting is repeated n_iter times.

## Cross Validation with Groups
Example - Emotion classifier. Take persons and n picture per person.
Training and Test data should have different persons.

GroupKFold
from sklearn.model_selection import GroupKFold
X, y = make_blobs(n_samples=12, random_state=0)
groups = [0, 0, 0, 1, 1, 1, 1, 2, 2, 3, 3, 3]
scores = cross_val_score(logreg, X, y, groups, cv=GroupKFold(n_splits=3))
Cross-validation scores: [ 0.75 0.8 0.667]

# Grid Search
Tunning the parameters.
    It is tricky task.
Try N values of parameters and form a grid.

## Simple Grid Search
For loops

for gamma in [0.001, 0.01, 0.1, 1, 10, 100]:
    for C in [0.001, 0.01, 0.1, 1, 10, 100]:
        svm = SVC(gamma=gamma, C=C)
        svm.fit(X_train, y_train)
        score = svm.score(X_test, y_test)
        if score > best_score:
            best_score = score
            best_parameters = {'C': C, 'gamma': gamma}

## The Danger of Overfitting the Parameters and the Validation Set
But this accuracy won’t necessarily carry over to new data.
Spit Data = Training; Validation (For paramters evaluation); Test

It avoid test data leakage.. for better evaluation in end.

## Grid Search with Corss Validation
for gamma in [0.001, 0.01, 0.1, 1, 10, 100]:
    for C in [0.001, 0.01, 0.1, 1, 10, 100]:
        svm = SVC(gamma=gamma, C=C)
        scores = cross_val_score(svm, X_trainval, y_trainval, cv=5)
        score = np.mean(scores)
        if score > best_score:
            best_score = score
            best_parameters = {'C': C, 'gamma': gamma}

svm = SVC(**best_parameters)
svm.fit(X_trainval, y_trainval)

GridSearchCV class provides the feature.

param_grid = {'C': [0.001, 0.01, 0.1, 1, 10, 100],
              'gamma': [0.001, 0.01, 0.1, 1, 10, 100]}
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC
grid_search = GridSearchCV(SVC(), param_grid, cv=5)

grid_search.fit(X_train, y_train)
grid_search.score(X_test, y_test)

grid_search.best_params_
grid_search.best_score_
grid_search.best_estimator_ // Actual Best Model

// When parameters mapping is not 1 to 1.. linear only C important.
param_grid = [{'kernel': ['rbf'], 'C': [0.001, 0.01, 0.1, 1, 10, 100], 'gamma': [0.001, 0.01, 0.1, 1, 10, 100]},
              {'kernel': ['linear'], 'C': [0.001, 0.01, 0.1, 1, 10, 100]}]

Using different cross-validation strategies with grid search - cv parameter in GridSearchCV
n_jobs parameter to the number of CPU cores you want to use.

For Spark users, there is also the recently developed spark-sklearn package, which allows running a grid search over an already established Spark cluster.

# Evaluating Metrics and Scoring
We evaluated classification performance using accuracy and regression performance by using R**2.
Sometimes need to choose proper metrics.

## Keep the End Goal in Mind
Business Metrics.
Goals - Avoid Traffic Incidents; Decrease Hospital Admission; Getting more user on Websites.
Pick model or parameter have most +ve influence on business metrics.

We can't play with parameters or model in production.

## Metrics for Binary Classification - 276 Page

## Metrics for Multiclass Classification

## Regression Metrics

## Using Evaluation Metrics in Model Selection
