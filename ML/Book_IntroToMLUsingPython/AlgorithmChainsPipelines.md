# Algorithm Chains and Pipelines
Pipeline - To create a pipeline to connect severals preprocessing and models.

# Parameter Selection with Preprocessing
Better parameters for SVC by using GridSearchCV

from sklearn.model_selection import GridSearchCV
param_grid = {'C': [0.001, 0.01, 0.1, 1, 10, 100], 'gamma': [0.001, 0.01, 0.1, 1, 10, 100]}
grid = GridSearchCV(SVC(), param_grid=param_grid, cv=5)
grid.fit(X_train_scaled, y_train)

the splitting of the dataset during cross-validation should be done before doing any preprocessing.
Any process that extracts knowledge from the dataset should only ever be applied to the training portion of the dataset, so any cross-validation should be the “outermost loop” in your processing.

Pipeline class to proper ordering. It have fit, predict, score method.

# Building Pipelines
from sklearn.pipeline import Pipeline
pipe = Pipeline([("scaler", MinMaxScaler()), ("svm", SVC())])
pipe.fit(X_train, y_train)
pipe.score(X_test, y_test)

# Using Pipelines in Grid Searches
param_grid = {'svm__C': [0.001, 0.01, 0.1, 1, 10, 100], 'svm__gamma': [0.001, 0.01, 0.1, 1, 10, 100]}
grid = GridSearchCV(pipe, param_grid=param_grid, cv=5)
grid.fit(X_train, y_train)

// MinMaxScaler is refit with only the training splits and no information is leaked from the test split into the parameter search.

# General Pipeline Interface
Only requirement for estimators in a pipeline is that all but the last step need to have a transform method, so they can produce a new representation of the data that can be used in the next step.

def predict(self, X):
    X_transformed = X
    for step in self.steps[:-1]:
        X_transformed = step[1].transform(X_transformed)
    return self.steps[-1][1].predict(X_transformed)

## Convenient Pipeline Creating with make_pipeline
from sklearn.pipeline import make_pipeline
pipe_short = make_pipeline(MinMaxScaler(), SVC(C=100))

## Accessing Step Attributes
// extract the first two principal components from the "pca" step
components = pipe.named_steps["pca"].components_

## Accessing attributes in a Grid Searched Pipeline
from sklearn.linear_model import LogisticRegression
pipe = make_pipeline(StandardScaler(), LogisticRegression())
param_grid = {'logisticregression__C': [0.01, 0.1, 1, 10, 100]}
X_train, X_test, y_train, y_test = train_test_split(cancer.data, cancer.target, random_state=4)
grid = GridSearchCV(pipe, param_grid, cv=5)
grid.fit(X_train, y_train)

best_model = grid.best_estimator_.named_steps["logisticregression"]

# Grid-Searching Preprocessing Step and Model Parameters
benefit, we can now adjust the parameters of the preprocessing using the outcome of a supervised task like regression or classification. 

from sklearn.datasets import load_boston
boston = load_boston()
X_train, X_test, y_train, y_test = train_test_split(boston.data, boston.target, random_state=0)

from sklearn.preprocessing import PolynomialFeatures
pipe = make_pipeline(StandardScaler(), PolynomialFeatures(), Ridge())

// How do we know which degree of polynomial to use.

param_grid = {'polynomialfeatures__degree': [1, 2, 3], 'ridge__alpha': [0.001, 0.01, 0.1, 1, 10, 100]}
grid = GridSearchCV(pipe, param_grid=param_grid, cv=5, n_jobs=-1)
grid.fit(X_train, y_train)

# Grid Searching Which Model to Use
RandomForest Classifier - No Preprocessing Or an SVC - StandardScalar

from sklearn.ensemble import RandomForestClassifier
param_grid = [
{'classifier': [SVC()], 'preprocessing': [StandardScaler(), None], 'classifier__gamma': [0.001,0.01,0.1,1,10,100],
            'classifier__C': [0.001, 0.01, 0.1, 1, 10, 100]},
{'classifier':[RandomForestClassifier(n_estimators=100)],'preprocessing':[None],'classifier__max_features':[1,2, 3]}]

grid = GridSearchCV(pipe, param_grid, cv=5)
grid.fit(X_train, y_train)

Best params: {'classifier':
SVC(C=10, cache_size=200, class_weight=None, coef0=0.0,
decision_function_shape=None, degree=3, gamma=0.01, kernel='rbf',
max_iter=-1, probability=False, random_state=None, shrinking=True,
tol=0.001, verbose=False),
'preprocessing':
StandardScaler(copy=True, with_mean=True, with_std=True),
'classifier__C': 10, 'classifier__gamma': 0.01}