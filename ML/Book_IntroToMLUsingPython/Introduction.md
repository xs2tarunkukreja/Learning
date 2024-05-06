# Why ML?
In past, mostly app is based on if and else.
    It doesn't learn.
    Complicate to change.
    Rules should be known in advance.
    Also, sometimes rules are difficult to define.

Supervised Learning - I/P and O/P > Model/Algorithm. Model learn.
    Identify zip code from handwritten digits on an envelope.
    Determine whether a tumor is benign based on medical image
    Detecting fraudulent activity in credit card and transaction.

    Main Job is to collect data.

Unsupervised Learning - Only Input data is known. They are actual harder to understand and evaluate.
    Identify topics in a set of blog post
    Segmenting customers into groups with similar preference.
    Detecting abnormal access pattern to a website

You need data that a computer can understand.
    Image - grayscale value of each pixel, size, shape or color.

Sample - Each entity or row = Data Point.
Features - Columns or Properties.
Output = Label

Feature extraction or feature engineering - Building a good representation of your data.

## Knowing Your Task and Data
It is important to understand data and how it relate to task.
It is not effective to randomly choose an algorithm and throw your data to it.
You should answers following questions -
    Task? Can data answer that question/task?
    Enough Data?
    Feature list
    How measure success?
    How integrate models with application?

# Why Python?
ML or Analysis is interactive process

# scikit-learn & Essential Libraries
Open Source. 
Depend on 2 python packages - numpy or scipy
plotting and interactive development - matplotlib, ipython, jupytor notebook.

pip install numpy scipy matplotlib ipython scikit-learn pandas

scikit-learn takes in data in form of NumPy array.

SciPy - A collection of functions for Scientific computing.
    Sparse matrices are used whenever we want to store a 2D array that contains mostly zeros

http://www.scipy-lectures.org/

# My First Application: Classify Iris Species
## Meet Data
from sklearn.datasets import load_iris
iris_dataset = load_iris()

X = iris_daataset['data'] = 2D that's why cap.
y = iris_dataset['target'] = 1D  or vector.

## Measuring Success
Traning and Testing Data
Thumb Rule = 25% test data.

## Look at Your Data
Is Data suitable to find answer?
Find abnormalities and peculiarities.

Scatter Plot - one feature as X-axis and another as Y-axis. draw dot for each data point.

Pair Plot - Data Point is colored as species. Matrix of Scatter Plot where pair for 2 featurs.
    [n,n] = histogram chart. 
    [x,y] = scatter plot

iris_df = pd.DataFrame(X_train, columns=iris_dataset.features_names)

## Build Your First Model: k-Nearest Neighbour
Here we will find k closest neighbours to a new point. then we predict based on majority of neighbour.
model.fit(X_train, y_train)

## Make Prediction
model.predict(X_new)

## Evaluate Model
Predict for Test Data and Compare with Actual result.
np.mean(y_pred == y_test)

model.score(x_test, y_test)

Both are same.