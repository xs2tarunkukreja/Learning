# Representing Data and Feature Engineering
Continuous Feature
Categorical Feature (Discrete Feature) - The property describe the product but don't vary continuous way.
    No natural ordering. 

Feature Engineering - How to represent your data best for a particular application?
    Representation can impact the performance of ML Model.

# Categorical Variable
Dataset - Age; WorkClass; Eduction; Gender; Hours/week; Occupation; Income (>=50k)
Age and Hours/Week is continuous features.
Workclass, education, sex and occupation are categorical variable.
## One Hot Encoding (Dummy Variable)
Replace with one or more feature with value 0 and 1.
Here one feature per category.

Workclass - Govt, Private, Self Employed and SE Incorporated. So, create 4 new featurs. One feature have 1 and other have 0.

### Pandas
Check for proper value... male or man ..
data.gender.value_counts() // It give value and their respective count. So, we check if there is any mistake in values...

data = pd.get_dummies(data) // It add one hot encoding columns.

One limitation - Train datasets and Test datasets should have same set of values... If any value is missing like govt. then model will not work as expected.

So, first call get_dummies of complete data set and then seggarigate test/train data.

## Numbers can encode categorical
Sometimes we can get categorical data in number like 1- Govt, 2 for Private and so on. 
Then it is difficult to say whether data is categorical or continous.

Pandas can't handle this.. scikit-learn can take care as we tell which is categorical and which is continuous.
Or you have to first typecast.
    demo_df['Integer Feature'] = demo_df['Integer Feature'].astype(str)

# Binning, Discretization, Linear Models and Trees
Best way to represent data depends not only on the semantics of the data, but also on the kind of model you are using.

One way to make linear models more powerful on continuous data is to use binning (also known as discretization) of the feature to split it up into multiple features.

Our input have range A to B. Now we take N bins. We divide A to B to N equal part.. We assign bin number to each data point. So, now continuous data converted in categorical data.

bins = np.linspace(-3, 3, 11) // A = -3, B = 3, 10 is number of bins. We get 11 numbers with equal distance.
which_bin = np.digitize(X, bins=bins) // Tell in which category it fall.

Now do One-Hot-Encoding
from sklearn.preprocessing import OneHotEncoder
encoder = OneHotEncoder(sparse=False)
encoder.fit(which_bin)
X_binned = encoder.transform(which_bin)

Binning features generally has no beneficial effect for tree-based models, as these models can learn to split up the data anywhere.

Some features have nonlinear relations with the output—binning can be a great way to increase modeling power.

# Interaction and Polynomials
Another way to enrich a feature representation, particularly for linear models, is adding interaction features and polynomial features of the original data. 

One way to add a slope to the linear model on the binned data is to add the original feature (the x-axis in the plot) back in.
X_combined = np.hstack([X, X_binned])
reg = LinearRegression().fit(X_combined, y)

Slope is share among all bins. We should have separate slope for separate bin. We can achieve this by adding an interaction or product feature that indicates which bin a data point is in and where it lies on the x-axis.
X_product = np.hstack([X_binned, X * X_binned]) 
reg = LinearRegression().fit(X_product, y)

## Polynomial
x ** 2, x ** 3
from sklearn.preprocessing import PolynomialFeatures
poly = PolynomialFeatures(degree=10, include_bias=False) // Include polynomical upto x ** 10:
poly.fit(X)
X_poly = poly.transform(X) 

reg = LinearRegression().fit(X_poly, y) // It perform better 

poly = PolynomialFeatures(degree=2).fit(X_train_scaled)
X_train.shape: (379, 13)
X_train_poly.shape: (379, 105)
These new features represent all possible interactions between two different original features, as well as the square of each original feature. degree=2 here means that we look at all features that are the product of up to two original features.

# Univariate Nonlinear Transformation
There are other transformations that often prove useful for transforming certain features: in particular, applying mathematical functions like log, exp, or sin. 

linear models and neural networks are very tied to the scale and distribution of each feature, and if there is a nonlinear relation between the feature and the target, that becomes hard to model —particularly in regression. The functions log and exp can help by adjusting the relative scales in the data so that they can be captured better by a linear model or neural network.

sin and cos functions can come in handy when dealing with data that encodes periodic patterns.

Most models work best when each feature (and in regression also the target) is loosely Gaussian distributed—that is, a histogram of each feature should have something resembling the familiar “bell curve” shape.

Such thing work better in count data like how many times user login in?

# Automatic Feature Selection
Adding more features makes all models more complex, and so increases the chance of overfitting.
How to choose which feature is important and which is not?
Three Basic Strategy - Univeriate statistic; model-based selection; iterative selection.
## Univeriate statistic
We compute whether there is a statistically significant relationship between feature and target. So, features with High Confidence included.

In case of classification, Analysis of Variance (ANOVA).

They consider each feature individually. Consequently, a feature will be discarded if it is only informative when combined with another feature.

Fast and Independent of model.
f_classif, f_regression - Test: p-value > threshold - discard the feature.

from sklearn.datasets import load_breast_cancer
from sklearn.feature_selection import SelectPercentile
from sklearn.model_selection import train_test_split

// get deterministic random numbers
rng = np.random.RandomState(42)
noise = rng.normal(size=(len(cancer.data), 50))
// add noise features to the data
// the first 30 features are from the dataset, the next 50 are noise
X_w_noise = np.hstack([cancer.data, noise])

// use f_classif (the default) and SelectPercentile to select 50% of features
select = SelectPercentile(percentile=50)
select.fit(X_train, y_train)
X_train_selected = select.transform(X_train)


## Model-based selection
Model-based feature selection uses a supervised machine learning model to judge the importance of each feature, and keeps only the most important ones.

For label, Decision Tree provides feature_importances_, linear models have coefficient

L1 penalty learn sparse coefficient, which only use a subset of features.

from sklearn.feature_selection import SelectFromModel
from sklearn.ensemble import RandomForestClassifier
select = SelectFromModel(RandomForestClassifier(n_estimators=100, random_state=42), threshold="median") 
// It select all feature more than threshold. Here we are using median as threshold.

select.fit(X_train, y_train)
X_train_l1 = select.transform(X_train)

## Iterative selection
In univariate testing we used no model, while in model-based selection we used a single model to select features.

In iterative feature selection, a series of models are built, with varying numbers of features. There are two basic methods: starting with no features and adding features one by one until some stopping criterion is reached, or
starting with all features and removing features one by one until some stopping criterion is reached.

Because a series of models are built, these methods are much more computationally expensive.

One particular method of this kind is recursive feature elimination (RFE), which starts with all features, builds a model, and discards the least important feature according to the model. Then a new model is built using all but the discarded feature, and so on until only a prespecified number of features are left.

from sklearn.feature_selection import RFE
select = RFE(RandomForestClassifier(n_estimators=100, random_state=42),
n_features_to_select=40)
select.fit(X_train, y_train)

# Utilizing Expert Knowledge
Often, domain experts can help in identifying useful features that are much more informative than the initial representation of the data.
