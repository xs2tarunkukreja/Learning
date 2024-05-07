# Neural Network
Deep Learning
Simple Method - Multilayer Perceptrons for Classifier and Regression
MLP are also known as Vanilla or feed forward NN or sometimes just NN.

## Neural Network Model
MLP is generalization of linear models that perform multiple stages of processing to come to a decision.

Linear Regressor - ŷ = w[0] * x[0] + w[1] * x[1] + ... + w[p] * x[p] + b
In MLP, the weights are calculated several times.
    The model has lots of more coefficient to learn.
    A trick, after computing weighted sum for each hidden unit, a non-linear function is applied to the result - usually the rectifying nonlinearity(relu) or tangen hyperbolicus (tanh)
Example -
 x[0-3] > h[0-2] > ŷ

h[0] = tanh(w[0, 0] * x[0] + w[1, 0] * x[1] + w[2, 0] * x[2] + w[3, 0] * x[3])
h[1] = tanh(w[0, 1] * x[0] + w[1, 1] * x[1] + w[2, 1] * x[2] + w[3, 1] * x[3])
h[2] = tanh(w[0, 2] * x[0] + w[1, 2] * x[1] + w[2, 2] * x[2] + w[3, 2] * x[3])
ŷ = v[0] * h[0] + v[1] * h[1] + v[2] * h[2]

w is wight between input and hidden layer.
v is weight between hidden layer and output layer.
v and w learned from data.

Important Parameter - Nodes in hidden layers.
    There may be n hidden layers.

## Tune NN
from sklearn.neural_network import MLPClassifier

MLPClassifier(algorithm='l-bfgs', random_state=0).fit(X_train, y_train)

By default, MLP has 100 hidden nodes. Small dataset, we can reduce this.
mlp = MLPClassifier(algorithm='l-bfgs', random_state=0, hidden_layer_sizes=[10])

Default, activation function or nonlinearity is relu. 
Add one more layer or use tanh for smooth boundary.
mlp = MLPClassifier(algorithm='l-bfgs', random_state=0, hidden_layer_sizes=[10, 10])

mlp = MLPClassifier(algorithm='l-bfgs', activation='tanh', random_state=0, hidden_layer_sizes=[10, 10])

Finally using an l2 penality to shrink weight to 0.

mlp = MLPClassifier(algorithm='l-bfgs', random_state=0, hidden_layer_sizes=[n_hidden_nodes, n_hidden_nodes],
                    alpha=alpha)

Initially weight are set randomly. For large datset, it has not too much impact. But for small dataset, it is important factor.

## Breast Cancer Data
Start with Default.
mlp = MLPClassifier(random_state=42)
mlp.fit(X_train, y_train)

Accuracy is good but not as other model. It is due to different scaling of data.
    It hope, mean = 0 and variance = 1

// compute the mean value per feature on the training set
mean_on_train = X_train.mean(axis=0)
// compute the standard deviation of each feature on the training set
std_on_train = X_train.std(axis=0)
// subtract the mean, and scale by inverse standard deviation afterward, mean=0 and std=1
X_train_scaled = (X_train - mean_on_train) / std_on_train
// use THE SAME transformation (using training mean and std) on the test set
X_test_scaled = (X_test - mean_on_train) / std_on_train

Increasing the number of iterations only increased the training set performance, not the generalization performance.

some gap between the training and the test performance, we might try to decrease the model’s complexity to get better generalization performance. Here, we choose to increase the alpha parameter (quite aggressively, from 0.0001 to 1) to add stronger regularization of the weights.

mlp = MLPClassifier(max_iter=1000, alpha=1, random_state=0)

Note - Use keras, lasagna, tensor-flow instead of scikit-learn.
scikit-learn doesn't take benefit for GPU.

## Strength, Weakness, Parameters
Strength
    Capture information contained in large dataset.
    Build complem models
    Given enough time and tuning, NN can beat other ML Models.
Weakness
    Long time to train.

Homogenous data - SVM or NN
Different kind of features - Tree Based Models.

parameters -
 algorithm - default adam. other l-bfgs. Other sgd - mostly used by researcher.

## Estimate Complexity
Number of weights.
Approach - First create NN to large to overfit then shrink network or increase alpha.
