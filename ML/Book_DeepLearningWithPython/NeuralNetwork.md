# Anatomy of Neural Network
Training a NN revolve around the following objects
    Layers
    Input Data and Corresponding Targets
    Loss Function - Which define feedback signal which is used for learning
    Optimizer - Which determine how the learning proceeds.

Input > Network composed of layer > Prediction > Loss Function compare Prediction and Actual Output > Optimizer use loss value to update weights.
## Layers - Lego Brick of Deep Learning
A layer is data processing module that take input one or more tensors and output one or more tensors.
Some layer are stateless But most have states: weights.
Layers learn with stochastic gradient descent.

Different layers are appropriate for different tensor formats and different types of data processing.
    Fully Connected Layers - Densely-Connected - Dense Layers: Simple Vector Data stored in 2D tensors. (samples, features)
    Recurrent Layer - LSTM Layer - Sequence Data stored in 3D tensor (samples, timesteps, features)
    2D Convolution Layer (conv2d) - Image Data stored in 4D tensors.

Building Deep Learning Model - Clipping together compatible layers to form useful data transformation pipeline.
    Layer compatibility means every layer accept input tensor of particular shape and resturn output of particular shape.

Example -
    from keras import layers
    layer = layers.Dense(32, input_shape=(784,)) // 32 Output Unit.
    // It accept only 2D tensor where first D is 784. It return a tensor with first D 32. It can connect to layer that accept 32D vector

    from keras import models
    model = models.Sequential()
    model.add(layers.Dense(32, input_shape=(784,)))
    model.add(layer.Dense(32)) // Here input is automatically calculated.
## Models - Network of Layers
DAG of layers. 
Most common is linear stack of layers.

Common Network Topology
    Two Branch Networks
    Multi-Head Networks
    Inception Block
The topology define an "hypothesis space". It contrained space to specific series of tensor operations, mapping input to output.

Picking right network architecture is an art.
## Loss Function and Optimizer: Key of configuring learning process
Loss function will be minimized during learning.
Optimizer implement a specific variant of stochastic gradient descent.

A NN with multiple output may have mulitple loss functions, however gradient descent process must be based on single scaler loss value. So, we average to all into 1 scaler value.

Choose loss function wisely.
    2 Class Classification - Bianry Crossentropy
    Many Class Classification - Categorical Crossentropy
    Regression - Mean Squared Error.
    Sequence Learning Problem - CTC
# Introduction of Keras
It is Deep Learning Framework for Python.

Key Features
    It allows the same code on CPU as well as GPU
    User Friendly API
    Support for Convolutional Networks(Vision), Recurrent Network(Sequence Processing) and any combination of both

## Keras, Tensor-Flow, Theano and CNTK
TensorFlow (Google), CNTK (Microsoft), Theano (MILA Lab) are some main platform for Deep Learning.

Keras doen't handle low level operation such as tensor manipulation and differntiation. It use backend engines. It support all 3 Tensorflow, Thenano and CNTK backend.

Preferred One is TensorFlow for backend.

## Developing with Keras: Quick Overview
Workflow
    Define your training data, input tensors and target tensors.
    Define network layers (or model)
    Configure learning process - loss function, optimizer and some metrics to measures.
    Iterate on your training data.

There are 2 ways to define model - Sequential class and funtional API.
Functional API:
    input_tensor = layers.Input(shape=(784,))
    x = layers.Dense(32, activation='relu')(input_tensor)
    output_tensor = layers.Dense(10, activation='softmax')(x)
    model = models.Model(input=input_tensor, output=output_tensor)

from keras import optimizers
model.compile(optimizer=optimizers.RMSprop(lr=0.001), loss='mse', metrics=['accuracy'])
model.fit(input_tensor, target_tensor, batch_size=128, epochs=10)

# Setting Up Work Station
## Preliminary Consideration
NVIDIA GPU or Cloud
Unix - Ubuntu
TensorFlow or CTNK or Theano
Jupytor Notebook

# Classification Moview Review
IMDB data is already preprocessed: review(sequence of words) have been turned into sequence of integers where each integer stand for specific word in dictionary.

Dense(16, activation='relu') // 16 is hidden unit in layers.
output = relu(dot(W, input) + b)
W will have shape (input_dimension, 16)

Epoch means iteration.

ClassificationMovieReview.ipynb

# Why are activation function necessary?
output = dot(W, input) + b // Without activation function. It learn only linear transformation.
Hypothesis Space would be 16 D only. Even if we increase hidden layer, it still remain same space 16D.

We need non-linearity. 

# Classifying Newswire: multi-class classification
ClassifyingNewswireMulticlass.ipynb

# Predicting House Price: Regression 
PredictingHousePrice.ipynb
