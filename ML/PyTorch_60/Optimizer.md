# Training a model is an iterative process; 
    in each iteration the model makes a guess about the output
    calculates the error in its guess (loss)
    collects the derivatives of the error with respect to its parameters (as we saw in the previous section), 
    optimizes these parameters using gradient descent.

# Hyperparameters
Hyperparameters are adjustable parameters that let you control the model optimization process. Different hyperparameter values can impact model training and convergence rates.

We define the following hyperparameters for training:
    Number of Epochs - the number times to iterate over the dataset
    
    Batch Size - the number of data samples propagated through the network before the parameters are updated
    
    Learning Rate - how much to update models parameters at each batch/epoch. Smaller values yield slow learning speed, while large values may result in unpredictable behavior during training.

# Optimization Loop
we can then train and optimize our model with an optimization loop. Each iteration of the optimization loop is called an epoch.

Each epoch consists of two main parts:
    1. The Train Loop - iterate over the training dataset and try to converge to optimal parameters.
    2. The Validation/Test Loop - iterate over the test dataset to check if model performance is improving.

# Loss Function
Common loss functions include 
    nn.MSELoss (Mean Square Error) for regression tasks
    nn.NLLLoss (Negative Log Likelihood) for classification.
    nn.CrossEntropyLoss combines nn.LogSoftmax and nn.NLLLoss.

# Optimizer
Optimization is the process of adjusting model parameters to reduce model error in each training step. 
Optimization algorithms define how this process is performed (e.g. SGD).
    Others - ADAM and RMSProp

Inside the training loop, optimization happens in three steps:
    Call optimizer.zero_grad() to reset the gradients of model parameters. Gradients by default add up; to prevent double-counting, we explicitly zero them at each iteration.

    Backpropagate the prediction loss with a call to loss.backward(). PyTorch deposits the gradients of the loss w.r.t. each parameter.

    Once we have our gradients, we call optimizer.step() to adjust the parameters by the gradients collected in the backward pass.