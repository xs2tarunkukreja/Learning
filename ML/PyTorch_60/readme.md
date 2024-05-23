# Tensors
Data Structure = array or matrics.
It encode i/p and o/p of a model and model's parameters.
Tensors are for GPU and other specialized hardware to accelerate computing.

tensor.ipynb

# A Gentle introduction to torch.autograd
It is pytorch automatic differentiation engine that powers NN training.
## Background
NN is a collection of nested functions that are executed on some input data. These functions are defined by parameters(weight and biases)

Forward Propagation - It run input data to though each of its functions to make guess.

Backward Propagation - Based on error in guess, it adjust its parameters. It does this by traversing backwards from the output, collecting the derivatives of the error with respect to the parameters of the functions (gradients), and optimizing the parameters using gradient descent.

autograd.ipynb

# Neural Network

nn.ipynb

# Training a Classifier

Generally, when you have to deal with image, text, audio or video data, you can use standard python packages that load data into a numpy array. Then you can convert this array into a torch.*Tensor.

For images, packages such as Pillow, OpenCV are useful
For audio, packages such as scipy and librosa
For text, either raw Python or Cython based loading, or NLTK and SpaCy are useful

classifier.ipynb


