# How NN Works?

Input Layer > Hidden Layers > Output Layer

## One perceptron

x1 -- w1

x2 -- w2
            -- z -- f -- Output
x3 -- w3

x4 -- w4


## Process -
    Inputs + Weights > Model > Results > Performace > Update Weights and rerun.

## Back Propagation
After getting output, we calculate loss and Back Propogate to adjust all the weights to reduce loss.

# Computer Vision Tasks
Object Detection 
6D Pose Detection (Location and Orientation)
Generate Models

# CNN Architecture

Inputs > Convolution (N Layers) > Pooling (N Layers) > Fully Connected > Output
<---          Feature Extraction                 ---> <--  Classification   -->

## What is wrong with Fully Connected NN?
image is 2D grid of pixels. 
NN expects a vector of number as inputs.

## Convolution Operation
Image Matrix CO Weigth Matrix = Output Volume

## Stride
Slide by one columns at a time = Stride 1
Slide by 2 columns at a time = Stride 2

## Max Pooling
Stride Convolution: Kernel slides along the image with a step > 1
Dilated Convolution: Kernel is spread out, step > 1 between kernel elements.

## Size of Output
 = (W-F+2P)/S + 1
 W = Width or height of image
 F = Size of kernel or filter
 P = Padding size.
 s is stride.

 ## Activation Function
 Sigmoid
 tanh
 ReLU
 Leaky ReLU
 Maxout
 ELU

 ## ML Pipeline
 Original Dataset = Training Set + Test Set
 Training Set = Training Set + Validation Set

 Training Set is use to Train Model
 Validation Set is use to evaluate model and then backtrack.

 Test Set = At end of cycle to test the performance.

 ## Key Performance Metrics
 Accuracy
 Cross Entropy
 Precision
 Recall
 F1 Score
 ROC AUC
 PR AUC
 Mean Average Precision
 Root Mean Squared Error
 Mean Absolute Error

 # CNN
 from keras.datasets import mnist
 (x_train, y_train), (x_test, y_test) = mnist.load_data()

from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras import backend as K

batch_size = 128
num_classes = 10
epochs = 5

img_rows, img_col = 28, 28 # Image Size

X_train = X_train.reshape(x_train.shape[0], img_rows, img_cols, 1)

#Normalizing input between 0 and 1
x_train /= 255
x_test /= 255


model = Sequential()

#Adding our first convolutional layer
#with activation ReLU
#This will produce 32 feature maps
model.add(Conv2D(32, kernel_size=(3, 3),
                 activation='relu',
                 input_shape=input_shape))
model.add(Conv2D(64, (3, 3), activation='relu'))# 3*3*1
model.add(MaxPooling2D(pool_size=(2, 2)))
#Adding Dropout for regularisation
model.add(Dropout(0.25))

#Flattening feature maps to vectors
model.add(Flatten())

#Adding Dense layers for classification
model.add(Dense(128, activation='relu'))

model.add(Dropout(0.5))

model.add(Dense(num_classes, activation='softmax'))


#Compiling with cross entropy loss
model.compile(loss=keras.losses.categorical_crossentropy,
              optimizer=keras.optimizers.Adam(),
              metrics=['accuracy'])

model.summary()

Image (28*28*1) > Convolutional (32 Channels) (26*26*32) > (64 Channels (24*24*64)> 64 Channels(12*12*64)) Max Pooling 9 (2*2) 

## Evaluate
score = model.evaluate(x_test, y_test, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])

history = model.fit(x_train, y_train,
          batch_size=batch_size,
          epochs=epochs,
          verbose=1,
          validation_data=(x_test, y_test))
# Plot training & validation accuracy values
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Model accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
plt.show()

# Plot training & validation loss values
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
plt.show()

## Visualize Kernels

## Visualize Feature Maps
