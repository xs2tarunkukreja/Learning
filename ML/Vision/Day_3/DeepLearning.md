# Why do we need pretrained model for classification?
ImageNet = 1.2M Train + 100K Test + 50K Validation + 10K Classes
CNN - Legends; LeNet-5; AlexNet; VGG16; GoogleNet

## VGG16
Input (224*224*3) > [ Conv 3*3 > Conv 3*3 > MaxPool 2*2 ] *2 > [ Conv 3*3 > Conv 3*3 > Conv 3*3 > MaxPool 2*2 ] *3 > 4096 > 4096 > 1000

import os
import numpy as np
import matplotlib.pyplot as plt

import keras
from keras.models import Sequential
from keras.layers import Dense, Conv2D, MaxPool2D , Flatten, Dropout

#Instantiating the model
model =  Sequential()

#Adding our first convolutional layer
#with activation ReLU
#This will produce 64 feature maps

model.add(Conv2D(input_shape=(224,224,3),
                 filters=64,
                 kernel_size=(3,3),
                 padding="same", 
                 activation="relu"))

#Adding conv layers
model.add(Conv2D(filters=64,
                 kernel_size=(3,3),
                 padding="same", 
                 activation="relu"))

#Adding Pooling layer
model.add(MaxPool2D(pool_size=(2,2),
                    strides=(2,2)))

#Adding conv layers
model.add(Conv2D(filters=128, 
                 kernel_size=(3,3), 
                 padding="same",
                 activation="relu"))

#Adding conv layers
model.add(Conv2D(filters=128, 
                 kernel_size=(3,3), 
                 padding="same", 
                 activation="relu"))

model.add(MaxPool2D(pool_size=(2,2),
                    strides=(2,2)))



#Adding conv layers
model.add(Conv2D(filters=256, 
                 kernel_size=(3,3), 
                 padding="same",
                 activation="relu"))

#Adding conv layers
model.add(Conv2D(filters=256, 
                 kernel_size=(3,3), 
                 padding="same", 
                 activation="relu"))

#Adding conv layers
model.add(Conv2D(filters=256, 
                 kernel_size=(3,3),
                 padding="same",
                 activation="relu"))


#Adding Pooling layer
model.add(MaxPool2D(pool_size=(2,2),
                    strides=(2,2)))
#Adding conv layers
model.add(Conv2D(filters=512, 
                 kernel_size=(3,3), 
                 padding="same", 
                 activation="relu"))

#Adding conv layers
model.add(Conv2D(filters=512, 
                 kernel_size=(3,3), 
                 padding="same",
                 activation="relu"))

#Adding conv layers
model.add(Conv2D(filters=512,
                 kernel_size=(3,3), 
                 padding="same", 
                 activation="relu"))

model.add(MaxPool2D(pool_size=(2,2),
                    strides=(2,2)))
#Adding conv layers
model.add(Conv2D(filters=512, 
                 kernel_size=(3,3), 
                 padding="same", 
                 activation="relu"))

#Adding conv layers
model.add(Conv2D(filters=512, 
                 kernel_size=(3,3), 
                 padding="same", 
                 activation="relu"))

#Adding conv layers
model.add(Conv2D(filters=512,
                 kernel_size=(3,3), 
                 padding="same", 
                 activation="relu"))

#Adding Pooling layer
model.add(MaxPool2D(pool_size=(2,2),
                    strides=(2,2)))

#Flattening feature maps to vectors
model.add(Flatten())

#Adding Dense layers for classification
model.add(Dense(units=4096,activation="relu"))
model.add(Dense(units=4096,activation="relu"))
model.add(Dense(units=1000, activation="softmax"))


model.compile(loss=keras.losses.categorical_crossentropy,
              optimizer=keras.optimizers.Adam(),
              metrics=['accuracy'])

model.summary()


## Inception Layer

