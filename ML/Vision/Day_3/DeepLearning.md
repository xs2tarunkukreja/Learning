# Why do we need pretrained model for classification?
Pretrained Model - Already trained.
Why we needed?
    Infrastructure + Amount of Data needed to train model (labelled as well)- Save it.

WordNet
ImageNet - Image Data Public Repository
    2012 - First CNN with 16%. Before this ML with higher error rate. - AlexNet
    2014 - VggNet - 8%
    2018 - Google - Inception Net - 3%. Human - 5% error rate.

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

## LeNet-5
Input (32*32*1) > conv 5*5 T > avg-pool 2*2 > conv 5*5 T > avg-pool 2*2 > T (120) > T(80) > S(10)
    10 classifications.
    avg-pool: We are losing edges and sharpness. - Problem.
    Tanh - Problem

## AlexNet
Input (224*224*3) > conv 11*11 R > max-pool 3*3 > conv 5*5 R > max-pool 3*3 > [conv 3*3]*3 R > max-pool 3*3 > R (4096) > R(4096) > S(1000)
    Issues - Lots of parameters. 
        Reduce window size

## VGG 16
Input (224*224*3) > (conv 3*3 R > conv 3*3 R > max-pool 2*2)*2 > (conv 3*3 R > conv 3*3 R > conv 3*3 R > max-pool 2*2 )*3> R (4096) > R(4096) > S(1000)
    It increase depth and reduce filter size.. so less parameters.
    Problem - If depth increase, chances of gradient vanishing.

Note - Should use small size filters.
       Depth should be more.

## Global Average/Max Pooling
h = 6, w = 6 , d = 3 => h=1, w=1, d=3. With one value, we can extract information.

## 1*1 Convolutions
w,h,d = w,h, d=1
## Inception Layer
1. Introduce Auxiliary Classifier in middle layer to avoid gradient vanishing effects.
2. Keep model wider more than deeper.
3. RMSProp Optimization - Learning rate change.
   7*7 Filters
   BatchNorm in Auxiliary Classifier - Normalize data.
   Label Smoothing - Regularizing component added to loss formula.. so, network become over confident abount a class.

Stem Block - sequential convolutional block.

Inception Block - Multiple filters in a layer. Layer have horizontal layers as well with differnet window size. functional model. Last operation is concat... it concat all o/p's.

Reduction Block 

Multiple kernel size at same layer.

1*1, 1*7, 3*3 etc.

## Resnet50
Skip Connection - 


Conv Block - 2 path.. 1 path have more,, 2nd have 1 conv.. > add
Idnetity Block - 2 path - 1 path have more, 2nd nothing > add.

Now we can compare the output of 2 path directly.. no need for gradient things in Deep Learning.


# Image Generator
To generate synthatic data.
One Image > Generate variation by performing various operation.

Data Augumentation
    Crop; Symmetry; Rotation; Scale; Original; Noise; Hue; Blur; Obstruction