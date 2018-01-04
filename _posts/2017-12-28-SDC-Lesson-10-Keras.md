---
layout: post
date:   2017-12-28 12:47
title: "SDC Lesson 10: Keras"
categories: Keras SDC
---

## Introduction

Use fewer lines to create deep neural networks.

## Deep Learning Frameworks

Behavioral cloning or end-to-end learning: the NN is learniong to predict steering angle and speed using only inputs from sensors.

## High Level Frameworks

Keras: sites on top of tensorflow and prvides a simplified interface.

## Keras Overview

Keras makes coding deep neural networks simpler. To demonstrate just how easy it is, you're going to build a simple fully-connected network in a few dozen lines of code.

We’ll be connecting the concepts that you’ve learned in the previous lessons to the methods that Keras provides.

The network you will build is similar to Keras’s sample network that builds out a convolutional neural network for MNIST. However for the network you will build you're going to use a small subset of the German Traffic Sign Recognition Benchmark dataset that you've used previously.

The general idea for this example is that you'll first load the data, then define the network, and then finally train the network.

## Neural Networks in Keras

### Sequential Model

```
from keras.models import Sequential

#Create the Sequential model
model = Sequential()
```

The `keras.models.Sequential` class is a wrapper for the neural network model. It provides common functions like `fit()`, `evaluate()`, and `compile()`. We'll cover these functions as we get to them. Let's start looking at the layers of the model.

### Layers

A Keras layer is just like a neural network layer. There are fully connected layers, max pool layers, and activation layers. You can add a layer to the model using the model's `add()` function. For example, a simple model would look like this:

```
from keras.models import Sequential
from keras.layers.core import Dense, Activation, Flatten

#Create the Sequential model
model = Sequential()

#1st Layer - Add a flatten layer
model.add(Flatten(input_shape=(32, 32, 3)))

#2nd Layer - Add a fully connected layer
model.add(Dense(100))

#3rd Layer - Add a ReLU activation layer
model.add(Activation('relu'))

#4th Layer - Add a fully connected layer
model.add(Dense(60))

#5th Layer - Add a ReLU activation layer
model.add(Activation('relu'))
```

Keras will automatically infer the shape of all layers after the first layer. This means you only have to set the input dimensions for the first layer.

The first layer from above, model.add(Flatten(input_shape=(32, 32, 3))), sets the input dimension to (32, 32, 3) and output dimension to (3072=32 x 32 x 3). The second layer takes in the output of the first layer and sets the output dimensions to (100). This chain of passing output to the next layer continues until the last layer, which is the output of the model.

### Quiz

In this quiz you will build a multi-layer feedforward neural network to classify traffic sign images using Keras.

1. Set the first layer to a Flatten() layer with the input_shape set to (32, 32, 3).

2. Set the second layer to a Dense() layer with an output width of 128.
Use a ReLU activation function after the second layer.
3. Set the output layer width to 5, because for this data set there are only 5 classes.
4. Use a softmax activation function after the output layer.
5. Train the model for 3 epochs. You should be able to get over 50% training accuracy.

To get started, review the Keras documentation about models and layers. The Keras example of [a Multi-Layer Perceptron](https://github.com/keras-team/keras/blob/master/examples/mnist_mlp.py) network is similar to what you need to do here. Use that as a guide, but keep in mind that there are a number of differences.

```
# mnist_mlp.py

from __future__ import print_function

import keras
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.optimizers import RMSprop

batch_size = 128
num_classes = 10
epochs = 20

# the data, shuffled and split between train and test sets
(x_train, y_train), (x_test, y_test) = mnist.load_data()

x_train = x_train.reshape(60000, 784)
x_test = x_test.reshape(10000, 784)
x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255
x_test /= 255
print(x_train.shape[0], 'train samples')
print(x_test.shape[0], 'test samples')

# convert class vectors to binary class matrices
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)

model = Sequential()
model.add(Dense(512, activation='relu', input_shape=(784,)))
model.add(Dropout(0.2))
model.add(Dense(512, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(num_classes, activation='softmax'))

model.summary()

model.compile(loss='categorical_crossentropy',
              optimizer=RMSprop(),
              metrics=['accuracy'])

history = model.fit(x_train, y_train,
                    batch_size=batch_size,
                    epochs=epochs,
                    verbose=1,
                    validation_data=(x_test, y_test))
score = model.evaluate(x_test, y_test, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])
```

```
# network.py

# Load pickled data
import pickle
import numpy as np
import tensorflow as tf
tf.python.control_flow_ops = tf

with open('../../Data/small-traffic-set/small_train_traffic.p', mode='rb') as f:
    data = pickle.load(f)

X_train, y_train = data['features'], data['labels']

# Initial Setup for Keras
from keras.models import Sequential
from keras.layers.core import Dense, Activation, Flatten

# TODO: Build the Fully Connected Neural Network in Keras Here
model = Sequential()

# 1st Layer - Add flatten
model.add(Flatten(input_shape=(32, 32, 3)))

#2nd Layer - Add a fully connected layer
model.add(Dense(128))

#3rd Layer - Add a ReLU activation layer
model.add(Activation('relu'))

# 4th Layer - 5 output
model.add(Dense(5))

# 5th Layer - Softmax
model.add(Activation('softmax'))


# preprocess data
X_normalized = np.array(X_train / 255.0 - 0.5 )

from sklearn.preprocessing import LabelBinarizer
label_binarizer = LabelBinarizer()
y_one_hot = label_binarizer.fit_transform(y_train)

model.compile('adam', 'categorical_crossentropy', ['accuracy'])
# TODO: change the number of training epochs to 3
history = model.fit(X_normalized, y_one_hot, nb_epoch=3, validation_split=0.2)

```

Output:

```
Train on 80 samples, validate on 20 samples
Epoch 1/3
32/80 [===========>..................] - ETA: 0s - loss: 1.7010 - acc: 0.2500
80/80 [==============================] - 0s - loss: 1.5112 - acc: 0.3875 - val_loss: 0.7146 - val_acc: 0.6500
Epoch 2/3
32/80 [===========>..................] - ETA: 0s - loss: 0.8754 - acc: 0.4688
80/80 [==============================] - 0s - loss: 0.8032 - acc: 0.6500 - val_loss: 0.5781 - val_acc: 0.7000
Epoch 3/3
32/80 [===========>..................] - ETA: 0s - loss: 0.6249 - acc: 0.6250
80/80 [==============================] - 0s - loss: 0.6633 - acc: 0.7125 - val_loss: 0.4719 - val_acc: 0.7500

Process finished with exit code 0
```

## Convolutions in Keras

1. Build from the previous network.
2. Add a [convolutional layer](https://keras.io/layers/convolutional/#convolution2d) with 32 filters, a 3x3 kernel, and valid padding before the flatten layer.
3. Add a ReLU activation after the convolutional layer.
4. Train for 3 epochs again, should be able to get over 50% accuracy.

Hint: The Keras example of a [convolutional neural network](https://github.com/keras-team/keras/blob/master/examples/mnist_cnn.py) for MNIST would be a good example to review.

```
# mnist_cnn.py

from __future__ import print_function
import keras
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras import backend as K

batch_size = 128
num_classes = 10
epochs = 12

# input image dimensions
img_rows, img_cols = 28, 28

# the data, shuffled and split between train and test sets
(x_train, y_train), (x_test, y_test) = mnist.load_data()

if K.image_data_format() == 'channels_first':
    x_train = x_train.reshape(x_train.shape[0], 1, img_rows, img_cols)
    x_test = x_test.reshape(x_test.shape[0], 1, img_rows, img_cols)
    input_shape = (1, img_rows, img_cols)
else:
    x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 1)
    x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 1)
    input_shape = (img_rows, img_cols, 1)

x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255
x_test /= 255
print('x_train shape:', x_train.shape)
print(x_train.shape[0], 'train samples')
print(x_test.shape[0], 'test samples')

# convert class vectors to binary class matrices
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)

model = Sequential()
model.add(Conv2D(32, kernel_size=(3, 3),
                 activation='relu',
                 input_shape=input_shape))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(num_classes, activation='softmax'))

model.compile(loss=keras.losses.categorical_crossentropy,
              optimizer=keras.optimizers.Adadelta(),
              metrics=['accuracy'])

model.fit(x_train, y_train,
          batch_size=batch_size,
          epochs=epochs,
          verbose=1,
          validation_data=(x_test, y_test))
score = model.evaluate(x_test, y_test, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])
```

```
# network.py
# Load pickled data
import pickle
import numpy as np
import tensorflow as tf
tf.python.control_flow_ops = tf

with open('../../Data/small-traffic-set/small_train_traffic.p', mode='rb') as f:
    data = pickle.load(f)

X_train, y_train= data['features'], data['labels']

# Initial Setup for Keras
from keras.models import Sequential
from keras.layers.core import Dense, Activation, Flatten
from keras.layers.convolutional import Convolution2D

# TODO: Build Convolutional Neural Network in Keras Here
model = Sequential()
model.add(Convolution2D(nb_filter=32,input_shape=(32, 32, 3),nb_col=3,nb_row=3,border_mode='valid'))
model.add(Activation('relu'))
model.add(Flatten())
model.add(Dense(128))
model.add(Activation('relu'))
model.add(Dense(5))
model.add(Activation('softmax'))

# Preprocess data
X_normalized = np.array(X_train / 255.0 - 0.5 )

from sklearn.preprocessing import LabelBinarizer
label_binarizer = LabelBinarizer()
y_one_hot = label_binarizer.fit_transform(y_train)

model.compile('adam', 'categorical_crossentropy', ['accuracy'])
history = model.fit(X_normalized, y_one_hot, nb_epoch=3, validation_split=0.2)
```
Output:
```
Using TensorFlow backend.
Train on 80 samples, validate on 20 samples
Epoch 1/3
32/80 [===========>..................] - ETA: 0s - loss: 1.5905 - acc: 0.2500
64/80 [=======================>......] - ETA: 0s - loss: 1.3997 - acc: 0.4062
80/80 [==============================] - 0s - loss: 1.4370 - acc: 0.4500 - val_loss: 1.5159 - val_acc: 0.7000
Epoch 2/3
32/80 [===========>..................] - ETA: 0s - loss: 1.8118 - acc: 0.5938
64/80 [=======================>......] - ETA: 0s - loss: 1.3614 - acc: 0.6406
80/80 [==============================] - 0s - loss: 1.2830 - acc: 0.6000 - val_loss: 0.5388 - val_acc: 0.7500
Epoch 3/3
32/80 [===========>..................] - ETA: 0s - loss: 0.8368 - acc: 0.5938
64/80 [=======================>......] - ETA: 0s - loss: 0.8626 - acc: 0.6250
80/80 [==============================] - 0s - loss: 0.8128 - acc: 0.6625 - val_loss: 0.3199 - val_acc: 0.8500

Process finished with exit code 0
```

## Pooling in Keras

1. Build from the previous network
2. Add a 2x2 [max pooling](https://keras.io/layers/pooling/#maxpooling2d) layer immediately following your convolutional layer.
3. Train for 3 epochs again. You should be able to get over 50% training accuracy.

```
# Load pickled data
import pickle
import numpy as np
import tensorflow as tf
tf.python.control_flow_ops = tf

with open('../../Data/small-traffic-set/small_train_traffic.p', mode='rb') as f:
    data = pickle.load(f)

X_train, y_train = data['features'], data['labels']

# Initial Setup for Keras
from keras.models import Sequential
from keras.layers.core import Dense, Activation, Flatten
from keras.layers.convolutional import Convolution2D
from keras.layers.pooling import MaxPooling2D

# TODO: Build Convolutional Neural Network in Keras Here
model = Sequential()
model.add(Convolution2D(32, 3, 3, input_shape=(32, 32, 3)))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Activation('relu'))
model.add(Flatten())
model.add(Dense(128))
model.add(Activation('relu'))
model.add(Dense(5))
model.add(Activation('softmax'))

# Preprocess data
X_normalized = np.array(X_train / 255.0 - 0.5 )

from sklearn.preprocessing import LabelBinarizer
label_binarizer = LabelBinarizer()
y_one_hot = label_binarizer.fit_transform(y_train)

model.compile('adam', 'categorical_crossentropy', ['accuracy'])
history = model.fit(X_normalized, y_one_hot, nb_epoch=3, validation_split=0.2)
```

## Dropout in Keras

1. Build from the previous network.
2. Add a [dropout](https://keras.io/layers/core/#dropout) layer after the pooling layer. Set the dropout rate to 50%.

```
# Load pickled data
import pickle
import numpy as np
import tensorflow as tf
tf.python.control_flow_ops = tf

with open('../../Data/small-traffic-set/small_train_traffic.p', mode='rb') as f:
    data = pickle.load(f)

X_train, y_train = data['features'], data['labels']

# Initial Setup for Keras
from keras.models import Sequential
from keras.layers.core import Dense, Activation, Flatten, Dropout
from keras.layers.convolutional import Convolution2D
from keras.layers.pooling import MaxPooling2D

# TODO: Build Convolutional Pooling Neural Network with Dropout in Keras Here
model = Sequential()
model.add(Convolution2D(32, 3, 3, input_shape=(32, 32, 3)))
model.add(MaxPooling2D((2, 2)))
model.add(Dropout(0.5))
model.add(Activation('relu'))
model.add(Flatten())
model.add(Dense(128))
model.add(Activation('relu'))
model.add(Dense(5))
model.add(Activation('softmax'))

# preprocess data
X_normalized = np.array(X_train / 255.0 - 0.5 )

from sklearn.preprocessing import LabelBinarizer
label_binarizer = LabelBinarizer()
y_one_hot = label_binarizer.fit_transform(y_train)

model.compile('adam', 'categorical_crossentropy', ['accuracy'])
history = model.fit(X_normalized, y_one_hot, nb_epoch=3, validation_split=0.2)
```

## Testing in Keras

Once you've picked out your best model, it's time to test it!

1. Try to get the highest validation accuracy possible. Feel free to use all the previous concepts and train for as many epochs as needed.
2. Select your best model and train it one more time.
3. Use the test data and the Keras `evaluate()` method to see how well the model does.

```
# Load pickled data
import pickle
import numpy as np
import tensorflow as tf

tf.python.control_flow_ops = tf

with open('../../Data/small-traffic-set/small_train_traffic.p', mode='rb') as f:
    data = pickle.load(f)

X_train, y_train = data['features'], data['labels']

# Initial Setup for Keras
from keras.models import Sequential
from keras.layers.core import Dense, Activation, Flatten, Dropout
from keras.layers.convolutional import Convolution2D
from keras.layers.pooling import MaxPooling2D

# TODO: Build the Final Test Neural Network in Keras Here
model = Sequential()
model.add(Convolution2D(32, 3, 3, border_mode='same',input_shape=(32, 32, 3)))
model.add(MaxPooling2D((2, 2)))
model.add(Dropout(0.5))
model.add(Activation('relu'))
model.add(Convolution2D(16, 3, 3,border_mode='same'))
model.add(MaxPooling2D((2, 2)))
model.add(Dropout(0.5))
model.add(Activation('relu'))
model.add(Flatten())
model.add(Dense(160))
model.add(Activation('relu'))
model.add(Dense(5))
model.add(Activation('softmax'))

# preprocess data
X_normalized = np.array(X_train / 255.0 - 0.5)

from sklearn.preprocessing import LabelBinarizer

label_binarizer = LabelBinarizer()
y_one_hot = label_binarizer.fit_transform(y_train)

model.compile('adam', 'categorical_crossentropy', ['accuracy'])
history = model.fit(X_normalized, y_one_hot, nb_epoch=30, validation_split=0.2)

with open('../../Data/small-traffic-set/small_test_traffic.p', mode='rb') as f:
    data_test = pickle.load(f)

X_test = data_test['features']
y_test = data_test['labels']

# preprocess data
X_normalized_test = np.array(X_test / 255.0 - 0.5)
y_one_hot_test = label_binarizer.fit_transform(y_test)

print("Testing")

# TODO: Evaluate the test data in Keras Here
metrics = model.evaluate(X_normalized_test,y_one_hot_test)
# TODO: UNCOMMENT CODE
for metric_i in range(len(model.metrics_names)):
    metric_name = model.metrics_names[metric_i]
    metric_value = metrics[metric_i]
    print('{}: {}'.format(metric_name, metric_value))

```
