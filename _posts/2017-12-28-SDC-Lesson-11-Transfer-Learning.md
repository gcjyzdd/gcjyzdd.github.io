---
layout: post
date:   2017-12-28 14:20
title: "SDC Lesson 11: Transfer Learning"
categories: TransferLearning SDC
author: Udacity
---

## Introduction

Engineers didn't start with a blank slate when they're building neural networks. Starting from scratch can be time-consuming. It's not just architecting the network, but also experimenting with it, training it and adjusting it, which can take days or even weeks.

To accelerate the process, engineers often begin with a pre-trained network and the modify it. Fine-tuning an existing network is a powerful techique because improving a network takes much less effort than creating one from scratch. Going even further, we can take an existing network and re-purpose it for a realted different task. 

Re-purposing a network is called Transfer Learning, because you're transferring the learning from an existing  network to a new one. 

## Transfer Learning

When you're tackling a new problem with a neural network, it might help to start with an existing neural network that was built for a similar task and then try to fine-tune it for your own task. There are a couple of good reasons to do this:

1. Existing neural network can be really useful. If somebody has taken days or weeks to train a neural network already, then a lot of intelligence is stored in that network. Taking advantage of that work can accelerate yopur own process.
2. Sometimes the data set for the problem you'll working on might be small. In those cases, look for an existing network that's designed for a similar problem for your own. If that network has already been trained on a large data set, then you can use it as a starting point to help your own network to generalize better. 

In order to do this, it's better to know the most prominent pre-trained networks that already exist. 

### The Four Main Cases When Using Transfer Learning

Transfer learning involves taking a pre-trained neural network and adapting the neural network to a new, different data set.

Depending on both:

* the size of the new data set, and
* the similarity of the new data set to the original data set

the approach for using transfer learning will be different. There are four main cases:

1. new data set is small, new data is similar to original training data
2. new data set is small, new data is different from original training data
3. new data set is large, new data is similar to original training data
4. new data set is large, new data is different from original training data

![]({{site.baseurl}}/assets/Screenshot from 2017-12-28 15-01-32.png)

A large data set might have one million images. A small data could have two-thousand images. The dividing line between a large data set and small data set is somewhat subjective. Overfitting is a concern when using transfer learning with a small data set.

Images of dogs and images of wolves would be considered similar; the images would share common characteristics. A data set of flower images would be different from a data set of dog images.

Each of the four transfer learning cases has its own approach. In the following sections, we will look at each case one by one.

### Demonstration Network
To explain how each situation works, we will start with a generic pre-trained convolutional neural network and explain how to adjust the network for each case. Our example network contains three convolutional layers and three fully connected layers:

![]({{site.baseurl}}/assets/Screenshot from 2017-12-28 15-05-40.png)

Here is an generalized overview of what the convolutional neural network does:

* the first layer will detect edges in the image
* the second layer will detect shapes
* the third convolutional layer detects higher level features

Each transfer learning case will use the pre-trained convolutional neural network in a different way.

### Case 1: Small Data Set, Similar Data

![]({{site.baseurl}}/assets/Screenshot from 2017-12-28 15-07-44.png)

If the new data set is small and similar to the original training data:

* slice off the end of the neural network
* add a new fully connected layer that matches the number of classes in the new data set
* randomize the weights of the new fully connected layer; freeze all the weights from the pre-trained network
* train the network to update the weights of the new fully connected layer

To avoid overfitting on the small data set, the weights of the original network will be held constant rather than re-training the weights.

Since the data sets are similar, images from each data set will have similar higher level features. Therefore most or all of the pre-trained neural network layers already contain relevant information about the new data set and should be kept.

Here's how to visualize this approach:

![]({{site.baseurl}}/assets/Screenshot from 2017-12-28 15-09-21.png)

### Case 2: Small Data Set, Different Data

![]({{site.baseurl}}/assets/Screenshot from 2017-12-28 15-10-19.png)

If the new data set is small and different from the original training data:

* slice off most of the pre-trained layers near the beginning of the network
* add to the remaining pre-trained layers a new fully connected layer that matches the number of classes in the new data set
* randomize the weights of the new fully connected layer; freeze all the weights from the pre-trained network
* train the network to update the weights of the new fully connected layer

Because the data set is small, overfitting is still a concern. To combat overfitting, the weights of the original neural network will be held constant, like in the first case.

But the original training set and the new data set do not share higher level features. In this case, the new network will only use the layers containing lower level features.

Here is how to visualize this approach:

![]({{site.baseurl}}/assets/Screenshot from 2017-12-28 15-11-38.png)

### Case 3: Large Data Set, Similar Data

![]({{site.baseurl}}/assets/Screenshot from 2017-12-28 15-12-49.png)

If the new data set is large and similar to the original training data:

* remove the last fully connected layer and replace with a layer matching the number of classes in the new data set
* randomly initialize the weights in the new fully connected layer
* initialize the rest of the weights using the pre-trained weights
* re-train the entire neural network

Overfitting is not as much of a concern when training on a large data set; therefore, you can re-train all of the weights.

Because the original training set and the new data set share higher level features, the entire neural network is used as well.

Here is how to visualize this approach:

![]({{site.baseurl}}/assets/Screenshot from 2017-12-28 15-14-13.png)

### Case 4: Large Data Set, Different Data

![]({{site.baseurl}}/assets/Screenshot from 2017-12-28 15-15-21.png)

If the new data set is large and different from the original training data:

* remove the last fully connected layer and replace with a layer matching the number of classes in the new data set
* retrain the network from scratch with randomly initialized weights
* alternatively, you could just use the same strategy as the "large and similar" data case

Even though the data set is different from the training data, initializing the weights from the pre-trained network might make training faster. So this case is exactly the same as the case with a large, similar data set.

If using the pre-trained network as a starting point does not produce a successful model, another option is to randomly initialize the convolutional neural network weights and train the network from scratch.

Here is how to visualize this approach:

![]({{site.baseurl}}/assets/Screenshot from 2017-12-28 15-17-00.png)

## Feature Extraction

The problem is that AlexNet was trained on the ImageNet database, which has 1000 classes of images. You can see the classes in the `caffe_classes.py` file. None of those classes involves traffic signs.

In order to successfully classify our traffic sign images, you need to remove the final, 1000-neuron classification layer and replace it with a new, 43-neuron classification layer.

This is called feature extraction, because you're basically extracting the image features inferred by the penultimate layer, and passing these features to a new classification layer.

Open `feature_extraction.py` and complete the `TODO`(s).

Your output will probably not precisely match the sample output below, since the output will depend on the (probably random) initialization of weights in the network. That being said, the output classes you see should be present in `signnames.csv`.

## VGG

The architecture of vgg network:

![]({{site.baseurl}}/assets/Screenshot from 2017-12-28 21-41-52.png)

## GoogLeNet

### Inception module:

![]({{site.baseurl}}/assets/Screenshot from 2017-12-28 21-48-13.png)

The total number of parameters is small and that's why GoogLeNet runs fast.

## ResNet

![]({{site.baseurl}}/assets/Screenshot from 2017-12-28 21-51-31.png)

## Transfer Learning with VGG, Inception (GoogLeNet) and ResNet

[The CIFAR-10 dataset](https://www.cs.toronto.edu/~kriz/cifar.html)

[ImageNet Dataset](http://www.image-net.org/)

In this lab, you will continue exploring transfer learning. You've already explored feature extraction with AlexNet and TensorFlow. Next, you will use Keras to explore feature extraction with the VGG, Inception and ResNet architectures. The models you will use were trained for days or weeks on the ImageNet dataset. Thus, the weights encapsulate higher-level features learned from training on thousands of classes.

There are some notable differences from AlexNet lab.

1. We're using two datasets. First, the German Traffic Sign dataset, and second, the Cifar10 dataset.
2. **Bottleneck Features**. Unless you have a very powerful GPU, running feature extraction on these models will take a significant amount of time, as you might have observed in the AlexNet lab. To make things easier we've precomputed bottleneck features for each (network, dataset) pair. This will allow you to experiment with feature extraction even on a modest CPU. You can think of bottleneck features as feature extraction but with caching. Because the base network weights are frozen during feature extraction, the output for an image will always be the same. Thus, once the image has already been passed through the network, we can cache and reuse the output.
3. Furthermore, we've limited each class in both training datasets to 100 examples. The idea here is to push feature extraction a bit further. It also greatly reduces the download size and speeds up training. The validation files remain the same.
The files are encoded as such:

* {network}_{dataset}_100_bottleneck_features_train.p
* {network}_{dataset}_bottleneck_features_validation.p

"network", in the above filenames, can be one of 'vgg', 'inception', or 'resnet'.

"dataset" can be either 'cifar10' or 'traffic'.

### Getting Started
1. Download one of the bottleneck feature packs. VGG is the smallest so you might want to give that a shot first. You can download these from the Supplement Materials at the bottom of this page.
2. Clone the lab repository
```
git clone https://github.com/udacity/CarND-Transfer-Learning-Lab
cd CarND-Transfer-Learning-Lab
```

Open feature_extraction.py in your favourite text editor. We'll go over the code next.


## Cifar10 Aside

Before you try feature extraction on pretrained models I'd like you to take a moment and run the classifier you used in the Traffic Sign project on the Cifar10 dataset. Cifar10 images are also (32, 32, 3) so the main thing you'll need to change is **the number of classes from 43 to 10**. Cifar10 also doesn't come with a validation set, so you can randomly split training data into a training and validation.

You can easily download and load the Cifar10 dataset like this:

```
from keras.datasets import cifar10
(X_train, y_train), (X_test, y_test) = cifar10.load_data()
# y_train.shape is 2d, (50000, 1). While Keras is smart enough to handle this
# it's a good idea to flatten the array.
y_train = y_train.reshape(-1)
y_test = y_test.reshape(-1)
```

You can then use sklearn to split off part of the data into a validation set:

```
from sklearn.model_selection import train_test_split
X_train, X_valid, y_train, y_valid = train_test_split(X_train, y_train, test_size=0.3, random_state=42, stratify = y_train)
```

The Cifar10 dataset contains 10 classes:

<div style="text-align:center"><img src ='{{site.baseurl}}/assets/Screenshot from 2017-12-28 22-20-48.png' /></div>

While the German Traffic Sign dataset has more classes, the Cifar10 dataset is harder to classify due to the complexity of the classes. A ship is drastically different from a frog, and a frog is nothing like a deer, etc. These are the kind of datasets where the advantage of using a pre-trained model will become much more apparent.

Train your model on the Cifar10 dataset and record your results, keep these in mind when you train from the bottleneck features. Don't be discouraged if you get results significantly worse than the Traffic Sign dataset.

Using LeNet, the validation accuracy is 64.8%.

## Feature Extraction with cifar10

Here is my code:
```
import pickle
import tensorflow as tf
# TODO: import Keras layers you need here
from keras.models import Sequential
from keras.layers.core import Dense, Activation, Flatten, Dropout
from keras.layers.convolutional import Convolution2D
from keras.layers.pooling import MaxPooling2D


flags = tf.app.flags
FLAGS = flags.FLAGS

# command line flags
flags.DEFINE_string('training_file', '', "Bottleneck features training file (.p)")
flags.DEFINE_string('validation_file', '', "Bottleneck features validation file (.p)")
flags.DEFINE_integer('BATCH', 128, 'Btach size')
flags.DEFINE_integer('EPOCH', 20, 'Epochs')

def load_bottleneck_data(training_file, validation_file):
    """
    Utility function to load bottleneck features.

    Arguments:
        training_file - String
        validation_file - String
    """
    print("Training file", training_file)
    print("Validation file", validation_file)

    with open(training_file, 'rb') as f:
        train_data = pickle.load(f)
    with open(validation_file, 'rb') as f:
        validation_data = pickle.load(f)

    X_train = train_data['features']
    y_train = train_data['labels']
    X_val = validation_data['features']
    y_val = validation_data['labels']

    return X_train, y_train, X_val, y_val


def main(_):
    # load bottleneck data
    X_train, y_train, X_val, y_val = load_bottleneck_data(FLAGS.training_file, FLAGS.validation_file)

    print(X_train.shape, y_train.shape)
    print(X_val.shape, y_val.shape)

    # TODO: define your model and hyperparams here
    # make sure to adjust the number of classes based on
    # the dataset
    # 10 for cifar10
    # 43 for traffic
    n_class = 10
    BATCH = FLAGS.BATCH
    EPOCH = FLAGS.EPOCH
    rate = 1e-3
    shp = X_train.shape[1:]

    model = Sequential()
    model.add(Flatten(input_shape = shp))
    model.add(Dropout(0.5))
    model.add(Dense(480))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    model.add(Dense(n_class))
    model.add(Activation('softmax'))

    from sklearn.preprocessing import LabelBinarizer

    label_binarizer = LabelBinarizer()
    y_one_hot = label_binarizer.fit_transform(y_train)

    # TODO: train your model here
    model.compile('adam', 'categorical_crossentropy', ['accuracy'])
    history = model.fit(X_train, y_one_hot, nb_epoch=EPOCH, validation_split=0.2)

    print("Testing")

    # TODO: Evaluate the test data in Keras Here
    metrics = model.evaluate(X_val, label_binarizer.fit_transform(y_val))

    for metric_i in range(len(model.metrics_names)):
        metric_name = model.metrics_names[metric_i]
        metric_value = metrics[metric_i]
        print('{}: {}'.format(metric_name, metric_value))


# parses flags and calls the `main` function above
if __name__ == '__main__':
    tf.app.run()

```

Here is the result:

| Model     | Accuracy  |
|:---------:|:---------:|
| vgg       | 0.77      |
| inception | 0.64      |
| resnet    | 0.71      |

## Summary

You've trained AlexNet, VGG, GoogLeNet, and ResNet as feature extractors!

To end this lab, let's summarize when we should consider:

1. Feature extraction (train only the top-level of the network, the rest of the network remains fixed)
2. Finetuning (train the entire network end-to-end, start with pre-trained weights)
3. Training from scratch (train the entire network end-to-end, start from random weights)

Consider feature extraction when ...

... the new dataset is small and similar to the original dataset. The higher-level features learned from the original dataset should transfer well to the new dataset.

Consider finetuning when ...

... the new dataset is large and similar to the original dataset. Altering the original weights should be safe because the network is unlikely to overfit the new, large dataset.

... the new dataset is small and very different from the original dataset. You could also make the case for training from scratch. If you choose to finetune, it might be a good idea to only use features from the first few layers of the pre-trained network; features from the final layers of the pre-trained network might be too specific to the original dataset.

Consider training from scratch when ...

... the dataset is large and very different from the original dataset. In this case we have enough data to confidently train from scratch. However, even in this case it might be beneficial to initialize the entire network with pretrained weights and finetune it on the new dataset.

Finally, keep in mind that for a lot of problems you won't need an architecture as complicated and powerful as VGG, Inception, or ResNet. These architectures were made for the task of classifying thousands of complex classes. A smaller network might be a better fit for a smaller problem, especially if you can comfortably train it on moderate hardware.