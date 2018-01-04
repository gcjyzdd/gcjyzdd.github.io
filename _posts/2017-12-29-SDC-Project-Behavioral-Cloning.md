---
layout: post
date:   2017-12-29 13:59
title: "SDC Project 3: Behavioral Cloning"
categories: TransferLearning SDC BehavioralCloning
author: Udacity
---

## Resources For Completing the Project
You'll need a few files to complete the Behavioral Cloning Project.

The [GitHub repository](https://github.com/udacity/CarND-Behavioral-Cloning-P3) has the following files:

* drive.py: a Python script that you can use to drive the car autonomously, once your deep neural network model is trained
* writeup_template.md: a writeup template
* video.py: a script that can be used to make a video of the vehicle when it is driving autonomously

The simulator contains two tracks.

We encourage you to drive the vehicle in training mode and collect your own training data, but we have also included sample driving data for the first track, which you can optionally use to train your network. You may need to collect additional data in order to get the vehicle to stay on the road.

Here are links to the resources that you will need:

* [GitHub repository](https://github.com/udacity/CarND-Behavioral-Cloning-P3)
* [Sample Training Data](https://d17h27t6h515a5.cloudfront.net/topher/2016/December/584f6edd_data/data.zip)
* [Project Rubric](https://review.udacity.com/#!/rubrics/432/view)

Simulator Download

* [Linux](https://d17h27t6h515a5.cloudfront.net/topher/2017/February/58ae46bb_linux-sim/linux-sim.zip)
* [macOS](https://d17h27t6h515a5.cloudfront.net/topher/2017/February/58ae4594_mac-sim.app/mac-sim.app.zip)
* [Windows](https://d17h27t6h515a5.cloudfront.net/topher/2017/February/58ae4419_windows-sim/windows-sim.zip)

NOTE * On Windows 8 there is an issue where drive.py is unable to establish a data connection with the simulator. If you are running Windows 8 It is advised to upgrade to Windows 10, which should be free, and then you should be able to run the project properly.

Here are the newest updates to the simulator:

1. Steering is controlled via position mouse instead of keyboard. This creates better angles for training. Note the angle is based on the mouse distance. To steer hold the left mouse button and move left or right. To reset the angle to 0 simply lift your finger off the left mouse button.
2. You can toggle record by pressing R, previously you had to click the record button (you can still do that).
3. When recording is finished, saves all the captured images to disk at the same time instead of trying to save them while the car is still driving periodically. You can see a save status and play back of the captured data.
4. You can takeover in autonomous mode. While W or S are held down you can control the car the same way you would in training mode. This can be helpful for debugging. As soon as W or S are let go autonomous takes over again.
5. Pressing the spacebar in training mode toggles on and off cruise control (effectively presses W for you).
6. Added a Control screen
7. Track 2 was replaced from a mountain theme to Jungle with free assets , Note the track is challenging
8. You can use brake input in drive.py by issuing negative throttle values

If you are interested here is the source code for the [simulator repository](https://github.com/udacity/self-driving-car-sim).

## Using multiple Cameras

The simulator captures images from three cameras mounted on the car: a center, right and left camera. That’s because of the issue of recovering from being off-center.

In the simulator, you can weave all over the road and turn recording on and off to record recovery driving. In a real car, however, that’s not really possible. At least not legally.

So in a real car, we’ll have multiple cameras on the vehicle, and we’ll map recovery paths from each camera. For example, if you train the model to associate a given image from the center camera with a left turn, then you could also train the model to associate the corresponding image from the left camera with a somewhat softer left turn. And you could train the model to associate the corresponding image from the right camera with an even harder left turn.

In that way, you can simulate your vehicle being in different positions, somewhat further off the center line. To read more about this approach, see this [paper](http://images.nvidia.com/content/tegra/automotive/images/2016/solutions/pdf/end-to-end-dl-using-px.pdf) by our friends at NVIDIA that makes use of this technique.

### Explanation of How Multiple Cameras Work
The image below gives a sense for how multiple cameras are used to train a self-driving car. This image shows a bird's-eye perspective of the car. The driver is moving forward but wants to turn towards a destination on the left.

From the perspective of the left camera, the steering angle would be less than the steering angle from the center camera. From the right camera's perspective, the steering angle would be larger than the angle from the center camera. The next section will discuss how this can be implemented in your project although there is no requirement to use the left and right camera images.

<div style="text-align:center"><img src ='{{"assets/carnd-using-multiple-cameras.png" | absolute_url}}' /></div>

Multiple Cameras in This Project
For this project, recording recoveries from the sides of the road back to center is effective. But it is also possible to use all three camera images to train the model. When recording, the simulator will simultaneously save an image for the left, center and right cameras. Each row of the csv log file, driving_log.csv, contains the file path for each camera as well as information about the steering measurement, throttle, brake and speed of the vehicle.

Here is some example code to give an idea of how all three images can be used:

```
    with open(csv_file, 'r') as f:
        reader = csv.reader(f)
        for row in reader:
            steering_center = float(row[3])

            # create adjusted steering measurements for the side camera images
            correction = 0.2 # this is a parameter to tune
            steering_left = steering_center + correction
            steering_right = steering_center - correction

            # read in images from center, left and right cameras
            path = "..." # fill in the path to your training IMG directory
            img_center = process_image(np.asarray(Image.open(path + row[0])))
            img_left = process_image(np.asarray(Image.open(path + row[1])))
            img_right = process_image(np.asarray(Image.open(path + row[2])))

            # add images and angles to data set
            car_images.extend(img_center, img_left, img_right)
            steering_angles.extend(steering_center, steering_left, steering_right)
```

During training, you want to feed the left and right camera images to your model as if they were coming from the center camera. This way, you can teach your model how to steer if the car drifts off to the left or the right.

Figuring out how much to add or subtract from the center angle will involve some experimentation.

During prediction (i.e. "autonomous mode"), you only need to predict with the center camera image.

It is not necessary to use the left and right images to derive a successful model. Recording recovery driving from the sides of the road is also effective.

## Pipeline

1. Preprocessing
    * normalize
    * zero mean centering
2. Powerful network
3. Data augmentation(flip left right)   
4. Using multiple cameras
5. Cropping images
6. Even more powerful network
7. More data collection
    * two or three laps of center lane driving
    * one lap of recovery driving from the sides
    * one lap focusing on driving smoothly around curves

Deep CNN:
<div style="text-align:center"><img src ='{{"assets/cnn-architecture-624x890.png" | absolute_url}}' /></div>


## Visualizing Loss

### Outputting Training and Validation Loss Metrics
In Keras, the `model.fit()` and `model.fit_generator()` methods have a `verbose` parameter that tells Keras to output loss metrics as the model trains. The `verbose` parameter can optionally be set to `verbose = 1` or `verbose = 2`.

Setting `model.fit(verbose = 1)` will

* output a progress bar in the terminal as the model trains.
* output the loss metric on the training set as the model trains.
* output the loss on the training and validation sets after each epoch.

With `model.fit(verbose = 2)`, Keras will only output the loss on the training set and validation set after each epoch.

### Model History Object
When calling `model.fit()` or `model.fit_generator()`, Keras outputs a history object that contains the training and validation loss for each epoch. Here is an example of how you can use the history object to visualize the loss:
<div style="text-align:center"><img src ='{{"assets/screen-shot-2017-02-14-at-8.29.09-pm.png" | absolute_url}}' /></div>

The following code shows how to use the `model.fit()` history object to produce the visualization.

```
from keras.models import Model
import matplotlib.pyplot as plt

history_object = model.fit_generator(train_generator, samples_per_epoch =
    len(train_samples), validation_data = 
    validation_generator,
    nb_val_samples = len(validation_samples), 
    nb_epoch=5, verbose=1)

### print the keys contained in the history object
print(history_object.history.keys())

### plot the training and validation loss for each epoch
plt.plot(history_object.history['loss'])
plt.plot(history_object.history['val_loss'])
plt.title('model mean squared error loss')
plt.ylabel('mean squared error loss')
plt.xlabel('epoch')
plt.legend(['training set', 'validation set'], loc='upper right')
plt.show()
```

## Generators

### How to Use Generators

The images captured in the car simulator are much larger than the images encountered in the Traffic Sign Classifier Project, a size of 160 x 320 x 3 compared to 32 x 32 x 3. Storing 10,000 traffic sign images would take about 30 MB but storing 10,000 simulator images would take over 1.5 GB. That's a lot of memory! Not to mention that preprocessing data can change data types from an `int` to a `float`, which can increase the size of the data by a factor of 4.

Generators can be a great way to work with large amounts of data. Instead of storing the preprocessed data in memory all at once, using a generator you can pull pieces of the data and process them on the fly only when you need them, which is much more memory-efficient.

A generator is like a [coroutine](https://en.wikipedia.org/wiki/Coroutine), a process that can run separately from another main routine, which makes it a useful Python function. Instead of using `return`, the generator uses `yield`, which still returns the desired output values but saves the current values of all the generator's variables. When the generator is called a second time it re-starts right after the `yield` statement, with all its variables set to the same values as before.

Below is a short quiz using a generator. This generator appends a new Fibonacci number to its list every time it is called. To pass, simply modify the generator's `yield` so it returns a list instead of `1`. The result will be we can get the first 10 Fibonacci numbers simply by calling our generator 10 times. If we need to go do something else besides generate Fibonacci numbers for a while we can do that and then always just call the generator again whenever we need more Fibonacci numbers.

```
def fibonacci():
    numbers_list = []
    while 1:
        if(len(numbers_list) < 2):
            numbers_list.append(1)
        else:
            numbers_list.append(numbers_list[-1] + numbers_list[-2])
        yield numbers_list # change this line so it yields its list instead of 1

our_generator = fibonacci()
my_output = []

for i in range(10):
    my_output = (next(our_generator))
    
print(my_output)

# Output
# [1, 1, 2, 3, 5, 8, 13, 21, 34, 55]
```

Here is an example of how you could use a generator to load data and preprocess it on the fly, in batch size portions to feed into your Behavioral Cloning model .

```
import os
import csv

samples = []
with open('./driving_log.csv') as csvfile:
    reader = csv.reader(csvfile)
    for line in reader:
        samples.append(line)

from sklearn.model_selection import train_test_split
train_samples, validation_samples = train_test_split(samples, test_size=0.2)

import cv2
import numpy as np
import sklearn

def generator(samples, batch_size=32):
    num_samples = len(samples)
    while 1: # Loop forever so the generator never terminates
        shuffle(samples)
        for offset in range(0, num_samples, batch_size):
            batch_samples = samples[offset:offset+batch_size]

            images = []
            angles = []
            for batch_sample in batch_samples:
                name = './IMG/'+batch_sample[0].split('/')[-1]
                center_image = cv2.imread(name)
                center_angle = float(batch_sample[3])
                images.append(center_image)
                angles.append(center_angle)

            # trim image to only see section with road
            X_train = np.array(images)
            y_train = np.array(angles)
            yield sklearn.utils.shuffle(X_train, y_train)

# compile and train the model using the generator function
train_generator = generator(train_samples, batch_size=32)
validation_generator = generator(validation_samples, batch_size=32)

ch, row, col = 3, 80, 320  # Trimmed image format

model = Sequential()
# Preprocess incoming data, centered around zero with small standard deviation 
model.add(Lambda(lambda x: x/127.5 - 1.,
        input_shape=(ch, row, col),
        output_shape=(ch, row, col)))
model.add(... finish defining the rest of your model architecture here ...)

model.compile(loss='mse', optimizer='adam')
model.fit_generator(train_generator, samples_per_epoch= /
            len(train_samples), validation_data=validation_generator, /
            nb_val_samples=len(validation_samples), nb_epoch=3)
```