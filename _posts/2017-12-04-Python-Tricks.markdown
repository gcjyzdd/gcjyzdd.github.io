---
layout: post
title:  "Python Tricks"
date:   2017-12-04 22:20
categories: Python Numpy
---

### Table of Contents
[Numpy](#Numpy)

# Python

## FIFO

You can add items using the `append` method and remove them using `pop`. For a `LIFO` this would look like this:

```
stack = list()
stack.append(1)
stack.append(2)
stack.append(3)

print stack.pop()  #3
print stack.pop()  #2
print stack.pop()  #1
```

If you supply an integer argument for `pop` you can specify which element to remove. For a FIFO use the index 0 for the first element:

```
stack = list()
stack.append(1)
stack.append(2)
stack.append(3)

print stack.pop(0)  #1
print stack.pop(0)  #2
print stack.pop(0)  #3
```

[Ref](https://stackoverflow.com/questions/19219903/python-first-in-first-out-print)

##  list: difference between append and extend

```
>>> stack = ['a','b']
>>> stack.append('c')
>>> stack
['a', 'b', 'c']


>>> stack.append(['e','f'])
>>> stack
['a', 'b', 'c', ['e', 'f']]


>>> stack.extend(['g','h'])
>>> stack
['a', 'b', 'c', ['e', 'f'], 'g', 'h']
```

Example:
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


# Numpy

## Transform a row to a column

Transform a row to a column:

```
import numpy as np


# a row
a = np.array([1, 2, 3])

# transform a to a column vector
a[:, None]
```

## numpy.empty_like

`numpy.empty_like(a, dtype=None, order='K', subok=True)` Return a new array with the same shape and type as a given array. 

Example:

```
>>> a = ([1,2,3], [4,5,6])                         # a is array-like
>>> np.empty_like(a)
array([[-1073741821, -1073741821,           3],    #random
       [          0,           0, -1073741821]])
>>> a = np.array([[1., 2., 3.],[4.,5.,6.]])
>>> np.empty_like(a)
array([[ -2.00000715e+000,   1.48219694e-323,  -2.00000572e+000],#random
       [  4.38791518e-305,  -2.00000715e+000,   4.17269252e-309]])
```

[Ref](https://docs.scipy.org/doc/numpy-1.13.0/reference/generated/numpy.empty_like.html)

## Load multiple images

```
from PIL import Image
x = np.array([np.array(Image.open(fname)) for fname in filelist])
```

To save a numpy array to file using pickle:

```
import pickle
pickle.dump( x, filehandle, protocol=2 )
```

## numpy.squeeze

`numpy.squeeze(a, axis=None)[source]` Remove single-dimensional entries from the shape of an array.

```
>>> x = np.array([[[0], [1], [2]]])
>>> x.shape
(1, 3, 1)
>>> np.squeeze(x).shape
(3,)
>>> np.squeeze(x, axis=0).shape
(3, 1)
>>> np.squeeze(x, axis=1).shape
Traceback (most recent call last):
...
ValueError: cannot select an axis to squeeze out which has size not equal to one
>>> np.squeeze(x, axis=2).shape
(1, 3)
```

## numpy.expand_dims

```
>>> x = np.array([1,2])
>>> x.shape
(2,)
```

The following is equivalent to `x[np.newaxis,:]` or `x[np.newaxis]`:

```
>>> y = np.expand_dims(x, axis=0)
>>> y
array([[1, 2]])
>>> y.shape
(1, 2)
```

```
>>> y = np.expand_dims(x, axis=1)  # Equivalent to x[:,newaxis]
>>> y
array([[1],
       [2]])
>>> y.shape
(2, 1)
```

Here is a real example:

```
print(gray_tmp.shape)
gray_tmp = gray_tmp[: ,:,:,np.newaxis]
print(gray_tmp.shape)

(34799, 32, 32)
(34799, 32, 32, 1)
```


## Create image from array

```
from PIL import Image
import numpy as np

w, h = 512, 512
data = np.zeros((h, w, 3), dtype=np.uint8)
data[256, 256] = [255, 0, 0]
img = Image.fromarray(data, 'RGB')
img.save('my.png')
img.show()
```

## RGB2Gray

The rgb2gray formula is:

```
Y' = 0.299 R + 0.587 G + 0.114 B 
```

We can apply this formula to implement rgb2gray as the following:

```
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

def rgb2gray(rgb):
    return np.dot(rgb[...,:3], [0.299, 0.587, 0.114])

img = mpimg.imread('image.png')     
gray = rgb2gray(img)    
plt.imshow(gray, cmap = plt.get_cmap('gray'))
plt.show()
```


## numpy.concatenate

`numpy.concatenate((a1, a2, ...), axis=0)` Join a sequence of arrays along an existing axis.

Example:

```
>>> a = np.array([[1, 2], [3, 4]])
>>> b = np.array([[5, 6]])
>>> np.concatenate((a, b), axis=0)
array([[1, 2],
       [3, 4],
       [5, 6]])
>>> np.concatenate((a, b.T), axis=1)
array([[1, 2, 5],
       [3, 4, 6]])
```

## numpy.mean

Syntax:

`numpy.mean(a, axis=None, dtype=None, out=None, keepdims=<class numpy._globals._NoValue>)`

Example:

```
>>> a = np.array([[1, 2], [3, 4]])
>>> np.mean(a)
2.5
>>> np.mean(a, axis=0)
array([ 2.,  3.])
>>> np.mean(a, axis=1)
array([ 1.5,  3.5])
```

## numpy.ndarray.astype

Syntax:

`ndarray.astype(dtype, order='K', casting='unsafe', subok=True, copy=True)`

Example:

```
>>> x = np.array([1, 2, 2.5])
>>> x
array([ 1. ,  2. ,  2.5])
>>> x.astype(int)
array([1, 2, 2])
```

## numpy.amax

```
>>> a = np.arange(4).reshape((2,2))
>>> a
array([[0, 1],
       [2, 3]])
>>> np.amax(a)           # Maximum of the flattened array
3
>>> np.amax(a, axis=0)   # Maxima along the first axis
array([2, 3])
>>> np.amax(a, axis=1)   # Maxima along the second axis
array([1, 3])
```

## numpy.divide

```
>>> np.divide(2.0, 4.0)
0.5
>>> x1 = np.arange(9.0).reshape((3, 3))
>>> x2 = np.arange(3.0)
>>> np.divide(x1, x2)
array([[ NaN,  1. ,  1. ],
       [ Inf,  4. ,  2.5],
       [ Inf,  7. ,  4. ]])
```

## numpy.mean


## numpy.maximum

## numpy.absolute

# Tensorflow

## tf.image.resize_images

`tf.image.resize_images` to resize images.

Example:
```
"""
The traffic signs are 32x32 so you
have to resize them to be 227x227 before
passing them to AlexNet.
"""
import time
import tensorflow as tf
import numpy as np
from scipy.misc import imread
from caffe_classes import class_names
from alexnet import AlexNet

x = tf.placeholder(tf.float32, (None, 32, 32, 3))
# TODO: Resize the images so they can be fed into AlexNet.
# HINT: Use `tf.image.resize_images` to resize the images
resized = tf.image.resize_images(x, (227,227))

assert resized is not Ellipsis, "resized needs to modify the placeholder image size to (227,227)"
probs = AlexNet(resized)

init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init)

# Read Images
im1 = imread("construction.jpg").astype(np.float32)
im1 = im1 - np.mean(im1)

im2 = imread("stop.jpg").astype(np.float32)
im2 = im2 - np.mean(im2)

# Run Inference
t = time.time()
output = sess.run(probs, feed_dict={x: [im1, im2]})

# Print Output
for input_im_ind in range(output.shape[0]):
    inds = np.argsort(output)[input_im_ind, :]
    print("Image", input_im_ind)
    for i in range(5):
        print("%s: %.3f" % (class_names[inds[-1 - i]], output[input_im_ind, inds[-1 - i]]))
    print()

print("Time: %.3f seconds" % (time.time() - t))

```