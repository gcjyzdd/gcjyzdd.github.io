---
layout: post
date:   2017-12-28 22:11
categories: DeepLearning Tensorflow
---

## Load Data

## Shuffle Data

## Reshape

Reshape a column to a row:
```
y_train = y_train.reshape(-1)
```

## Split Data

Example:

```
from sklearn.model_selection import train_test_split
X_train, X_valid, y_train, y_valid = train_test_split(X_train, y_train, test_size=0.3, random_state=42, stratify = y_train)
```

## Rotate images

## Resize images

## Padding

For the `'SAME'` padding, the output height and width are computed as:

```
out_height = ceil(float(in_height) / float(strides[1]))
out_width  = ceil(float(in_width) / float(strides[2]))
```
## Cropping2D

Crop top 70 pixels to remove unimportant environments and bottom 20 pixels of the car hood.

```
from keras.models import Sequential, Model
from keras.layers import Cropping2D
import cv2

# set up cropping2D layer
model = Sequential()
model.add(Cropping2D(cropping=((50,20), (0,0)), input_shape=(3,160,320)))
...
```

<div style="text-align:center"><img src ='{{"assets/Screenshot from 2017-12-29 22-46-33.png" | absolute_url}}' /></div>
