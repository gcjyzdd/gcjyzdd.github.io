---
layout: post
date:   2017-12-28 22:11
categories: DeepLearning Tensorflow
---

## Create virtual env with miniconda

Install `miniconda`:

```
wget https://repo.continuum.io/miniconda/Miniconda3-latest-Linux-x86_64.sh
chmod +x Miniconda3-latest-Linux-x86_64.sh
./Miniconda3-latest-Linux-x86_64.sh
source ~/.bashrc
```

Create `environment-gpu.yml`:
```
name: env-changjie
channels:
    - https://conda.anaconda.org/menpo
    - conda-forge
dependencies:
    - python==3.5.2
    - numpy
    - matplotlib
    - jupyter
    - opencv3
    - pillow
    - scikit-learn
    - scikit-image
    - scipy
    - h5py
    - eventlet
    - flask-socketio
    - seaborn
    - pandas
    - ffmpeg
    - imageio=2.1.2
    - pyqt=4.11.4
    - pip:
        - moviepy
        - https://storage.googleapis.com/tensorflow/linux/gpu/tensorflow_gpu-0.12.1-cp35-cp35m-linux_x86_64.whl
        - keras==1.2.1
```

Create `env`:
```
conda env create -f environment-gpu.yml
```

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
