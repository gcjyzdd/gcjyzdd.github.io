---
layout: post
date:   2018-01-02 20:51
title: "SDC Project 4: Advanced Lane Finding"
categories: LaneFinding SDC BehavioralCloning
author: Udacity
---

# Welcome to Computer Vision

<div style="text-align:center"><img src ='{{"assets/Screenshot from 2018-01-02 20-53-08.png" | absolute_url}}' /></div>


# Overview

Goals:

 * Lane detection: lines and curves --> steering angle?
 * Object detection: pedestrians, vehicles detection and tracking --> acceleration or braking? decision-making process



# Pinhole Camera Model

<div style="text-align:center"><img src ='{{"assets/Screenshot from 2018-01-02 21-04-25.png" | absolute_url}}' /></div>

<div style="text-align:center"><img src ='{{"assets/Screenshot from 2018-01-02 21-05-48.png" | absolute_url}}' /></div>

<div style="text-align:center"><img src ='{{"assets/Screenshot from 2018-01-02 21-07-05.png
" | absolute_url}}' /></div>

## Types of Distortion

Real cameras use curved lenses to form an image, and light rays often bend a little too much or too little at the edges of these lenses. This creates an effect that distorts the edges of images, so that lines or objects appear more or less curved than they actually are. This is called **radial distortion**, and it’s the most common type of distortion.

Another type of distortion, is **tangential distortion**. This occurs when a camera’s lens is not aligned perfectly parallel to the imaging plane, where the camera film or sensor is. This makes an image look tilted so that some objects appear farther away or closer than they actually are.

Tanential Distortion: it happens when the lens is not perfectly aligned parallel to the image plane.

<div style="text-align:center"><img src ='{{"assets/Screenshot from 2018-01-02 21-11-50.png" | absolute_url}}' /></div>
<div style="text-align:center"><img src ='{{"assets/Screenshot from 2018-01-02 21-12-55.png" | absolute_url}}' /></div>

# Finding Corners

In this exercise, you'll use the OpenCV functions `findChessboardCorners()` and `drawChessboardCorners()` to automatically find and draw corners in an image of a chessboard pattern.

To learn more about both of those functions, you can have a look at the OpenCV documentation here: [cv2.findChessboardCorners()](https://docs.opencv.org/2.4/modules/calib3d/doc/camera_calibration_and_3d_reconstruction.html#cv2.findChessboardCorners) and [cv2.drawChessboardCorners()](https://docs.opencv.org/2.4/modules/calib3d/doc/camera_calibration_and_3d_reconstruction.html#cv2.drawChessboardCorners).

Applying these two functions to a sample image, you'll get a result like this:

<div style="text-align:center"><img src ='{{"assets/corners-found3.jpg" | absolute_url}}' /></div>

In the following exercise, your job is simple. Count the number of corners in any given row and enter that value in nx. Similarly, count the number of corners in a given column and store that in ny. Keep in mind that "corners" are only points where two black and two white squares intersect, in other words, only count inside corners, not outside corners.

```
import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

# prepare object points
nx = 0#TODO: enter the number of inside corners in x
ny = 0#TODO: enter the number of inside corners in y

# Make a list of calibration images
fname = 'calibration_test.png'
img = cv2.imread(fname)

# Convert to grayscale
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Find the chessboard corners
ret, corners = cv2.findChessboardCorners(gray, (nx, ny), None)

# If found, draw corners
if ret == True:
    # Draw and display the corners
    cv2.drawChessboardCorners(img, (nx, ny), corners, ret)
    plt.imshow(img)
```

# Calibrating Your Camera

## Note Regarding Corner Coordinates

Since the origin corner is (0,0,0) the final corner is (6,4,0) relative to this corner rather than (7,5,0).

### Examples of Useful Code

Converting an image, imported by cv2 or the glob API, to grayscale:

```
gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
```

*Note*: If you are reading in an image using mpimg.imread() this will read in an **RGB** image and you should convert to grayscale using `cv2.COLOR_RGB2GRAY`, but if you are using cv2.imread() or the glob API, as happens in this video example, this will read in a **BGR** image and you should convert to grayscale using `cv2.COLOR_BGR2GRAY`. We'll learn more about color conversions later on in this lesson, but please keep this in mind as you write your own code and look at code examples.

Finding chessboard corners (for an 8x6 board):

```
ret, corners = cv2.findChessboardCorners(gray, (8,6), None)
```

Drawing detected corners on an image:

```
img = cv2.drawChessboardCorners(img, (8,6), corners, ret)
```

Camera calibration, given object points, image points, and the shape of the grayscale image:

```
ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)
```

Undistorting a test image:

```
dst = cv2.undistort(img, mtx, dist, None, mtx)
```

## A note on image shape

The shape of the image, which is passed into the calibrateCamera function, is just the height and width of the image. One way to retrieve these values is by retrieving them from the grayscale image shape array gray.shape[::-1]. This returns the image width and height in pixel values like (1280, 960).

Another way to retrieve the image shape, is to get them directly from the color image by retrieving the first two values in the color image shape array using img.shape[1::-1]. This code snippet asks for just the first two values in the shape array, and reverses them. Note that in our case we are working with a greyscale image, so we only have 2 dimensions (color images have three, height, width, and depth), so this is not necessary.

It's important to use an entire grayscale image shape or the first two values of a color image shape. This is because the entire shape of a color image will include a third value -- the number of color channels -- in addition to the height and width of the image. For example the shape array of a color image might be (960, 1280, 3), which are the pixel height and width of an image (960, 1280) and a third value (3) that represents the three color channels in the color image which you'll learn more about later, and if you try to pass these three values into the calibrateCamera function, you'll get an error.

# Lane Curvature

1. Detect the lane lines using some masking and thresholding techniques;
2. Perform a perspective transform to get a birds eye view of the lane.
3. Extract the curvature of the lines from polynomials


## Calculating Lane Curvature

Self-driving cars need to be told the correct steering angle to turn, left or right. You can calculate this angle if you know a few things about the speed and dynamics of the car and how much the lane is curving.

One way to calculate the curvature of a lane line, is to fit a 2nd degree polynomial to that line, and from this you can easily extract useful information.

For a lane line that is close to vertical, you can fit a line using this formula: **f(y) = Ay^2 + By + C**, where A, B, and C are coefficients.

A gives you the curvature of the lane line, B gives you the heading or direction that the line is pointing, and C gives you the position of the line based on how far away it is from the very left of an image (y = 0).

# Transform a Stop Sign


## Examples of Useful Code

Compute the perspective transform, M, given source and destination points:
```
M = cv2.getPerspectiveTransform(src, dst)
```

Compute the inverse perspective transform:
```
Minv = cv2.getPerspectiveTransform(dst, src)
```

Warp an image using the perspective transform, M:
```
warped = cv2.warpPerspective(img, M, img_size, flags=cv2.INTER_LINEAR)
```

*Note*: When you apply a perspective transform, choosing four source points manually, as we did in this video, is often not the best option. There are many other ways to select source points. For example, many perspective transform algorithms will programmatically detect four source points in an image based on edge or corner detection and analyzing attributes like color and surrounding pixels.

# Undistort and Transform Perspective

<div style="text-align:center"><img src ='{{"assets/undist-and-warp.png" | absolute_url}}' /></div>

<img src='{{"assets/undist-and-warp.png" | absolute_url}}' style="float: left; width: 45%; margin-right: 1%; margin-bottom: 0.5em;">
<img src='{{"assets/undist-and-warp2.png" | absolute_url}}' style="float: left; width: 45%; margin-right: 1%; margin-bottom: 0.5em;">
<p style="clear: both;">

Here's a tricky quiz for you! You have now seen how to find corners, calibrate your camera, undistort an image, and apply a perspective transform. Now it's your chance to perform all these steps on an image. In the last quiz you calibrated the camera, so here I'm giving you the camera matrix, `mtx`, and the distortion coefficients `dist` to start with.

Your goal is to generate output like the image shown above. To do that, you need to write a function that takes your distorted image as input and completes the following steps:

* Undistort the image using `cv2.undistort()` with `mtx` and `dist`
* Convert to grayscale
* Find the chessboard corners
* Draw corners
* Define 4 source points (the outer 4 corners detected in the chessboard pattern)
* Define 4 destination points (must be listed in the same order as src points!)
* Use `cv2.getPerspectiveTransform()` to get `M`, the transform matrix
* Use `cv2.warpPerspective()` to apply `M` and warp your image to a top-down view

*HINT*: Source points are the x and y pixel values of any four corners on your chessboard, you can extract these from the `corners` array output from `cv2.findChessboardCorners()`. Your destination points are the x and y pixel values of where you want those four corners to be mapped to in the output image.

If you run into any errors as you run your code, please refer to the Examples of Useful Code section in the previous video and make sure that your code syntax matches up! For this example, please also refer back to the examples in the 