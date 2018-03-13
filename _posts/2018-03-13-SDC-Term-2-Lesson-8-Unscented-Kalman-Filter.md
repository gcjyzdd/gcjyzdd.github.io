---
layout: post
date:   2018-03-08 21:40
categories: SDC UKF DataFusion
title: Unscented Kalman Filter
---

<script src="https://cdnjs.cloudflare.com/ajax/libs/mathjax/2.7.0/MathJax.js?config=TeX-AMS-MML_HTMLorMML" type="text/javascript"></script>


## The CTRV Model

### Motion Models and Kalman Filters

In the extended kalman filter lesson, we used a constant velocity model (CV). A constant velocity model is one of the most basic motion models used with object tracking.

But there are many other models including:

* constant turn rate and velocity magnitude model (CTRV)
* constant turn rate and acceleration (CTRA)
* constant steering angle and velocity (CSAV)
* constant curvature and acceleration (CCA)

Each model makes different assumptions about an object's motion. In this lesson, you will work with the CTRV model.

Keep in mind that you can use any of these motion models with either the extended Kalman filter or the unscented Kalman filter, but we wanted to expose you to more than one motion model.

### Robot Motion and Trigonometry
Motion model development relies on some essential concepts of trigonometry. As a trigonometry refresher in the context of robot motion, we have created this [content](https://classroom.udacity.com/nanodegrees/nd013/parts/40f38239-66b6-46ec-ae68-03afd8a601c8/modules/0949fca6-b379-42af-a919-ee50aa304e6a/lessons/4c4f9180-7ab8-42ba-a1bd-24b17fcf2178/concepts/3d7a14af-6afa-446e-9399-622360eddd6c).

### Limitations of the Constant Velocity (CV) Model

<div style="text-align:center"><img src ='{{site.baseurl}}/assets/SDC-T2/screenshot-from-2017-02-27-20-35-58.png' /></div>

The model performs badly when it's turning assuming that vehicles are moving with constant speed.

<div style="text-align:center"><img src ='{{site.baseurl}}/assets/SDC-T2/Screenshot from 2018-03-13 22-13-35.png' /></div>

## The CTRV Model State Vector


<div style="text-align:center"><img width='100%' src ='{{site.baseurl}}/assets/SDC-T2/Screenshot from 2018-03-13 22-15-36.png' /></div>

<div style="text-align:center"><img src ='{{site.baseurl}}/assets/SDC-T2/screenshot-from-2017-02-27-20-45-49.png' /></div>

## The CTRV Differential Equation

<div style="text-align:center"><img src ='{{site.baseurl}}/assets/SDC-T2/Screenshot from 2018-03-13 22-22-43.png' /></div>

<div style="text-align:center"><img src ='{{site.baseurl}}/assets/SDC-T2/Screenshot from 2018-03-13 22-28-25.png' /></div>

### CTRV Integral


<div style="text-align:center"><img src ='{{site.baseurl}}/assets/SDC-T2/Screenshot from 2018-03-13 22-30-24.png' /></div>

<div style="text-align:center"><img src ='{{site.baseurl}}/assets/SDC-T2/Screenshot from 2018-03-13 22-32-21.png' /></div>


<div style="text-align:center"><img src ='{{site.baseurl}}/assets/SDC-T2/Screenshot from 2018-03-13 22-34-11.png' /></div>

## CTRV Zero Yaw Rate

<div style="text-align:center"><img src ='{{site.baseurl}}/assets/SDC-T2/Screenshot from 2018-03-13 22-38-06.png' /></div>

When yaw rate is zero, we get zero as denumenators. And we need to derive the process equation again.

Since yaw rate is zero, the vehicle is moving straightly. As a result, it's very easy to calculate chnage of x position and y position.

## CTRV Process Noise Vector

<div style="text-align:center"><img src ='{{site.baseurl}}/assets/SDC-T2/Screenshot from 2018-03-13 22-48-26.png' /></div>

<div style="text-align:center"><img src ='{{site.baseurl}}/assets/SDC-T2/Screenshot from 2018-03-13 22-50-50.png' /></div>


## CTRV Process Noise Position

<div style="text-align:center"><img src ='{{site.baseurl}}/assets/SDC-T2/Screenshot from 2018-03-13 22-57-16.png' /></div>





