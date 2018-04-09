---
layout: post
date:   2018-03-08 21:40
categories: SDC UKF DataFusion
title: Unscented Kalman Filter
---

<script src="https://cdnjs.cloudflare.com/ajax/libs/mathjax/2.7.0/MathJax.js?config=TeX-AMS-MML_HTMLorMML" type="text/javascript"></script>

## Formulas

<iframe src ='{{site.baseurl}}/assets/SDC-T2/UKF_summary.pdf' width="100%" height="800em"></iframe>

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

We approximate that the vehicle is going perfectly straight and simplify the noise of position x and y.

<div style="text-align:center"><img src ='{{site.baseurl}}/assets/SDC-T2/Screenshot from 2018-03-14 21-40-08.png' /></div> 

<div style="text-align:center"><img src ='{{site.baseurl}}/assets/SDC-T2/Screenshot from 2018-03-14 21-40-25.png' /></div> 

## UKF Process Chain

Unscented Kalman Filter Introduction
Now that you have learned the CTRV motion model equations, we will discuss how the unscented Kalman filter works. As you go through the lectures, recall that the extended Kalman filter uses the Jacobian matrix to linearize non-linear functions.

The unscented Kalman filter, on the other hand, does not need to linearize non-linear functions; instead, the unscented Kalman filter takes representative points from a Gaussian distribution. These points will be plugged into the non-linear equations as you'll see in the lectures.

**Unscented transform**

## What Problem Does the UKF Solve?

<div style="text-align:center"><iframe width="560" height="315" src="https://www.youtube.com/embed/OFb47Lu9JfM" frameborder="0" allow="autoplay; encrypted-media" allowfullscreen></iframe></div>

<div style="text-align:center"><img src ='{{site.baseurl}}/assets/SDC-T2/Screenshot from 2018-03-14 22-07-27.png' /></div> 

<div style="text-align:center"><img src ='{{site.baseurl}}/assets/SDC-T2/Screenshot from 2018-03-14 22-12-39.png' /></div> 

## UKF Basics Unscented Transform

<div style="text-align:center"><iframe width="560" height="315" src="https://www.youtube.com/embed/8jbckHQDl4A" frameborder="0" allow="autoplay; encrypted-media" allowfullscreen></iframe></div>

<div style="text-align:center"><img src ='{{site.baseurl}}/assets/SDC-T2/Screenshot from 2018-03-14 22-15-38.png' /></div> 

1. A good way to choose sigma points
2. How to predict the sigma points
3. Calculate the prediction mean and covariance

## Generating Sigma Points

<div style="text-align:center"><img src ='{{site.baseurl}}/assets/SDC-T2/Screenshot from 2018-03-14 22-20-54.png' /></div> 

<div style="text-align:center"><img src ='{{site.baseurl}}/assets/SDC-T2/Screenshot from 2018-03-14 22-23-05.png' /></div> 

<div style="text-align:center"><img src ='{{site.baseurl}}/assets/SDC-T2/Screenshot from 2018-03-14 22-24-19.png' /></div> 

<div style="text-align:center"><img src ='{{site.baseurl}}/assets/SDC-T2/Screenshot from 2018-03-14 22-25-43.png' /></div> 

<div style="text-align:center"><img src ='{{site.baseurl}}/assets/SDC-T2/Screenshot from 2018-03-14 22-26-05.png' /></div> 

<div style="text-align:center"><img src ='{{site.baseurl}}/assets/SDC-T2/Screenshot from 2018-03-14 22-28-37.png' /></div> 

<div style="text-align:center"><img src ='{{site.baseurl}}/assets/SDC-T2/Screenshot from 2018-03-14 22-31-31.png' /></div> 

## Generating Sigma Points

<div style="text-align:center"><img src ='{{site.baseurl}}/assets/SDC-T2/Screenshot from 2018-03-15 21-26-51.png' /></div> 

* [Eigen Quick Reference Guide](https://eigen.tuxfamily.org/dox/group__QuickRefPage.html)
* [Eigen Documentation of Cholesky Decomposition](https://eigen.tuxfamily.org/dox/classEigen_1_1LLT.html)

Please note that the algorithm used in the quiz `(P.llt().matrixL())` produces the lower triangular matrix `L` of the matrix `P` such that `P = L*L^`.

```cpp
#include <iostream>
#include "ukf.h"

UKF::UKF() {
  //TODO Auto-generated constructor stub
  Init();
}

UKF::~UKF() {
  //TODO Auto-generated destructor stub
}

void UKF::Init() {

}

/*******************************************************************************
* Programming assignment functions: 
*******************************************************************************/


void UKF::GenerateSigmaPoints(MatrixXd* Xsig_out) {

  //set state dimension
  int n_x = 5;

  //define spreading parameter
  double lambda = 3 - n_x;

  //set example state
  VectorXd x = VectorXd(n_x);
  x <<   5.7441,
         1.3800,
         2.2049,
         0.5015,
         0.3528;

  //set example covariance matrix
  MatrixXd P = MatrixXd(n_x, n_x);
  P <<     0.0043,   -0.0013,    0.0030,   -0.0022,   -0.0020,
          -0.0013,    0.0077,    0.0011,    0.0071,    0.0060,
           0.0030,    0.0011,    0.0054,    0.0007,    0.0008,
          -0.0022,    0.0071,    0.0007,    0.0098,    0.0100,
          -0.0020,    0.0060,    0.0008,    0.0100,    0.0123;

  //create sigma point matrix
  MatrixXd Xsig = MatrixXd(n_x, 2 * n_x + 1);

  //calculate square root of P
  MatrixXd A = P.llt().matrixL();

/*******************************************************************************
 * Student part begin
 ******************************************************************************/

  //your code goes here 
  
  //calculate sigma points ...
  //set sigma points as columns of matrix Xsig
  Xsig.col(0) = x;

  float a = std::sqrt(3);

  for(size_t i=0; i<n_x; i++)
  {
	  Xsig.col(1+i) = x + a*A.col(i);
  }
  for(size_t i=0; i<n_x; i++)
  {
	  Xsig.col(6+i) = x - a*A.col(i);
  }
/*******************************************************************************
 * Student part end
 ******************************************************************************/

  //print result
  //std::cout << "Xsig = " << std::endl << Xsig << std::endl;

  //write result
  *Xsig_out = Xsig;


/* expected result:
   Xsig =
    5.7441  5.85768   5.7441   5.7441   5.7441   5.7441  5.63052   5.7441   5.7441   5.7441   5.7441
      1.38  1.34566  1.52806     1.38     1.38     1.38  1.41434  1.23194     1.38     1.38     1.38
    2.2049  2.28414  2.24557  2.29582   2.2049   2.2049  2.12566  2.16423  2.11398   2.2049   2.2049
    0.5015  0.44339 0.631886 0.516923 0.595227   0.5015  0.55961 0.371114 0.486077 0.407773   0.5015
    0.3528 0.299973 0.462123 0.376339  0.48417 0.418721 0.405627 0.243477 0.329261  0.22143 0.286879
*/

}
```

## UKF Augmentation

<div style="text-align:center"><img src ='{{site.baseurl}}/assets/SDC-T2/Screenshot from 2018-03-15 21-57-24.png' /></div> 

<div style="text-align:center"><img src ='{{site.baseurl}}/assets/SDC-T2/Screenshot from 2018-03-15 21-58-22.png' /></div> 

<div style="text-align:center"><img src ='{{site.baseurl}}/assets/SDC-T2/Screenshot from 2018-03-15 21-59-41.png' /></div> 

## Augmentation Assignment

<div style="text-align:center"><img src ='{{site.baseurl}}/assets/SDC-T2/Screenshot from 2018-03-15 22-06-57.png' /></div> 

<div style="text-align:center"><img src ='{{site.baseurl}}/assets/SDC-T2/Screenshot from 2018-03-15 22-07-16.png' /></div> 


## Sigma Point Prediction

<div style="text-align:center"><img src ='{{site.baseurl}}/assets/SDC-T2/Screenshot from 2018-03-15 22-30-22.png' /></div> 

## Sigma Point Prediction Assignment

<div style="text-align:center"><img src ='{{site.baseurl}}/assets/SDC-T2/Screenshot from 2018-03-15 22-37-55.png' /></div> 

## Predict Mean and Covariance

<div style="text-align:center"><img src ='{{site.baseurl}}/assets/SDC-T2/Screenshot from 2018-03-21 21-07-24.png' /></div> 

<div style="text-align:center"><img src ='{{site.baseurl}}/assets/SDC-T2/Screenshot from 2018-03-21 21-13-12.png' /></div> 

## Predict Mean and Covariance Assignment

<div style="text-align:center"><img src ='{{site.baseurl}}/assets/SDC-T2/Screenshot from 2018-03-21 21-15-28.png' /></div> 

## Measurement Prediction

<div style="text-align:center"><img src ='{{site.baseurl}}/assets/SDC-T2/Screenshot from 2018-03-21 21-38-27.png' /></div> 

<div style="text-align:center"><img src ='{{site.baseurl}}/assets/SDC-T2/Screenshot from 2018-03-21 21-40-52.png' /></div> 

<div style="text-align:center"><img src ='{{site.baseurl}}/assets/SDC-T2/Screenshot from 2018-03-21 21-41-56.png' /></div> 

## Predict Radar Measurement Assignment

<div style="text-align:center"><img src ='{{site.baseurl}}/assets/SDC-T2/Screenshot from 2018-03-21 21-44-50.png' /></div> 

<div style="text-align:center"><img src ='{{site.baseurl}}/assets/SDC-T2/Screenshot from 2018-03-21 21-45-13.png' /></div> 

## UKF Updates

<div style="text-align:center"><img src ='{{site.baseurl}}/assets/SDC-T2/Screenshot from 2018-03-21 22-34-06.png' /></div> 

<div style="text-align:center"><img src ='{{site.baseurl}}/assets/SDC-T2/Screenshot from 2018-03-21 22-34-55.png' /></div> 

## UKF Update Assignment

<div style="text-align:center"><img src ='{{site.baseurl}}/assets/SDC-T2/Screenshot from 2018-03-21 22-58-45.png' /></div> 

## Parameters and Consistency

<div style="text-align:center"><img src ='{{site.baseurl}}/assets/SDC-T2/Screenshot from 2018-03-22 22-29-55.png' /></div> 

<div style="text-align:center"><img src ='{{site.baseurl}}/assets/SDC-T2/Screenshot from 2018-03-22 22-30-54.png' /></div> 

<div style="text-align:center"><img src ='{{site.baseurl}}/assets/SDC-T2/Screenshot from 2018-03-22 22-32-07.png' /></div> 

<div style="text-align:center"><img src ='{{site.baseurl}}/assets/SDC-T2/Screenshot from 2018-03-22 22-33-17.png' /></div> 

<div style="text-align:center"><img src ='{{site.baseurl}}/assets/SDC-T2/Screenshot from 2018-03-22 22-33-57.png' /></div> 

<div style="text-align:center"><img src ='{{site.baseurl}}/assets/SDC-T2/Screenshot from 2018-03-22 22-34-21.png' /></div> 

<div style="text-align:center"><img src ='{{site.baseurl}}/assets/SDC-T2/Screenshot from 2018-03-22 22-34-58.png' /></div> 

<div style="text-align:center"><img src ='{{site.baseurl}}/assets/SDC-T2/Screenshot from 2018-03-22 22-36-47.png' /></div> 

<div style="text-align:center"><img src ='{{site.baseurl}}/assets/SDC-T2/Screenshot from 2018-03-22 22-37-50.png' /></div> 

<div style="text-align:center"><img src ='{{site.baseurl}}/assets/SDC-T2/Screenshot from 2018-03-22 22-38-20.png' /></div> 

<div style="text-align:center"><img src ='{{site.baseurl}}/assets/SDC-T2/Screenshot from 2018-03-22 22-38-53.png' /></div> 

<div style="text-align:center"><img src ='{{site.baseurl}}/assets/SDC-T2/Screenshot from 2018-03-22 22-44-55.png' /></div> 

<div style="text-align:center"><img src ='{{site.baseurl}}/assets/SDC-T2/Screenshot from 2018-03-22 22-52-33.png' /></div> 

## What to expect from the project

<div style="text-align:center"><iframe width="560" height="315" src="https://www.youtube.com/embed/WAt_g6HgYvs" frameborder="0" allow="autoplay; encrypted-media" allowfullscreen></iframe></div>

<div style="text-align:center"><img src ='{{site.baseurl}}/assets/SDC-T2/Screenshot from 2018-03-22 23-00-14.png' /></div> 

<div style="text-align:center"><img src ='{{site.baseurl}}/assets/SDC-T2/Screenshot from 2018-03-22 23-00-29.png' /></div> 

