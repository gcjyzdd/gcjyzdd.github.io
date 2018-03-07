---
layout: post
date:   2018-03-06 21:00
categories: SDC KalmanFilter DataFusion
title: Lidar and Radar Fusion with Kalman Filters
---

<script src="https://cdnjs.cloudflare.com/ajax/libs/mathjax/2.7.0/MathJax.js?config=TeX-AMS-MML_HTMLorMML" type="text/javascript"></script>


## Lesson Map and Fusion Flow

<div style="text-align:center"><img src ='{{site.baseurl}}/assets/SDC-T2/Screenshot from 2018-03-06 20-53-58.png' /></div>

Imagine you are in a car equipped with sensors on the outside. The car sensors can detect objects moving around: for example, the sensors might detect a pedestrian, as described in the video, or even a bicycle. For variety, let's step through the Kalman Filter algorithm using the bicycle example.

The Kalman Filter algorithm will go through the following steps:

* first measurement - the filter will receive initial measurements of the bicycle's position relative to the car. These measurements will come from a radar or lidar sensor.
* initialize state and covariance matrices - the filter will initialize the bicycle's position based on the first measurement.
* then the car will receive another sensor measurement after a time period $$\Delta{t}$$.
* predict - the algorithm will predict where the bicycle will be after time $$\Delta{t}$$. One basic way to predict the bicycle location after $$\Delta{t}$$ is to assume the bicycle's velocity is constant; thus the bicycle will have moved velocity times $$\Delta{t}$$. In the extended Kalman filter lesson, we will assume the velocity is constant; in the unscented Kalman filter lesson, we will introduce a more complex motion model.
* update - the filter compares the "predicted" location with what the sensor measurement says. The predicted location and the measured location are combined to give an updated location. The Kalman filter will put more weight on either the predicted location or the measured location depending on the uncertainty of each value.
* then the car will receive another sensor measurement after a time period $$\Delta{t}$$. The algorithm then does another predict and update step.

## Estimation Problem Refresh

Time updates and measurement updates:

<div style="text-align:center"><img src ='{{site.baseurl}}/assets/SDC-T2/Screenshot from 2018-03-06 21-32-51.png' /></div>

Multiple sensors:

<div style="text-align:center"><img src ='{{site.baseurl}}/assets/SDC-T2/Screenshot from 2018-03-06 21-36-11.png' /></div>

**Definition of Variables**

* $$x$$ is the mean state vector. For an extended Kalman filter, the mean state vector contains information about the object's position and velocity that you are tracking. It is called the "mean" state vector because position and velocity are represented by a gaussian distribution with mean xx.

* $$P$$ is the state covariance matrix, which contains information about the uncertainty of the object's position and velocity. You can think of it as containing standard deviations.

* $$k$$ represents time steps. So $$x_k$$ refers to the object's position and velocity vector at time $$k$$.

* The notation $$k+1 \| k$$ refers to the prediction step. At time $$k+1$$, you receive a sensor measurement. Before taking into account the sensor measurement to update your belief about the object's position and velocity, you predict where you think the object will be at time $$k+1$$. You can predict the position of the object at $$k+1$$ based on its position and velocity at time $$k$$. Hence $$x_{k+1\|k}$$ means that you have predicted where the object will be at $$k+1$$ but have not yet taken the sensor measurement into account.

* $$x_{k+1}$$ means that you have now predicted where the object will be at time $$k+1$$ and then used the sensor measurement to update the object's position and velocity.

Time flow plot:

<div style="text-align:center"><img src ='{{site.baseurl}}/assets/SDC-T2/Screenshot from 2018-03-06 21-39-12.png' /></div>


<div style="text-align:center"><img width='100%' src ='{{site.baseurl}}/assets/SDC-T2/Screenshot from 2018-03-06 22-21-34.png' /></div>

## Kalman Filter Intuition

<div style="text-align:center"><img src ='{{site.baseurl}}/assets/SDC-T2/measureupdatequizpost.png' /></div>

Because we have already run a prediction-update iteration with the first sensor at time k+3, the output of the second prediction at time k+3 will actually be identical to the output from the update step with the first sensor. So, in theory, you could skip the second prediction step and just run a prediction, update, update iteration.

### Kalman Filter Intuition

The Kalman equation contains many variables, so here is a high level overview to get some intuition about what the Kalman filter is doing.

**Prediction**

Let's say we know an object's current position and velocity , which we keep in the $$x$$ variable. Now one second has passed. We can predict where the object will be one second later because we knew the object position and velocity one second ago; we'll just assume the object kept going at the same velocity.

The $$x' = Fx + \nu$$ equation does these prediction calculations for us.

But maybe the object didn't maintain the exact same velocity. Maybe the object changed direction, accelerated or decelerated. So when we predict the position one second later, our uncertainty increases. $$P' = FPF^T + Q$$ represents this increase in uncertainty.

Process noise refers to the uncertainty in the prediction step. We assume the object travels at a constant velocity, but in reality, the object might accelerate or decelerate. The notation $$\nu \sim N(0, Q)$$ defines the process noise as a gaussian distribution with mean zero and covariance $$Q$$.

**Update**

Now we get some sensor information that tells where the object is relative to the car. First we compare where we think we are with what the sensor data tells us $$y = z - Hx'$$.

The $$K$$ matrix, often called the Kalman filter gain, combines the uncertainty of where we think we are $$P'$$ with the uncertainty of our sensor measurement $$R$$. If our sensor measurements are very uncertain (R is high relative to P'), then the Kalman filter will give more weight to where we think we are: $$x'$$. If where we think we are is uncertain ($$P'$$ is high relative to $$R$$), the Kalman filter will put more weight on the sensor measurement: $$z$$.

Measurement noise refers to uncertainty in sensor measurements. The notation $$\omega \sim N(0, R)$$ defines the measurement noise as a gaussian distribution with mean zero and covariance $$R$$. Measurement noise comes from uncertainty in sensor measurements.

### A Note About the State Transition Function: Bu

If you go back to the video, you'll notice that the state transition function was first given as 

$$x' = Fx + Bu + \nu$$.

But then $$Bu$$ was crossed out leaving $$x' = Fx + \nu$$ 

$$B$$ is a matrix called the control input matrix and uu is the control vector.

As an example, let's say we were tracking a car and we knew for certain how much the car's motor was going to accelerate or decelerate over time; in other words, we had an equation to model the exact amount of acceleration at any given moment. $$Bu$$ would represent the updated position of the car due to the internal force of the motor. We would use $$\nu$$ to represent any random noise that we could not precisely predict like if the car slipped on the road or a strong wind moved the car.

For the Kalman filter lessons, we will assume that there is no way to measure or know the exact acceleration of a tracked object. For example, if we were in an autonomous vehicle tracking a bicycle, pedestrian or another car, we would not be able to model the internal forces of the other object; hence, we do not know for certain what the other object's acceleration is. Instead, we will set $$Bu = 0$$ and represent acceleration as a random noise with mean $$\nu$$.

Kalman filter applied to a 1D example:

<div style="text-align:center"><img width='100%' src ='{{site.baseurl}}/assets/SDC-T2/Screenshot from 2018-03-06 22-27-10.png' /></div>

The model of the walking pedestrian:

<div style="text-align:center"><img width='100%' src ='{{site.baseurl}}/assets/SDC-T2/Screenshot from 2018-03-06 22-28-18.png' /></div>

Time updates and measurement updates:

<div style="text-align:center"><img width='100%' src ='{{site.baseurl}}/assets/SDC-T2/Screenshot from 2018-03-06 22-28-47.png' /></div>

## Kalman Filter Equations in C++ Part I

Now, let's do a quick refresher of the Kalman Filter for a simple 1D motion case. Let's say that your goal is to track a pedestrian with state xx that is described by a position and velocity.

$$x = \begin{pmatrix} p \\ v \end{pmatrix} $$

### Prediction Step

When designing the Kalman filter, we have to define the two linear functions: the state transition function and the measurement function. The state transition function is

$$x' = F*x + noise$$,

where,

$$F = \begin{pmatrix} 1 & \Delta t \\ 0 & 1 \end{pmatrix}$$

and $$x'$$ is where we predict the object to be after time $$\Delta t$$.

$$F$$ is a matrix that, when multiplied with $$x$$, predicts where the object will be after time $$\Delta t$$.

By using the linear motion model with a constant velocity, the new location, $$p'$$ is calculated as

$$p' = p + v * \Delta t$$,

where $$p$$ is the old location and $$v$$, the velocity, will be the same as the new velocity ($$v' = v$$) because the velocity is constant.

We can express this in a matrix form as follows:

$$\begin{pmatrix} p' \\ v' \end{pmatrix} = \begin{pmatrix}1 & \Delta t \\ 0 & 1 \end{pmatrix} \begin{pmatrix} p \\ v \end{pmatrix}$$

Remember we are representing the object location and velocity as gaussian distributions with mean $$x$$. When working with the equation $$x' = F* x + noise$$, we are calculating the mean value of the state vector. The noise is also represented by a gaussian distribution but with mean zero; hence, noise = 0 is saying that the mean noise is zero. The equation then becomes $$x' = F*x$$

But the noise does have uncertainty. The uncertainty shows up in the $$Q$$ matrix as acceleration noise.

### Update Step

For the update step, we use the measurement function to map the state vector into the measurement space of the sensor. To give a concrete example, lidar only measures an object's position. But the extended Kalman filter models an object's position and velocity. So multiplying by the measurement function $$H$$ matrix will drop the velocity information from the state vector $$x$$. Then the lidar measurement position and our belief about the object's position can be compared.


$$z = H*x + w$$

where $$w$$ represents sensor measurement noise.

So for lidar, the measurement function looks like this:

$$z = p'$$.

It also can be represented in a matrix form:

$$z=\begin{pmatrix} 1 & 0 \end{pmatrix} \begin{pmatrix} p' \\ v' \end{pmatrix}$$.

As we already know, the general algorithm is composed of a prediction step where I predict the new state and covariance, $$P$$.

And we also have a measurement update (or also called many times a correction step) where we use the latest measurements to update our estimate and our uncertainty.

Here is the code:

```cpp
// kf.cpp
// Write a function 'filter()' that implements a multi-
// dimensional Kalman Filter for the example given
//============================================================================
#include <iostream>
#include "Dense"
#include <vector>

using namespace std;
using namespace Eigen;

//Kalman Filter variables
VectorXd x;	// object state
MatrixXd P;	// object covariance matrix
VectorXd u;	// external motion
MatrixXd F; // state transition matrix
MatrixXd H;	// measurement matrix
MatrixXd R;	// measurement covariance matrix
MatrixXd I; // Identity matrix
MatrixXd Q;	// process covariance matrix

vector<VectorXd> measurements;
void filter(VectorXd &x, MatrixXd &P);


int main() {
	/**
	 * Code used as example to work with Eigen matrices
	 */
//	//you can create a  vertical vector of two elements with a command like this
//	VectorXd my_vector(2);
//	//you can use the so called comma initializer to set all the coefficients to some values
//	my_vector << 10, 20;
//
//
//	//and you can use the cout command to print out the vector
//	cout << my_vector << endl;
//
//
//	//the matrices can be created in the same way.
//	//For example, This is an initialization of a 2 by 2 matrix
//	//with the values 1, 2, 3, and 4
//	MatrixXd my_matrix(2,2);
//	my_matrix << 1, 2,
//			3, 4;
//	cout << my_matrix << endl;
//
//
//	//you can use the same comma initializer or you can set each matrix value explicitly
//	// For example that's how we can change the matrix elements in the second row
//	my_matrix(1,0) = 11;    //second row, first column
//	my_matrix(1,1) = 12;    //second row, second column
//	cout << my_matrix << endl;
//
//
//	//Also, you can compute the transpose of a matrix with the following command
//	MatrixXd my_matrix_t = my_matrix.transpose();
//	cout << my_matrix_t << endl;
//
//
//	//And here is how you can get the matrix inverse
//	MatrixXd my_matrix_i = my_matrix.inverse();
//	cout << my_matrix_i << endl;
//
//
//	//For multiplying the matrix m with the vector b you can write this in one line as let’s say matrix c equals m times v.
//	//
//	MatrixXd another_matrix;
//	another_matrix = my_matrix*my_vector;
//	cout << another_matrix << endl;


	//design the KF with 1D motion
	x = VectorXd(2);
	x << 0, 0;

	P = MatrixXd(2, 2);
	P << 1000, 0, 0, 1000;

	u = VectorXd(2);
	u << 0, 0;

	F = MatrixXd(2, 2);
	F << 1, 1, 0, 1;

	H = MatrixXd(1, 2);
	H << 1, 0;

	R = MatrixXd(1, 1);
	R << 1;

	I = MatrixXd::Identity(2, 2);

	Q = MatrixXd(2, 2);
	Q << 0, 0, 0, 0;

	//create a list of measurements
	VectorXd single_meas(1);
	single_meas << 1;
	measurements.push_back(single_meas);
	single_meas << 2;
	measurements.push_back(single_meas);
	single_meas << 3;
	measurements.push_back(single_meas);

	//call Kalman filter algorithm
	filter(x, P);

	return 0;

}


void filter(VectorXd &x, MatrixXd &P) {

	for (unsigned int n = 0; n < measurements.size(); ++n) {

		VectorXd z = measurements[n];
		//YOUR CODE HERE
		
		// KF Measurement update step
		VectorXd y = z - H * x;
		MatrixXd S = H * P * H.transpose() + R;
		MatrixXd K = P * H.transpose() * S.inverse();
		P = (I - K * H) * P;

		// new state
		x = x + K * y;
		// KF Prediction step
		x = F * x;
		P = F * P * F.transpose() + Q;

		std::cout << "x=" << std::endl <<  x << std::endl;
		std::cout << "P=" << std::endl <<  P << std::endl;


	}
}
```

To compile the code:

```bash
g++ -I ./Eigen/ kf.cpp -o main
```

Or create a Makefile:

```
# Makefile
all:
    g++ -I ./Eigen/ kf.cpp -o main
```

Compile the Makefile and run the generated executable file:

```bash
make && ./main
```

## State Prediction

Now consider a 2D pedestrian and we get the following model:

<div style="text-align:center"><img width='100%' src ='{{site.baseurl}}/assets/SDC-T2/Screenshot from 2018-03-07 20-56-28.png' /></div>

And variable sample time:

<div style="text-align:center"><img width='100%' src ='{{site.baseurl}}/assets/SDC-T2/Screenshot from 2018-03-07 20-57-08.png' /></div>

Quiz:

<div style="text-align:center"><img width='100%' src ='{{site.baseurl}}/assets/SDC-T2/Screenshot from 2018-03-07 21-07-13.png' /></div>

## Process Covariance Matrix

Roadmap of this lesson:

<div style="text-align:center"><img width='100%' src ='{{site.baseurl}}/assets/SDC-T2/Screenshot from 2018-03-07 21-09-00.png' /></div>

Let's look at the covariance of process noise:

<div style="text-align:center"><img width='100%' src ='{{site.baseurl}}/assets/SDC-T2/Screenshot from 2018-03-07 21-10-58.png' /></div>

We can seperate sample time and covariance of acceleration:

<div style="text-align:center"><img width='100%' src ='{{site.baseurl}}/assets/SDC-T2/Screenshot from 2018-03-07 21-13-20.png' /></div>

Rephrase in this way:

<div style="text-align:center"><img width='100%' src ='{{site.baseurl}}/assets/SDC-T2/Screenshot from 2018-03-07 21-14-05.png' /></div>

and get $$Q_{\nu}$$:

<div style="text-align:center"><img width='100%' src ='{{site.baseurl}}/assets/SDC-T2/Screenshot from 2018-03-07 21-15-04.png' /></div>

Finallly:

<div style="text-align:center"><img width='100%' src ='{{site.baseurl}}/assets/SDC-T2/Screenshot from 2018-03-07 21-15-35.png' /></div>

**Note on Notation**

Some authors describe $$Q$$ as the complete process noise covariance matrix. And some authors describe $$Q$$ as the covariance matrix of the individual noise processes. In our case, the covariance matrix of the individual noise processes matrix is called $$Q_\nu$$, which is something to be aware of.

## Lidar Measurements

<div style="text-align:center"><img width='100%' src ='{{site.baseurl}}/assets/SDC-T2/Screenshot from 2018-03-07 21-26-59.png' /></div>


## Measurement Noise Covariance Matrix R continued

For laser sensors, we have a 2D measurement vector. Each location component $$p_x, p_y$$ are affected by a random noise. So our noise vector $$\omega$$ has the same dimension as $$z$$. And it is a distribution with zero mean and a 2 x 2 covariance matrix which comes from the product of the vertical vector $$\omega$$ and its transpose.

$$R = E[\omega \omega^T] = \begin{pmatrix} \sigma^2_{px} & 0 \\ 0 & \sigma^2_{py} \end{pmatrix}$$

where $$R$$ is the measurement noise covariance matrix; in other words, the matrix $$R$$ represents the uncertainty in the position measurements we receive from the laser sensor.

Generally, the parameters for the random noise measurement matrix will be provided by the sensor manufacturer. For the extended Kalman filter project, we have provided $$R$$ matrices values for both the radar sensor and the lidar sensor.

Remember that the off-diagonal $$0$$s in $$R$$ indicate that the noise processes are uncorrelated.

You have all you need for laser-only tracking! Now, I want you to apply what you've learned in a programming assignment.

## Radar Measurements


<div style="text-align:center"><img width='100%' src ='{{site.baseurl}}/assets/SDC-T2/Screenshot from 2018-03-07 23-35-11.png' /></div>

Use radar sensor to enhance the performance of Kalman filter because Lidar sensor cannot measure velocity directly while radar can. But radar has a relatively low resolution. It's a good idea to combine lidar and radar sensors.

<div style="text-align:center"><img width='100%' src ='{{site.baseurl}}/assets/SDC-T2/Screenshot from 2018-03-07 23-38-52.png' /></div>

Let's build the model with radar:

<div style="text-align:center"><img width='100%' src ='{{site.baseurl}}/assets/SDC-T2/Screenshot from 2018-03-07 23-40-24.png' /></div>

And radar sensors measure objects in a polar coordinate system :

<div style="text-align:center"><img width='100%' src ='{{site.baseurl}}/assets/SDC-T2/Screenshot from 2018-03-07 23-42-18.png' /></div>

As a result, it has a different measurement function:

<div style="text-align:center"><img width='100%' src ='{{site.baseurl}}/assets/SDC-T2/Screenshot from 2018-03-07 23-43-37.png' /></div>

which is nonlinear:

<div style="text-align:center"><img width='100%' src ='{{site.baseurl}}/assets/SDC-T2/Screenshot from 2018-03-07 23-44-21.png' /></div>

<div style="text-align:center"><img width='100%' src ='{{site.baseurl}}/assets/SDC-T2/Screenshot from 2018-03-07 23-48-48.png' /></div>

<div style="text-align:center"><img width='100%' src ='{{site.baseurl}}/assets/SDC-T2/Screenshot from 2018-03-07 23-49-15.png' /></div>

<div style="text-align:center"><img width='100%' src ='{{site.baseurl}}/assets/SDC-T2/Screenshot from 2018-03-07 23-49-30.png' /></div>

<div style="text-align:center"><img width='100%' src ='{{site.baseurl}}/assets/SDC-T2/Screenshot from 2018-03-07 23-49-43.png' /></div>



