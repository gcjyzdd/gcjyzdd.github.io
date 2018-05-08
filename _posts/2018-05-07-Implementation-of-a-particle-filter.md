---
layout: post
date:   2018-05-07 18:43
categories: SDC Localization ParticleFilter
---

<script src="https://cdnjs.cloudflare.com/ajax/libs/mathjax/2.7.0/MathJax.js?config=TeX-AMS-MML_HTMLorMML" type="text/javascript"></script>

## Pseudocode

### Process and Implementation

As an accompaniment to the videos we will follow the particle filter algorithm process and implementation details.

### Particle Filter Algorithm Steps and Inputs

The flowchart below represents the steps of the particle filter algorithm as well as its inputs.

<div style="text-align:center"><img src ='{{site.baseurl}}/assets/SDC-T2/L14_IPF/02-l-pseudocode.00-00-47-13.still006.png' /></div>

### Psuedo Code

This is an outline of steps you will need to take with your code in order to implement a particle filter for localizing an autonomous vehicle. The pseudo code steps correspond to the steps in the algorithm flow chart, initialization, prediction, particle weight updates, and resampling. Python implementation of these steps was covered in the previous lesson.

<div style="text-align:center"><img src ='{{site.baseurl}}/assets/SDC-T2/L14_IPF/02-l-pseudocode.00-00-14-28.still001.png' /></div>

At the initialization step we estimate our position from GPS input. The subsequent steps in the process will refine this estimate to localize our vehicle.

<div style="text-align:center"><img src ='{{site.baseurl}}/assets/SDC-T2/L14_IPF/02-l-pseudocode.00-00-16-01.still002.png' /></div>

During the prediction step we add the control input (yaw rate & velocity) for all particles

<div style="text-align:center"><img src ='{{site.baseurl}}/assets/SDC-T2/L14_IPF/02-l-pseudocode.00-00-30-05.still003.png' /></div>

During the update step, we update our particle weights using map landmark positions and feature measurements.

<div style="text-align:center"><img src ='{{site.baseurl}}/assets/SDC-T2/L14_IPF/02-l-pseudocode.00-00-35-08.still004.png' /></div>

During resampling we will resample M times (M is range of 0 to length_of_particleArray) drawing a particle i (i is the particle index) proportional to its weight . Sebastian covered one implementation of this in his [discussion and implementation of a resampling wheel](https://classroom.udacity.com/nanodegrees/nd013/parts/40f38239-66b6-46ec-ae68-03afd8a601c8/modules/2c318113-724b-4f9f-860c-cb334e6e4ad7/lessons/48704330/concepts/487480820923).

<div style="text-align:center"><img src ='{{site.baseurl}}/assets/SDC-T2/L14_IPF/02-l-pseudocode.00-00-40-01.still005.png' /></div>

The new set of particles represents the Bayes filter posterior probability. We now have a refined estimate of the vehicles position based on input evidence. 

## Initialization

<div style="text-align:center"><img src ='{{site.baseurl}}/assets/SDC-T2/L14_IPF/03-l-initialization.00-01-53-01.still001.png' /></div>

The most practical way to initialize our particles and generate real time output, is to make an initial estimate using GPS input. As with all sensor based operations, this step is impacted by noise.

Project Implementation

  * Particles shall be implemented by sampling a Gaussian distribution, taking into account Gaussian sensor noise around the initial GPS position and heading estimates.
  * Use the [C++ standard library normal distribution](http://en.cppreference.com/w/cpp/numeric/random/normal_distribution) and [C++ standard library random engine](http://www.cplusplus.com/reference/random/default_random_engine) functions to sample positions around GPS measurements.

## Program Gaussian Sampling

### Coding Instructions

I have provided you with a function that takes a GPS position and initial heading as input. I want you to print out to the terminal 3 samples from a normal distribution with mean equal to the GPS position and initial heading measurements and standard deviation of 2 m for the x and y position and 0.05 radians for the heading of the car.

Fill out the "TODO" sections in the code.

```cpp
/*
 * print_samples.cpp
 *
 * Print out to the terminal 3 samples from a normal distribution with
 * mean equal to the GPS position and IMU heading measurements and
 * standard deviation of 2 m for the x and y position and 0.05 radians
 * for the heading of the car. 
 *
 * Author: Tiffany Huang
 */

#include <random> // Need this for sampling from distributions
#include <iostream>

using namespace std;

// @param gps_x 	GPS provided x position
// @param gps_y 	GPS provided y position
// @param theta		GPS provided yaw
void printSamples(double gps_x, double gps_y, double theta) {
	default_random_engine gen;
	double std_x, std_y, std_theta; // Standard deviations for x, y, and theta

	// TODO: Set standard deviations for x, y, and theta.
	 
	
	
	// This line creates a normal (Gaussian) distribution for x.
	normal_distribution<double> dist_x(gps_x, std_x);
	
	// TODO: Create normal distributions for y and theta.
	
	

	for (int i = 0; i < 3; ++i) {
		double sample_x, sample_y, sample_theta;
		
		// TODO: Sample  and from these normal distrubtions like this: 
		// sample_x = dist_x(gen);
		// where "gen" is the random engine initialized earlier.

		
		
		// Print your samples to the terminal.
		cout << "Sample " << i + 1 << " " << sample_x << " " << sample_y << " " << sample_theta << endl;
	}

}

int main() {
	
	// Set GPS provided state of the car.
	double gps_x = 4983;
	double gps_y = 5029;
	double theta = 1.201;
	
	// Sample from the GPS provided position.
	printSamples(gps_x, gps_y, theta);
	
	return 0;
}
```

## Prediction Step

<div style="text-align:center"><img src ='{{site.baseurl}}/assets/SDC-T2/L14_IPF/05-l-predictionstep.00-00-38-28.still001.png' /></div>

Now that we have initialized our particles it's time to predict the vehicle's position. Here we will use what we learned in the motion models lesson to predict where the vehicle will be at the next time step, by updating based on yaw rate and velocity, while accounting for Gaussian sensor noise.

## Calculate Prediction Step: Quiz

<div style="text-align:center"><img src ='{{site.baseurl}}/assets/SDC-T2/L14_IPF/Screenshot from 2018-05-07 19-35-11.png' /></div>

<div style="text-align:center"><img src ='{{site.baseurl}}/assets/SDC-T2/L14_IPF/Screenshot from 2018-05-07 19-36-21.png' /></div>

## Data Association: Nearest Neighbor

<div style="text-align:center"><img src ='{{site.baseurl}}/assets/SDC-T2/L14_IPF/Screenshot from 2018-05-07 19-40-20.png' /></div>

<div style="text-align:center"><img src ='{{site.baseurl}}/assets/SDC-T2/L14_IPF/Screenshot from 2018-05-07 19-47-32.png' /></div>

<div style="text-align:center"><img src ='{{site.baseurl}}/assets/SDC-T2/L14_IPF/Screenshot from 2018-05-07 19-50-04.png' /></div>


<div style="text-align:center"><img src ='{{site.baseurl}}/assets/SDC-T2/L14_IPF/Screenshot from 2018-05-07 19-50-57.png' /></div>


<div style="text-align:center"><img src ='{{site.baseurl}}/assets/SDC-T2/L14_IPF/Screenshot from 2018-05-07 19-51-45.png' /></div>


<div style="text-align:center"><img src ='{{site.baseurl}}/assets/SDC-T2/L14_IPF/Screenshot from 2018-05-07 19-52-37.png' /></div>


<div style="text-align:center"><img src ='{{site.baseurl}}/assets/SDC-T2/L14_IPF/Screenshot from 2018-05-07 19-53-31.png' /></div>

## Update Step

Note that the x and y errors are depicted from the point of view of the map (x is horizontal, y is vertical) rather than the point of view of the car where x is in the direction of the car’s heading,( i.e. It points to where the car is facing), and y is orthogonal (90 degrees) to the left of the x-axis (pointing out of the left side of the car).

<div style="text-align:center"><img src ='{{site.baseurl}}/assets/SDC-T2/L14_IPF/07-l-data-association-nearest-neighbor.00-00-17-03.still003.png' /></div>


<div style="text-align:center"><img src ='{{site.baseurl}}/assets/SDC-T2/L14_IPF/09-l-update-step.00-00-17-03.still001.png' /></div>

<div style="text-align:center"><img src ='{{site.baseurl}}/assets/SDC-T2/L14_IPF/Screenshot from 2018-05-07 19-59-16.png' /></div>

<div style="text-align:center"><img src ='{{site.baseurl}}/assets/SDC-T2/L14_IPF/Screenshot from 2018-05-07 20-01-19.png' /></div>

<div style="text-align:center"><img src ='{{site.baseurl}}/assets/SDC-T2/L14_IPF/Screenshot from 2018-05-07 20-02-25.png' /></div>

Now that we have incorporated velocity and yaw rate measurement inputs into our filter, we must update particle weights based on LIDAR and RADAR readings of landmarks. We will practice calculating particle weights, later in this lesson, with the Particle Weights Quiz.

## Calculating Error

<div style="text-align:center"><img src ='{{site.baseurl}}/assets/SDC-T2/L14_IPF/Screenshot from 2018-05-07 20-07-17.png' /></div>

## Transformations and Associations

In the project you will need to correctly perform observation measurement transformations, along with identifying measurement landmark associations in order to correctly calculate each particle's weight. Remember, our ultimate goal is to find a weight parameter for each particle that represents how well that particle fits to being in the same location as the actual car.

In the quizzes that follow we will be given a single particle with its position and heading along with the car's observation measurements. We will first need to transform the car's measurements from its local car coordinate system to the map's coordinate system. Next, each measurement will need to be [associated with a landmark identifier](https://classroom.udacity.com/nanodegrees/nd013/parts/40f38239-66b6-46ec-ae68-03afd8a601c8/modules/2c318113-724b-4f9f-860c-cb334e6e4ad7/lessons/5c50790c-5370-4c80-aff6-334659d5c0d9/concepts/44dc964a-7cff-4b31-b0b2-94b90d68b96b), for this part we will take the closest landmark to each transformed observation. Finally, we will use this information to calculate the weight value of the particle.

<div style="text-align:center"><img src ='{{site.baseurl}}/assets/SDC-T2/L14_IPF/localization-map-concept-copy.png' /></div>

In the graph above we have a car (**ground truth position**) that observes three nearby landmarks, each one labeled OBS1, OBS2, OBS3. Each observation measurement has x, and y values in the car's coordinate system. We have a particle "P" (**estimated position of the car**) above with position (4,5) on the map with heading -90 degrees. The first task is to transform each observation marker from the vehicle's coordinates to the map's coordinates, with respect to our particle.

## Converting Landmark Observations

Here is another example that might help your intuition.

Referring to the figures below:

Suppose the map coordinate system (grey lines) and the vehicle coordinate system (orange lines) are offset, as depicted below. If we know the location of the observation in vehicle coordinates (grey lines), we would need to rotate the entire system, observation included, -45 degrees to find it in map coordinates (grey lines), Once this rotation is done, we can easily see the location of the observation in map coordinates.

Particle (blue dot) in Map Frame (grey)
<div style="text-align:center"><img src ='{{site.baseurl}}/assets/SDC-T2/L14_IPF/45deg-1.png' /></div>

Particle (blue dot) in Vehicle Frame (orange)
<div style="text-align:center"><img src ='{{site.baseurl}}/assets/SDC-T2/L14_IPF/45deg-2.png' /></div>

## Particle Weights

Now we that we have done the measurement transformations and associations, we have all the pieces we need to calculate the particle's final weight. The particles final weight will be calculated as the product of each measurement's Multivariate-Gaussian probability density.

The Multivariate-Gaussian probability density has two dimensions, x and y. The mean of the Multivariate-Gaussian is the measurement's associated landmark position and the Multivariate-Gaussian's standard deviation is described by our initial uncertainty in the x and y ranges. The Multivariate-Gaussian is evaluated at the point of the transformed measurement's position. The formula for the Multivariate-Gaussian can be seen below.

$$P(x,y) =\frac{1}{2\pi \sigma_{x}\sigma_{y}}e^{-(\frac{(x-\mu_x)^2}{2\sigma_x^2}+\frac{(y-\mu_y)^2}{2\sigma_y^2})}$$

To complete the next set of quizzes, calculate each measurement's Multivariate-Gaussian probability density using the formula above and the previously calculated values. In this example the standard deviation for both x and y is 0.3.

Note that x and y are the observations in map coordinates from the landmarks quiz and $$\mu_x, \mu_y$$​ are the coordinates of the nearest landmarks. These should correspond to the correct responses from previous quizzes.
