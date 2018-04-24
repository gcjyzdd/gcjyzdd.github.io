---
layout: post
date:   2018-04-21 22:43
categories: SDC Localization
title: Markov Localization
---

<script src="https://cdnjs.cloudflare.com/ajax/libs/mathjax/2.7.0/MathJax.js?config=TeX-AMS-MML_HTMLorMML" type="text/javascript"></script>

## Overview

### Markov Localization and the Kidnapped Vehicle Project

The localization module culminates in the Kidnapped Vehicle Project. In that project our vehicle has been kidnapped and placed in an unknown location. We must leverage our knowledge of localization to determine where our vehicle is. The Kidnapped Vehicle Project relies heavily on the particle filter approach to localization, particularly "Implementation of a Particle Filter," an upcoming lesson. This leaves the question; How does Markov Localization relate to the Kidnapped Vehicle project?

Markov Localization or Bayes Filter for Localization is a generalized filter for localization and all other localization approaches are realizations of this approach, as discussed [here](https://classroom.udacity.com/nanodegrees/nd013/parts/40f38239-66b6-46ec-ae68-03afd8a601c8/modules/2c318113-724b-4f9f-860c-cb334e6e4ad7/lessons/47f9b7a1-317f-4fab-88d3-bb3ce215d575/concepts/22bc5a5c-4c44-453f-9a17-ab904e351fe4). By learning how to derive and implement (coding exercises) this filter we develop intuition and methods that will help us solve any vehicle localization task, including implementation of a particle filter. We don't know exactly where our vehicle is at any given time, but can approximate it's location. As such, we generally think of our vehicle location as a probability distribution, each time we move, our distribution becomes more diffuse (wider). We pass our variables (map data, observation data, and control data) into the filter to concentrate (narrow) this distribution, at each time step. Each state prior to applying the filter represents our prior and the narrowed distribution represents our Bayes' posterior.

<div style="text-align:center"><img src ='{{site.baseurl}}/assets/SDC-T2/Screenshot from 2018-04-21 22-49-29.png' /></div>

## Formal Definition of Variables

<div style="text-align:center"><img src ='{{site.baseurl}}/assets/SDC-T2/Screenshot from 2018-04-21 22-51-51.png' /></div>

## Localization Posterior Explanation

<div style="text-align:center"><img src ='{{site.baseurl}}/assets/SDC-T2/Screenshot from 2018-04-21 22-53-11.png' /></div>

## Bayes' Rule

Before we dive into deeper into Markov localization, we should review Bayes' Rule. This will serve as a refresher for those familiar with Bayesian methods and we provide some additional resources for those less familiar.

Recall that Bayes' Rule enables us to determine the conditional probability of a state given evidence P(a\|b) by relating it to the conditional probability of the evidence given the state (P(b\|a) in the form of:

$$P(a)\times P(b\|a) = P(b)\times P(a\|b)$$

which can be rearranged to:

$$P(a\|b) = \frac{P(b\|a) \, P(a)}{P(b)}$$​	 

In other words the probability of state a, given evidence b, is the probability of evidence b, given state a, multiplied by the probability of state a, normalized by the total probability of b over all states.

Let's move on to an example to illustrate the utility of Bayes' Rule.

### Bayes' Rule Applied

Let's say we have two bags of marbles, bag 1 and bag 2, filled with two types of marbles, red and blue. Bag 1 contains 10 blue marbles and 30 red marbles, whereas bag 2 contains 20 of each color marble.

If a friend were to choose a bag at random and then a marble at random, from that bag, how can we determine the probability that that marble came from a specific bag? If you guessed Bayes' Rule, you are definitely paying attention.

In this scenario, our friend produces a red marble, in that case, what is the probability that the marble came from bag 1? Rewriting this in terms of Bayes' Rule, our solution becomes:

$$P(Bag1\|Red) = \frac{P(Red\|Bag1) \, P(Bag1)}{P(Red)}$$​	 

**Bayesian Methods Resources**
* [Sebastian Discusses Bayes Rule](https://classroom.udacity.com/nanodegrees/nd013/parts/40f38239-66b6-46ec-ae68-03afd8a601c8/modules/2c318113-724b-4f9f-860c-cb334e6e4ad7/lessons/48739381/concepts/487221690923)
* [More Bayes Rule Content from Udacity](https://classroom.udacity.com/courses/st101/lessons/48703346/concepts/483698470923)
* [Bayes Rule with Ratios](https://betterexplained.com/articles/understanding-bayes-theorem-with-ratios)
* [A Deep Dive into Bayesian Methods, for Programmers](http://greenteapress.com/wp/think-bayes/)

## Bayes' Filter For Localization

We can apply Bayes' Rule to vehicle localization by passing variables through Bayes' Rule for each time step, as our vehicle moves. This is known as a Bayes' Filter for Localization. We will cover the specific as the lesson continues, but the generalized form Bayes' Filter for Localization is shown below. You may recognize this as being similar to a Kalman filter. In fact, many localization filters, including the Kalman filter are special cases of Bayes' Filter.

Remember the general form for Bayes' Rule:

$$P(a|b) = \frac{P(b|a) \, P(a)}{P(b)}$$

With respect to localization, these terms are:

* $$P(location\|observation)$$: This is P(a\|b), the normalized probability of a position given an observation (posterior).
* $$P(observation\|location)$$: This is P(b\|a), the probability of an observation given a position (likelihood)
* $$P(location)$$: This is P(a), the prior probability of a position
* $$P(observation)$$: This is P(b), the total probability of an observation

Without going into detail yet, be aware that $$P(location)$$ is determined by the motion model. The probability returned by the motion model is the product of the transition model probability (the probability of moving from $$x_{t-1}$$ --> $$x_t$$  and the probability of the state $$x_{t-1}$$​	 .

Over the course of this lesson, you’ll build your own Bayes’ filter. In the next few quizzes, you’ll write code to:

1. Compute Bayes’ rule
2. Calculate Bayes' posterior for localization
3. Initialize a prior belief state
4. Create a function to initialize a prior belief state given landmarks and assumptions

## Calculate Localization Posterior

To continue developing our intuition for this filter and prepare for later coding exercises, let's walk through calculations for determining posterior probabilities at several pseudo positions x, for a single time step. We will start with a time step after the filter has already been initialized and run a few times. We will cover initialization of the filter in an upcoming concept.

<div style="text-align:center"><img src ='{{site.baseurl}}/assets/SDC-T2/Screenshot from 2018-04-23 21-31-14.png' /></div>

<div style="text-align:center"><img src ='{{site.baseurl}}/assets/SDC-T2/Screenshot from 2018-04-23 21-32-43.png' /></div>

<div style="text-align:center"><img src ='{{site.baseurl}}/assets/SDC-T2/Screenshot from 2018-04-23 21-32-52.png' /></div>

## Initialize Belief State

To help develop an intuition for this filter and prepare for later coding exercises, let's walk through the process of initializing our prior belief state. That is, what values should our initial belief state take for each possible position? Let's say we have a 1D map extending from 0 to 25 meters. We have landmarks at x = 5.0, 10.0, and 20.0 meters, with position standard deviation of 1.0 meter. If we know that our car's initial position is at one of these three landmarks, how should we define our initial belief state?

Since we know that we are parked next to a landmark, we can set our probability of being next to a landmark as 1.0. Accounting for a positon precision of +/- 1.0 meters, this places our car at an initial position in the range 4 - 6 (5 +/- 1), 9 - 11 (10 +/- 1), or 19 - 21 (20 +/- 1). All other positions, not within 1.0 meter of a landmark, are initialized to 0. We normalize these values to a total probability of 1.0 by dividing by the total number of positions that are potentially occupied. In this case, that is 9 positions, 3 for each landmark (the landmark position and one position on either side). This gives us a value of 1.11E-01 for positions +/- 1 from our landmarks (1.0/9). So, our initial belief state is:

```
{0, 0, 0, 1.11E-01, 1.11E-01, 1.11E-01, 0, 0, 1.11E-01, 1.11E-01, 1.11E-01, 0, 0, 0, 0, 0, 0, 0, 1.11E-01, 1.11E-01, 1.11E-01, 0, 0, 0, 0}
```

To reinforce this concept, let's practice with a quiz.

* map size: 100 meters
* landmark positions: {8, 15, 30, 70, 80}
* position standard deviation: 2 meters

Assuming we are parked next to a landmark, answer the following questions about our initial belief state.

<div style="text-align:center"><img src ='{{site.baseurl}}/assets/SDC-T2/Screenshot from 2018-04-23 21-42-39.png' /></div>

## Initialize Priors Function

In this quiz we will create a function that initializes priors (initial belief state for each position on the map) given landmark positions, a position standard deviation (+/- 1.0), and the assumption that our car is parked next to a landmark.

Note that the control standard deviation represents the spread from movement (movement is the result of our control input in this case). We input a control of moving 1 step but our actual movement could be in the range of 1 +/- control standard deviation. The position standard deviation is the spread in our actual position. For example, we may believe start at a particular location, but we could be anywhere in that location +/- our position standard deviation.

```cpp
#include <iostream>
#include <algorithm>
#include <vector>

using namespace std;

//initialize priors assumimg vehicle at landmark +/- 1.0 meters position stdev
std::vector<float> initialize_priors(int map_size, std::vector<float> landmark_positions,
                                     float control_stdev);

int main() {

    //set standard deviation of position:
    float control_stdev = 1.0f;


    //set map horizon distance in meters 
    int map_size = 25;

    //initialize landmarks
    std::vector<float> landmark_positions {5, 10, 20};

    // initialize priors
    std::vector<float> priors = initialize_priors(map_size, landmark_positions,
                                                  control_stdev);
    
    //print values to stdout 
    for (unsigned int p = 0; p < priors.size(); p++) {
        std::cout << priors[p] << endl;
    }
        
    return 0;

};

//TODO: Complete the initialize_priors function
std::vector<float> initialize_priors(int map_size, std::vector<float> landmark_positions,
                                     float control_stdev) {

//initialize priors assumimg vehicle at landmark +/- 1.0 meters position stdev

    //YOUR CODE HERE
  std::vector<float>	priors(map_size,0);
  float p = 1./(landmark_positions.size() * (1+2*control_stdev));
    for(int i=0;i<map_size;i++)
	{
	  for(size_t j=0;j<landmark_positions.size();j++)
	  {
		if(abs((float)i-landmark_positions[j]) <= control_stdev)
		{
		  priors[i] = p;
		}		
	  }
	}
    return priors;
}
```

<div style="text-align:center"><img src ='{{site.baseurl}}/assets/SDC-T2/Screenshot from 2018-04-23 22-09-05.png' /></div>

<div style="text-align:center"><img src ='{{site.baseurl}}/assets/SDC-T2/Screenshot from 2018-04-23 22-16-03.png' /></div>

## Derivation Outline

<div style="text-align:center"><img src ='{{site.baseurl}}/assets/SDC-T2/Screenshot from 2018-04-23 22-18-59.png' /></div>


## Apply Bayes Rule with Additional Conditions

<div style="text-align:center"><img src ='{{site.baseurl}}/assets/SDC-T2/06-l-apply-bayes-rule-with-additional-conditions.00-01-30-28.still002.png' /></div>

We aim to estimate state beliefs $$bel(x_t)$$ without the need to carry our entire observation history. We will accomplish this by manipulating our posterior $$p(x_t\|z_{1:t-1},\mu_{1:t},m)$$, obtaining a recursive state estimator. For this to work, we must demonstrate that our current belief $$bel(x_t)$$ can be expressed by the belief one step earlier $$bel(x_{t-1})$$, then use new data to update only the current belief. This recursive filter is known as the Bayes Localization filter or Markov Localization, and enables us to avoid carrying historical observation and motion data. We will achieve this recursive state estimator using Bayes Rule, the Law of Total Probability, and the Markov Assumption.

<div style="text-align:center"><img src ='{{site.baseurl}}/assets/SDC-T2/06-l-apply-bayes-rule-with-additional-conditions.00-01-48-09.still003.png' /></div>

We take the first step towards our recursive structure by splitting our observation vector $$z_{1:t}$$ into current observations $$z_t$$ and previous information $$z_{1:t-1}$$. 

The posterior can then be rewritten as $$p(x_t\|z_t,z_{1:t-1},u_{1:t}, m)$$.

<div style="text-align:center"><img src ='{{site.baseurl}}/assets/SDC-T2/06-l-apply-bayes-rule-with-additional-conditions.00-02-12-10.still004.png' /></div>

Now, we apply Bayes' rule, with an additional challenge, the presence of multiple distributions on the right side (likelihood, prior, normalizing constant). How can we best handle multiple conditions within Bayes Rule? As a hint, we can use substitution, where $$x_t$$ is a, and the observation vector at time t, is b. Don’t forget to include $$u$$ and $$m$$ as well.

<div style="text-align:center"><img src ='{{site.baseurl}}/assets/SDC-T2/Screenshot from 2018-04-23 22-42-51.png' /></div>

## Bayes Rule and Law of Total Probability

<div style="text-align:center"><img src ='{{site.baseurl}}/assets/SDC-T2/Screenshot from 2018-04-23 23-31-40.png' /></div>

<div style="text-align:center"><img src ='{{site.baseurl}}/assets/SDC-T2/Screenshot from 2018-04-23 23-31-30.png' /></div>

## Total Probability and Markov Assumption

<div style="text-align:center"><img src ='{{site.baseurl}}/assets/SDC-T2/Screenshot from 2018-04-23 23-34-40.png' /></div>

## Markov Assumption for Motion Model: Quiz

<div style="text-align:center"><img src ='{{site.baseurl}}/assets/SDC-T2/Screenshot from 2018-04-23 23-40-54.png' /></div>

## Markov Assumption for Motion Model: Explanation

<div style="text-align:center"><img src ='{{site.baseurl}}/assets/SDC-T2/Screenshot from 2018-04-23 23-41-59.png' /></div>

<div style="text-align:center"><img src ='{{site.baseurl}}/assets/SDC-T2/Screenshot from 2018-04-23 23-43-47.png' /></div>

<div style="text-align:center"><img src ='{{site.baseurl}}/assets/SDC-T2/Screenshot from 2018-04-23 23-44-26.png' /></div>

<div style="text-align:center"><img src ='{{site.baseurl}}/assets/SDC-T2/Screenshot from 2018-04-23 23-44-45.png' /></div>

<div style="text-align:center"><img src ='{{site.baseurl}}/assets/SDC-T2/Screenshot from 2018-04-23 23-45-55.png' /></div>

<div style="text-align:center"><img src ='{{site.baseurl}}/assets/SDC-T2/Screenshot from 2018-04-23 23-46-28.png' /></div>

<div style="text-align:center"><img src ='{{site.baseurl}}/assets/SDC-T2/Screenshot from 2018-04-23 23-47-33.png' /></div>

## After Applying Markov Assumption: Quiz

<div style="text-align:center"><img src ='{{site.baseurl}}/assets/SDC-T2/Screenshot from 2018-04-23 23-55-55.png' /></div>


## Recursive Structure

<div style="text-align:center"><img src ='{{site.baseurl}}/assets/SDC-T2/recursive_structure.png' /></div>

We have achieved a very important step towards the final form of our recursive state estimator. Let’s see why. If we rewrite the second term in our integral to split $$z_{1-t}$$ to $$z_{t-1}$$ and $$z_{t-2}$$ we arrive at a function that is exactly the belief from the previous time step, namely $$bel(x_{t-1})$$.

<div style="text-align:center"><img src ='{{site.baseurl}}/assets/SDC-T2/14-l-explain-recursive-structure-.00-00-38-09.still002.png' /></div>

Now we can rewrite out integral as the belief of $$x_{t-1}$$.

<div style="text-align:center"><img src ='{{site.baseurl}}/assets/SDC-T2/14-l-explain-recursive-structure-.00-01-05-00.still003.png' /></div>

The amazing thing is that we have a recursive update formula and can now use the estimated state from the previous time step to predict the current state at t. This is a critical step in a recursive Bayesian filter because it renders us independent from the entire observation and control history. So in the graph structure, we will replace the previous state terms (highlighted) with our belief of the state at x at t -1 (next image).

<div style="text-align:center"><img src ='{{site.baseurl}}/assets/SDC-T2/14-l-explain-recursive-structure-.00-01-55-15.still004.png' /></div>

Finally, we replace the integral by a sum over all $$x_i$$ because we have a discrete localization scenario in this case, to get the same formula in Sebastian's lesson for localization. The process of predicting $$x_t$$ with a previous beliefs ($$x_{t-1}$$) and the transition model is technically a convolution. If you take a look to the formula again, it is essential that the belief at $$x_t = 0$$ is initialized with a meaningful assumption. It depends on the localization scenario how you set the belief or in other words, how you initialize your filter. For example, you can use GPS to get a coarse estimate of your location.

<div style="text-align:center"><img src ='{{site.baseurl}}/assets/SDC-T2/14-l-explain-recursive-structure-.00-02-36-09.still005.png' /></div>


Summing up, here is what we have learned so far:

* How to apply the law of total probability by including the new variable $$x_{t-1}$$	 .
* The Markov assumption, which is very important for probabilistic reasoning, and allows us to make recursive state estimation without carrying our entire history of information
* How to derive the recursive filter structure. Next you will implement a motion model in C++ and earn how to initialize our localizer.

## Implementation Details

<div style="text-align:center"><img src ='{{site.baseurl}}/assets/SDC-T2/Screenshot from 2018-04-24 17-50-41.png' /></div>

## Determine Probabilities

To implement these models in code, we need a function to which we can pass model parameters/values and return a probability. Fortunately, we can use a normalized probability density function (PDF). Let's revisit Sebastian's discussion of this topic.

We have implemented this Gaussian Distribution as a C++ function, `normpdf`, and will practice using it at the end of this concept. `normpdf` accepts a value, a parameter, and a standard deviation, returning a probability.

Additional Resources for Gaussian Distributions

* [Udacity's Statistics Course content on PDF](https://classroom.udacity.com/courses/st095/lessons/86217921/concepts/1020887710923)
* [http://mathworld.wolfram.com/NormalDistribution.html](http://mathworld.wolfram.com/NormalDistribution.html)
* [http://stattrek.com/statistics/dictionary.aspx?definition=Probability_density_function](http://stattrek.com/statistics/dictionary.aspx?definition=Probability_density_function)

Let's practice using `normpdf` to determine transition model probabilities. Specifically, we need to determine the probability of moving from $$x_{t-1}$$ --> $$x_t$$. The value entered into `normpdf` will be the distance between these two positions. We will refer to potential values of these positions as pseudo position and pre-pseudo position. For example, if our pseudo position x is 8 and our pre-pseudo position is 5, our sample value will be 3, and our transition will be from x - 3 --> x.

To calculate our transition model probability, pass any difference in distances into `normpdf` along with our control parameter and position standard deviation.

## Motion Model Probabiity I

Now we will practice implementing the motion model to determine P(location) for our Bayesian filter.

Recall that we derived the following recursive structure for the motion model:

$$\int p(x_t\|x_{t-1}, u_t, m)bel(x_{t-1})dx_{t-1}$$​	 

and that we will implement this in the discretized form:

$$\sum\limits_{i} p(x_t\|x_{t-1}^{(i)}, u_t, m)bel(x_{t-1}^{(i)})$$ 

<div style="text-align:center"><img src ='{{site.baseurl}}/assets/SDC-T2/Screenshot from 2018-04-24 20-21-30.png' /></div>

## Motion Model Probability II

<div style="text-align:center"><img src ='{{site.baseurl}}/assets/SDC-T2/Screenshot from 2018-04-24 20-39-25.png' /></div>

<div style="text-align:center"><img src ='{{site.baseurl}}/assets/SDC-T2/Screenshot from 2018-04-24 20-40-24.png' /></div>

## Coding the Motion Model

Now that we have manually calculated each step for determining the motion model probability, we will implement these steps in a function. The starter code below steps through each position x, calls the motion_model function and prints the results to stdout. To complete this exercise fill in the `motion_model` function which will involve:

For each $$x_{t}$$:

* Calculate the transition probability for each potential value $$x_{t-1}$$​	 
* Calculate the discrete motion model probability by multiplying the transition model probability by the belief state (prior) for $$x_{t-1}$$​	 
* Return total probability (sum) of each discrete probability

## Observation Model Introduction

<div style="text-align:center"><img src ='{{site.baseurl}}/assets/SDC-T2/Screenshot from 2018-04-24 21-34-05.png' /></div>

<div style="text-align:center"><img src ='{{site.baseurl}}/assets/SDC-T2/Screenshot from 2018-04-24 21-36-40.png' /></div>

## Markov Assumption for Observation Model

<div style="text-align:center"><img src ='{{site.baseurl}}/assets/SDC-T2/20-i-markov-assumption-for-observation-model-first-try.00-00-22-16.still001.png' /></div>

The Markov assumption can help us simplify the observation model. Recall that the Markov Assumption is that the next state is dependent only upon the preceding states and that preceding state information has already been used in our state estimation. As such, we can ignore terms in our observation model prior to $$x_t$$ since these values have already been accounted for in our current state and assume that t is independent of previous observations and controls.

<div style="text-align:center"><img src ='{{site.baseurl}}/assets/SDC-T2/20-i-markov-assumption-for-observation-model-first-try.00-00-36-11.still002.png' /></div>

With these assumptions we simplify our posterior distribution such that the observations at t are dependent only on x at time t and the map.

<div style="text-align:center"><img src ='{{site.baseurl}}/assets/SDC-T2/20-i-markov-assumption-for-observation-model-first-try.00-01-18-09.still003.png' /></div>

Since $$z_t$$ can be a vector of multiple observations we rewrite our observation model to account for the observation models for each single range measurement. We assume that the noise behavior of the individual range values $$z_t^1$$ to $$z_t^k$$ is independent and that our observations are independent, allowing us to represent the observation model as a product over the individual probability distributions of each single range measurement. Now we must determine how to define the observation model for a single range measurement.

<div style="text-align:center"><img src ='{{site.baseurl}}/assets/SDC-T2/Screenshot from 2018-04-24 21-40-22.png' /></div>

<div style="text-align:center"><img src ='{{site.baseurl}}/assets/SDC-T2/20-i-markov-assumption-for-observation-model-first-try.00-03-23-08.still004.png' /></div>

In general there exists a variety of observation models due to different sensor, sensor specific noise behavior and performance, and map types. For our 1D example we assume that our sensor measures to the n closest objects in the driving direction, which represent the landmarks on our map. We also assume that observation noise can be modeled as a Gaussian with a standard deviation of 1 meter and that our sensor can measure in a range of 0 – 100 meters.

<div style="text-align:center"><img src ='{{site.baseurl}}/assets/SDC-T2/Screenshot from 2018-04-24 21-46-58.png' /></div>

## Finalize the Bayes Localization Filter

<div style="text-align:center"><img src ='{{site.baseurl}}/assets/SDC-T2/21-i-finalize-the-bayes-localization-filter.00-01-00-15.still001.png' /></div>

We have accomplished a lot in this lesson.

* Starting with the generalized form of Bayes Rule we expressed our posterior, the belief of x at t as $$\eta$$ (normalizer) multiplied with the observation model and the motion model.
* We simplified the observation model using the Markov assumption to determine the probability of z at time t, given only x at time t, and the map.
* We expressed the motion model as a recursive state estimator using the Markov assumption and the law of total probability, resulting in a model that includes our belief at t – 1 and our transition model.
* Finally we derived the general Bayes Filter for Localization (Markov Localization) by expressing our belief of x at t as a simplified version of our original posterior expression (top equation), $$\eta$$ multiplied by the simplified observation model and the motion model. Here the motion model is written as $$\hat{bel}$$, a prediction model.

<div style="text-align:center"><img src ='{{site.baseurl}}/assets/SDC-T2/21-i-finalize-the-bayes-localization-filter.00-01-17-24.still002.png' /></div>

The Bayes Localization Filter dependencies can be represented as a graph, by combining our sub-graphs. To estimate the new state x at t we only need to consider the previous belief state, the current observations and controls, and the map.

<div style="text-align:center"><img src ='{{site.baseurl}}/assets/SDC-T2/21-i-finalize-the-bayes-localization-filter.00-01-35-19.still003.png' /></div>

It is a common practice to represent this filter without the belief $$x_t$$ and to remove the map from the motion model. Ultimately we define $$bel(x_t)$$ as the following expression.

## Bayes Filter Theory Summary

<div style="text-align:center"><img src ='{{site.baseurl}}/assets/SDC-T2/22-l-bayes-filter-theory-summary.00-00-22-29.still001.png' /></div>

The image above sums up the core achievements of this lesson.

* The Bayes Localization Filter Markov Localization is a general framework for recursive state estimation.
* That means this framework allows us to use the previous state (state at t-1) to estimate a new state (state at t) using only current observations and controls (observations and control at t), rather than the entire data history (data from 0:t).

<div style="text-align:center"><img src ='{{site.baseurl}}/assets/SDC-T2/22-l-bayes-filter-theory-summary.00-00-52-03.still002.png' /></div>

* The motion model describes the prediction step of the filter while the observation model is the update step.
* The state estimation using the Bayes filter is dependent upon the interaction between prediction (motion model) and update (observation model steps) and all the localization methods discussed so far are realizations of the Bayes filter.
* In the next few sections we will learn how to estimate pseudo ranges, calculate the observation model probability, and complete the implementation of the observation model in C++.

## Observation Model Probability

We will complete our Bayes' filter by implementing the observation model. The observation model uses pseudo range estimates and observation measurements as inputs. Let's recap what is meant by a pseudo range estimate and an observation measurement.

For the figure below, the top 1d map (green car) shows our observation measurements. These are the distances from our actual car position at time t, to landmarks, as detected by sensors. In this example, those distances are 19m and 37m.

The bottom 1d map (yellow car) shows our pseudo range estimates. These are the distances we would expect given the landmarks and assuming a given position x at time t, of 20m. In this example, those distances are 5, 11, 39, and 57m.

<div style="text-align:center"><img src ='{{site.baseurl}}/assets/SDC-T2/obs-model-measurements-pseudoranges.png' /></div>

The observation model will be implemented by performing the following at each time step:

* Measure the range to landmarks up to 100m from the vehicle, in the driving direction (forward)
* Estimate a pseudo range from each landmark by subtracting pseudo position from the landmark position
* Match each pseudo range estimate to its closest observation measurement
* For each pseudo range and observation measurement pair, calculate a probability by passing relevant values to norm_pdf: norm_pdf(observation_measurement, pseudo_range_estimate, observation_stdev)
* Return the product of all probabilities

Why do we multiply all the probabilities in the last step? Our final signal (probability) must reflect all pseudo range, observation pairs. This blends our signal. For example, if we have a high probability match (small difference between the pseudo range estimate and the observation measurement) and low probability match (large difference between the pseudo range estimate and the observation measurement), our resultant probability will be somewhere in between, reflecting the overall belief we have in that state.

Let's practice this process using the following information and norm_pdf.

* pseudo position: x = 10m
* vector of landmark positions from our map: [6m, 15m, 21m, 40m]
* observation measurements: [5.5m, 11m]
* observation standard deviation: 1.0m

Why do we multiply all the probabilities in the last step? Our final signal (probability) must reflect all pseudo range, observation pairs. This blends our signal. For example, if we have a high probability match (small difference between the pseudo range estimate and the observation measurement) and low probability match (large difference between the pseudo range estimate and the observation measurement), our resultant probability will be somewhere in between, reflecting the overall belief we have in that state.




