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


