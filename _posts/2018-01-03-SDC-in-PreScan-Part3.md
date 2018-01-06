---
layout: post
title: "SDC in PreScan: Step 3-- Control Vehicle using Keyboard Inputs"
Date: 2017-01-04 10:39
---

# Goals/Steps:

1. Build a deep learning server on linux machine(or windows) using Python
2. Build an interface on a Simulink model. Take keyboard inputs to control vehicle
3. Use keyboard inputs to control vehicle and use three parallel cameras to generate proper data set
4. Use the data set from step 3 to train the neural network
5. Send image from Simulink to the server using tcp/udp socket, and receive steering angles from the server

# Introduction


## Demo

Here is a screenshot of using this block:

<div style="text-align:center"><img src="{{site.baseurl}}/assets/SDC-PreScan/keyboard_inputs.png" /></div>

# Summary

In this post, we created a simulation block using `Matlab S-Function Level 2` that takes directional keyboard inputs and outputs steering direction and acceleration.
The follwing techniques are applied:

