---
layout: post
date:   2018-03-08 21:40
categories: SDC KalmanFilter DataFusion
title: "Project: Extended Kalman Filter"
---

<script src="https://cdnjs.cloudflare.com/ajax/libs/mathjax/2.7.0/MathJax.js?config=TeX-AMS-MML_HTMLorMML" type="text/javascript"></script>

## Project Introduction

Now that you have learned how the extended Kalman filter works, you are going to implement the extended Kalman filter in C++. We are providing simulated lidar and radar measurements detecting a bicycle that travels around your vehicle. You will use a Kalman filter, lidar measurements and radar measurements to track the bicycle's position and velocity.

The first step is to download the Term 2 simulator, which contains all the projects for Term 2 Self-Driving Car Nanodegree. More detailed instruction about setting up the simulator with uWebSocketIO can be found at the end of this section.

Lidar measurements are red circles, radar measurements are blue circles with an arrow pointing in the direction of the observed angle, and estimation markers are green triangles. The video below shows what the simulator looks like when a c++ script is using its Kalman filter to track the object. The simulator provides the script the measured data (either lidar or radar), and the script feeds back the measured estimation marker, and RMSE values from its Kalman filter.


### Download Links for Term 2 Simulator

[Term 2 Simulator Release](https://github.com/udacity/self-driving-car-sim/releases/)

**Running the Program**

1. Download the simulator and open it. In the main menu screen select Project 1: Bicycle tracker with EKF.

2. Once the scene is loaded you can hit the START button to observe how the object moves and how measurement markers are positioned in the data set. Also for more experimentation, "Data set 2" is included which is a reversed version of "Data set 1", also the second data set starts with a radar measurement where the first data set starts with a lidar measurement. At any time you can press the PAUSE button, to pause the scene or hit the RESTART button to reset the scene. Also the ARROW KEYS can be used to move the camera around, and the top left ZOOM IN/OUT buttons can be used to focus the camera. Pressing the ESCAPE KEY returns to the simulator main menu.

3. The [EKF project Github repository README](https://github.com/udacity/CarND-Extended-Kalman-Filter-Project) has more detailed instructions for installing and using c++ uWebScoketIO.

**NOTES:**

* Currently hitting Restart or switching between Data sets only refreshes the simulator state and not the Kalman Filter's saved results. The current procedure for refreshing the Kalman Filter is to close the connection, `ctrl+c` and reopen it, `./ExtendedKF`. If you don't do this when trying to run a different Data set or running the same Data set multiple times in a row, the RMSE values will become large because of the the previous different filter results still being observed in memory.

* Students have reported rapid expansion of log files when using the term 2 simulator. This appears to be associated with not being connected to uWebSockets. If this does occur, please make sure you are connected to uWebSockets. The following workaround may also be effective at preventing large log files.

    * create an empty log file
    * remove write permissions so that the simulator can't write to log

**What You'll Need to Do**

1. Read the repo's README for more detailed instructions.
2. Complete the Extended Kalman Filter algorithm in C++.
3. Ensure that your project compiles.
    * From the root of the repo:
      1. `mkdir build && cd build`
      2. `cmake .. && make`
      3. `./ExtendedKF`
4. Test your Kalman Filter in the simulator with Dataset 1. Ensure that the px, py, vx, and vy RMSE are below the values specified in the rubric.
5. Submit your project!

The project interface recently changed that allows the simulator to be used to run the Kalman filter and visualize the scene instead of only getting feedback from text files. The old project directory can still be accessed from Github in the [branch](https://github.com/udacity/CarND-Extended-Kalman-Filter-Project/commit/2d34d04936ace0fec7bfc5c434435a40872851cf) and for the time being students can submit either version of the project, using either the new simulator interface or the previous text based interface. Running the previous project set up requires `./ExtendedKF path/to/input.txt path/to/output.txt`

Example of Tracking with Lidar
Check out the video below to see a real world example of object tracking with lidar. In this project, you will only be tracking one object, but the video will give you a sense for how object tracking with lidar works:

<div style="text-align:center"><iframe width="560" height="315" src="https://www.youtube.com/embed/FMNJPX_sszU" frameborder="0" allow="autoplay; encrypted-media" allowfullscreen></iframe></div>


## uWebSocketIO Starter Guide

All of the projects in Term 2 and some in Term 3 involve using an open source package called [uWebSocketIO](https://github.com/uNetworking/uWebSockets). This package facilitates the same connection between the simulator and code that was used in the Term 1 Behavioral Cloning Project, but now with C++. The package does this by setting up a web socket server connection from the C++ program to the simulator, which acts as the host. In the project repository there are two scripts for installing uWebSocketIO - one for Linux and the other for macOS.

Note: Only uWebSocketIO branch e94b6e1, which the scripts reference, is compatible with the package installation.

**Linux Installation:**
From the project repository directory run the script: `install-ubuntu.sh`

**Mac Installation:**
From the project repository directory run the script: `install-mac.sh`

Some users report needing to use cmakepatch.txt which is automatically referenced and is also located in the project repository directory.

**Windows Installation**
Although it is possible to install uWebSocketIO to native Windows, the process is quite involved. Instead, you can use one of several Linux-like environments on Windows to install and run the package.

**Bash on Windows**
One of the newest features to Windows 10 users is an Ubuntu Bash environment that works great and is easy to setup and use. Here is a nice [step by step guide](https://www.howtogeek.com/249966/how-to-install-and-use-the-linux-bash-shell-on-windows-10/) for setting up the utility.

We recommend using the newest version of Ubunut Bash 16.04, which is able to run the `install-ubuntu.sh` script without complications. The link [here](https://www.howtogeek.com/278152/how-to-update-the-windows-bash-shell/) can help you check which version of Ubuntu Bash you are running, and also help you upgrade if you need to.

**Docker**
If you don't want to use Bash on Windows, or you don't have Windows 10, then you can use a virtual machine to run a Docker image that already contains all the project dependencies.

First [install Docker Toolbox for Windows](https://docs.docker.com/toolbox/toolbox_install_windows/).

Next, launch the Docker Quickstart Terminal. The default Linux virtual environment should load up. You can test that Docker is setup correctly by running `docker version` and `docker ps`.

You can enter a Docker image that has all the Term 2 project dependencies by running:

docker run -it -p 4567:4567 -v 'pwd':/work udacity/controls_kit:latest

Once inside Docker you can clone over the GitHub project repositories and run the project from there.

**Port forwarding is required when running code on VM and simulator on host**
For security reasons, the VM does not automatically open port forwarding, so you need to manually [enable port 4567](https://www.howtogeek.com/122641/how-to-forward-ports-to-a-virtual-machine-and-use-it-as-a-server/). This is needed for the C++ program to successfully connect to the host simulator.

Port Forwarding Instructions:

1. First open up Oracle VM VirtualBox
2. Click on the default session and select settings.
3. Click on Network, and then Advanced.
4. Click on Port Forwarding
5. Click on the green plus, adds new port forwarding rule.
6. Add a rule that assigns 4567 as both the host port and guest Port, as in the screenshot.

<div style="text-align:center"><img src ='{{site.baseurl}}/assets/SDC-T2/port-forward.png' /></div>

## Overview of a Kalman Filter: Initialize, Predict, Update

To review what we learned in the extended Kalman filter lectures, let's discuss the three main steps for programming a Kalman filter:

* **initializing** Kalman filter variables
* **predicting** where our object is going to be after a time step $$\Delta{t}$$
* **updating** where our object is based on sensor measurements

Then the prediction and update steps repeat themselves in a loop.

To measure how well our Kalman filter performs, we will then calculate root mean squared error comparing the Kalman filter results with the provided ground truth.

These three steps (initialize, predict, update) plus calculating RMSE encapsulate the entire extended Kalman filter project.

### Files in the Github src Folder

The files you need to work with are in the src folder of the github repository.

* main.cpp - communicates with the Term 2 Simulator receiving data measurements, calls a function to run the Kalman filter, calls a function to calculate RMSE
* FusionEKF.cpp - initializes the filter, calls the predict function, calls the update function
* kalman_filter.cpp- defines the predict function, the update function for lidar, and the update function for radar
* tools.cpp- function to calculate RMSE and the Jacobian matrix

The only files you need to modify are FusionEKF.cpp, kalman_filter.cpp, and tools.cpp.

###How the Files Relate to Each Other

Here is a brief overview of what happens when you run the code files:

1. Main.cpp reads in the data and sends a sensor measurement to FusionEKF.cpp
2. FusionEKF.cpp takes the sensor data and initializes variables and updates variables. The Kalman filter equations are not in this file. FusionEKF.cpp has a variable called `ekf_`, which is an instance of a KalmanFilter class. The `ekf_` will hold the matrix and vector values. You will also use the `ekf_` instance to call the predict and update equations.
3. The KalmanFilter class is defined in kalman_filter.cpp and kalman_filter.h. You will only need to modify 'kalman_filter.cpp', which contains functions for the prediction and update steps.


## main.cpp

Here we will discuss the `main.cpp` file. Although you will not need to modify this file, the project is easier to implement once you understand what the file is doing. As a suggestion, open the github repository for the project and look at the code files simultaneously with this lecture slide.

### main.cpp

You do not need to modify the `main.cpp`, but let's discuss what the file does.

The Term 2 simulator is a client, and the c++ program software is a web server.

We already discussed how `main.cpp` reads in the sensor data. Recall that `main.cpp` reads in the sensor data line by line from the client and stores the data into a measurement object that it passes to the Kalman filter for processing. Also a ground truth list and an estimation list are used for tracking RMSE.

`main.cpp` is made up of several functions within `main()`, these all handle the uWebsocketIO communication between the simulator and it's self.

All the main code loops in `h.onMessage()`, to have access to intial variables that we created at the beginning of `main()`, we pass pointers as arguments into the header of `h.onMessage()`.

For example

```cpp
h.onMessage([&fusionEKF,&tools,&estimations,&ground_truth](uWS::WebSocket<uWS::SERVER> ws, char *data, size_t length, uWS::OpCode opCode).
```

The rest of the arguments in `h.onMessage` are used to set up the server.

```cpp
 // Create a Fusion EKF instance
  FusionEKF fusionEKF;

  // used to compute the RMSE later
  vector<VectorXd> estimations;
  vector<VectorXd> ground_truth;

  //Call the EKF-based fusion
  fusionEKF.ProcessMeasurement(meas_package);
```

The code is:

* creating an instance of the `FusionEKF` class
* Receiving the measurement data calling the `ProcessMeasurement()` function. `ProcessMeasurement()` is responsible for the initialization of the Kalman filter as well as calling the prediction and update steps of the Kalman filter. You will be implementing the `ProcessMeasurement()` function in `FusionEKF.cpp`

Finally,

The rest of `main.cpp` will output the following results to the simulator:

* estimation position
* calculated RMSE

`main.cpp` will call a function to calculate root mean squared error:

```cpp
  // compute the accuracy (RMSE)
  Tools tools;
  cout << "Accuracy - RMSE:" << endl << tools.CalculateRMSE(estimations, ground_truth) << endl;
```

You will implement an RMSE function in the `tools.cpp` file.