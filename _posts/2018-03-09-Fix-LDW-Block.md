---
layout: post
date:   2018-03-09 10:06
categories: Matlab Simulink LDW
---

<script src="https://cdnjs.cloudflare.com/ajax/libs/mathjax/2.7.0/MathJax.js?config=TeX-AMS-MML_HTMLorMML" type="text/javascript"></script>

## Matlab Example of LDW

Type ` vipldws` in Matlab command window and it will propmpt a LDW example.

1. Identify lines
2. Get closest points at the base line
3. Apply a threshold to decide departure warning.


## Manual Camera Calibration


<div style="text-align:center"><img src ='{{site.baseurl}}/assets/tass/proj_formula.png' /></div>

<div style="text-align:center"><img src ='{{site.baseurl}}/assets/tass/pinhole_camera_model.png' /></div>

Take distortion into consideration:

<div style="text-align:center"><img src ='{{site.baseurl}}/assets/tass/proj_dist_formula.png' /></div>

### Set chessboard in PreScan

<div style="text-align:center"><img src ='{{site.baseurl}}/assets/tass/chessboard_ps.png' /></div>

* Distance between camera sensor and left edge is 4.5m
* Grid size = 0.5m
* Camera height = 1.32m

### Camera position

<div style="text-align:center"><img src ='{{site.baseurl}}/assets/tass/cam_pos.png' /></div>

* Image height = 720px
* Image width = 960px

### Calibration

<div style="text-align:center"><img src ='{{site.baseurl}}/assets/tass/initFrame.jpg' /></div>

Look at the corner inside the red circle of the above image. We know its position in real world is (x, y, z) = [-1.5m, -1.32m, 5.5m].

$$u = f_x * x/z + c_x$$


$$v = f_y * y/z + c_y$$

where $$c_x=480, c_y=360$$.

From the image, we get $$u = 174, v = 630$$.

**As a result, we obtain** $$f_x = f_y = 1125$$.

