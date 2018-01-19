---
layout: post
date:   2018-01-19 13:22
categories: OpenCV Matlab
---

## Compile C++ OpenCV Code in Matlab

Use `mex` to compile C++ code with external dependencies:
```matlab
OCVRoot = 'C:\Users\guanc\Downloads\opencv\build';
IPath = ['-I',fullfile(OCVRoot,'include')];
LPath = fullfile(OCVRoot, 'x64\vc14\lib');
lib1 = fullfile(LPath,'opencv_world340.lib');
lib2 = fullfile(LPath,'cxcore210d.lib');


>>>
Building with 'Microsoft Visual C++ 2015 Professional'.
MEX completed successfully.
```

Call `OpenCV` function to display `matlab` array as an image:

```cpp
/*==========================================================
 * encodeImage.cpp -an interface encoding image using OpenCV
 *
 *
 *
 * This is a MEX-file for MATLAB.
 * Author: Changjie Guan<changjie.guan@tassinternational.com>
 * Date: Jan 19, 2018
 *
 *========================================================*/

#include<stdio.h>

#include <iostream>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <string>

#include "mex.h"
 
using namespace cv;

/* The computational routine */
void arrayProduct(mxArray* input)
{
    Mat img = Mat(960, 1280, CV_8UC3, (uchar*)mxGetPr(input));
    
    namedWindow("Original Image", CV_WINDOW_AUTOSIZE);
    moveWindow("Original Image", 100, 100);
    imshow("Original Image", img );
    
    // wait for a key
    cvWaitKey(0);
 
}

/* The gateway function */
void mexFunction( int nlhs, mxArray *plhs[],
                  int nrhs, mxArray *prhs[])
{
    /* check for proper number of arguments */
    if(nrhs!=1) {
        mexErrMsgIdAndTxt("MyToolbox:arrayProduct:nrhs","One inputs required.");
    }
    
    mexPrintf("Input size = %d", mxGetNumberOfElements(prhs[0]));      
    
    /* create a pointer to the real data in the input matrix  */
    //inMatrix = reinterpret_cast<uchar*>(mxGetPr(prhs[0]));
   
    /* call the computational routine */
    arrayProduct(prhs[0]);
}
```

Notes: `matlab` stores image as RGB channel but `OpenCV` uses BGR channel. To use the function above, we need to swap channels of array in `matlab`. What's more, `matlab` uses column indexing and `C++` arrays use row indexing. To fix this, we `permute` the array in `matlab`.

```matlab
out = permute(img(:,:,[3 2 1]), [3 2 1]);
```

The first indexing `[3 2 1]` swaps `RGB` channel to `BGR`; and the second indexing `[3 2 1]` changes indexing order from 3rd dimension -> 2nd dimension -> first dimension (`BBB...GGG...RRR`) to `BGRBGRBGR...BGR` which OpenCV can parse.

To test the code, run:

```matlab
% load image
tmp =imread('CAPTURE2.JPEG');
% prepare data for OpenCV
tmp3 = permute(tmp(:,:,[3 2 1]), [3,2,1]);
% display image using OpenCV
encodeImageOCV_Changjie(tmp3);
```

You will see a correct image display.