---
layout: post
date:   2018-01-17 16:31
categories: OpenCV TCP Socket ImageProcessing
---

## Introduction

## Configuration

## Contents

* Part 1
  1. Use OpenCV with Visual Studio 2017
  2. Build boost with VS2017
  3. Sending images with sockets
* Part 2
  1. Sockets on Windows and Linux
  2. Test Speed
  3. Increase transfer speed by compressing images
  
## Use OpenCV with VS 2017

1. Download binary files of OpenCV from internet. 
2. Unzip files.   
3. Add path of `bin` to environment variable `PATH`
4. Include OpenCV headers and libs
5. Add additional dependencies

## Build Boost on Windows with VS2017

1. Select architecture
2. Select versions: x86 or x64
3. Select `release` version or `debug` version
4. Select components
5. Use boost in VS2017

## Using sockets on Windows: winsock

### Refs
[http://www.binarytides.com/winsock-socket-programming-tutorial/](http://www.binarytides.com/winsock-socket-programming-tutorial/)

## Using sockets on Linux

### Refs

[https://stackoverflow.com/questions/15445207/sending-image-jpeg-through-socket-in-c-linux](https://stackoverflow.com/questions/15445207/sending-image-jpeg-through-socket-in-c-linux)

## Connect Two PCs with Crossover Cable

## Sending Image Files

## Sending Uchar Array

## Test Speed

## Compress Images


My final code block:
```cpp
std::string pic("BMW_M6_G-power_1082_1280x960.jpg");
Mat img;

img = imread(pic.c_str(), IMREAD_COLOR);
namedWindow("Display window", WINDOW_AUTOSIZE); // Create a window for display.
imshow("Display window", img);                // Show our image inside it.

waitKey(0); //

uchar *ptr = img.data;
int len = img.rows * img.cols * img.channels();
std::cout << "size of image: " << "  " << len << std::endl;

printf("\n");

//************Compress images*********
std::vector<uchar> buff;//buffer for coding
std::vector<int> param(2);
param[0] = cv::IMWRITE_JPEG_QUALITY;
param[1] = 80;//default(95) 0-100


auto start = std::chrono::high_resolution_clock::now();

cv::imencode(".jpeg", img, buff, param);

auto finish = std::chrono::high_resolution_clock::now();
float te = std::chrono::duration_cast<std::chrono::milliseconds>(finish - start).count();
printf("Compression Consumes %2.3fms\n", te);

std::cout << "Compressed Length = "<< buff.size() << std::endl;

cv::Mat imgbuf = Mat(1280, 960, CV_8UC3, buff.data());
cv::Mat imgMat = cv::imdecode(Mat(buff), CV_LOAD_IMAGE_COLOR);
std::cout << "de size =  "<< imgMat.rows*imgMat.cols*imgMat.channels() <<std::endl;
namedWindow("Display window2", WINDOW_AUTOSIZE);
imshow("Display window2", imgMat);

Mat dec;
dec = imdecode(buff, cv::IMREAD_COLOR);
namedWindow("Display window3", WINDOW_AUTOSIZE);
imshow("Display window3", dec);
waitKey(0); //
```

Here is the console output:

<div style="text-align:center"><img src="{{site.baseurl}}/assets/SendingImages/2018-01-18 08_57_23-x64_Release_server_exe.png" /></div>

References from internet:


```cpp
    std::vector<uchar> buff;//buffer for coding
    std::vector<int> param(2);
    param[0] = cv::IMWRITE_JPEG_QUALITY;
    param[1] = 80;//default(95) 0-100
    cv::imencode(".jpg", mat, buff, param);
```

C version:

```cpp
#include <opencv/cv.h>
#include <opencv/highgui.h>

int
main(int argc, char **argv)
{
    char *cvwin = "camimg";

    cvNamedWindow(cvwin, CV_WINDOW_AUTOSIZE);

    // setup code, initialization, etc ...
    [ ... ]

    while (1) {      
        // getImage was my routine for getting a jpeg from a camera
        char *img = getImage(fp);
        CvMat mat;

   // substitute 640/480 with your image width, height 
        cvInitMatHeader(&mat, 640, 480, CV_8UC3, img, 0);
        IplImage *cvImg = cvDecodeImage(&mat, CV_LOAD_IMAGE_COLOR);
        cvShowImage(cvwin, cvImg);
        cvReleaseImage(&cvImg);
        if (27 == cvWaitKey(1))         // exit when user hits 'ESC' key
        break;
    }

    cvDestroyWindow(cvwin);
}
```

```cpp
// decode jpg (or other image from a pointer)
// imageBuf contains the jpg image
    cv::Mat imgbuf = cv::Mat(480, 640, CV_8U, imageBuf);
    cv::Mat imgMat = cv::imdecode(imgbuf, CV_LOAD_IMAGE_COLOR);
// imgMat is the decoded image

// encode image into jpg
    cv::vector<uchar> buf;
    cv::imencode(".jpg", imgMat, buf, std::vector<int>() );
// encoded image is now in buf (a vector)
    imageBuf = (unsigned char *) realloc(imageBuf, buf.size());
    memcpy(imageBuf, &buf[0], buf.size());
//  size of imageBuf is buf.size();
```


```py
>>> img_str = cv2.imencode('.jpg', img)[1].tostring()
>>> type(img_str)
 'str'
 

>>> nparr = np.fromstring(STRING_FROM_DATABASE, np.uint8)
>>> img = cv2.imdecode(nparr, cv2.CV_LOAD_IMAGE_COLOR)
```


[Ref1](https://stackoverflow.com/questions/33535151/compress-mat-into-jpeg-and-save-the-result-into-memory)

[Ref2](https://docs.opencv.org/3.1.0/d4/da8/group__imgcodecs.html#ga461f9ac09887e47797a54567df3b8b63)

[https://stackoverflow.com/questions/801199/opencv-to-use-in-memory-buffers-or-file-pointers](https://stackoverflow.com/questions/801199/opencv-to-use-in-memory-buffers-or-file-pointers)


[https://stackoverflow.com/questions/17967320/python-opencv-convert-image-to-byte-string](https://stackoverflow.com/questions/17967320/python-opencv-convert-image-to-byte-string)

