---
layout: post
date:   2018-01-25 22:57
categories: ObjectDetection SDC
author: Udacity
---

<script src="https://cdnjs.cloudflare.com/ajax/libs/mathjax/2.7.0/MathJax.js?config=TeX-AMS-MML_HTMLorMML" type="text/javascript"></script>

## Manual Vehicle Detection

<div style="text-align:center"><img src ='{{site.baseurl}}/assets/VDT/bbox-example-image.jpg' /></div>

Here's your chance to be a human vehicle detector! In this lesson, you will be drawing a lot of bounding boxes on vehicle positions in images. Eventually, you'll have an algorithm that's outputting bounding box positions and you'll want an easy way to plot them up over your images. So, now is a good time to get familiar with the `cv2.rectangle()` function ([documentation](http://docs.opencv.org/2.4/modules/core/doc/drawing_functions.html)) that makes it easy to draw boxes of different size, shape and color.

In this exercise, your goal is to write a function that takes as arguments an image and a list of bounding box coordinates for each car. Your function should then draw bounding boxes on a copy of the image and return that as its output.

**Your output should look something like this**:

<div style="text-align:center"><img src ='{{site.baseurl}}/assets/VDT/manual-bbox-quiz-output.jpg' /></div>

Here, I don't actually care whether you identify the same cars as I do, or that you draw the same boxes, only that your function takes the appropriate inputs and yields the appropriate output. So here's what it should look like:

```py
# Define a function that takes an image, a list of bounding boxes, 
# and optional color tuple and line thickness as inputs
# then draws boxes in that color on the output

def draw_boxes(img, bboxes, color=(0, 0, 255), thick=6):
    # make a copy of the image
    # draw each bounding box on your image copy using cv2.rectangle()
    # return the image copy with boxes drawn
```

You'll draw bounding boxes with `cv2.rectangle()` like this:

```py
cv2.rectangle(image_to_draw_on, (x1, y1), (x2, y2), color, thick)
```

In this call to `cv2.rectangle()` your `image_to_draw_on` should be the copy of your image, then `(x1, y1)` and `(x2, y2)` are the `x` and `y` coordinates of any two opposing corners of the bounding box you want to draw. `color` is a 3-tuple, for example, `(0, 0, 255)` for blue, and `thick` is an optional integer parameter to define the box thickness.

Have a look at the image above with labeled axes, where I've drawn some bounding boxes and "guesstimate" where some of the box corners are. You should pass your bounding box positions to your `draw_boxes()` function as a list of tuple pairs, like this:

```py
bboxes = [((x1, y1), (x2, y2)), ((,),(,)), ...]
```

```py
import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

image = mpimg.imread('../data/bbox-example-image.jpg')

# Define a function that takes an image, a list of bounding boxes,
# and optional color tuple and line thickness as inputs
# then draws boxes in that color on the output

def draw_boxes(img, bboxes, color=(0, 0, 255), thick=6):
    # make a copy of the image
    draw_img = np.copy(img)
    # draw each bounding box on your image copy using cv2.rectangle()
    # return the image copy with boxes drawn

    for i in range(len(bboxes)):
        cv2.rectangle(draw_img,bboxes[i][0], bboxes[i][1], color=color,
                      thickness=thick)
    return draw_img # Change this line to return image copy with boxes
# Add bounding boxes in this format, these are just example coordinates.
bboxes = [((100, 100), (200, 200)), ((300, 300), (400, 400))]

result = draw_boxes(image, bboxes)
plt.imshow(result)
plt.show()
```

## Template Matching

<div style="text-align:center"><img src ='{{site.baseurl}}/assets/VDT/bbox-example-image.jpg' /></div>

To figure out when template matching works and when it doesn't, let's play around with the OpenCV `cv2.matchTemplate()` function! In the bounding boxes exercise, I found six cars in the image above. This time, we're going to play the opposite game. Assuming we know these six cars are what we're looking for, we can use them as templates and search the image for matches.

**Let's suppose you want to find the templates shown below in the image shown above (you can download these images [here](https://s3.amazonaws.com/udacity-sdc/Vehicle_Tracking/cutouts.zip) if you like):**

<div style="text-align:center"><img src ='{{site.baseurl}}/assets/VDT/cutouts.jpg' /></div>

Your goal in this exercise is to write a function that takes in an image and a list of templates, and returns a list of the best fit location (bounding box) for each of the templates within the image. OpenCV provides you with the handy function `cv2.matchTemplate()` ([documentation](http://docs.opencv.org/2.4/modules/imgproc/doc/object_detection.html)) to search the image, and `cv2.minMaxLoc()` ([documentation](http://docs.opencv.org/2.4/modules/core/doc/operations_on_arrays.html?highlight=minmaxloc#cv2.minMaxLoc)) to extract the location of the best match.

You can choose between "squared difference" or "correlation" methods in using `cv2.matchTemplate()`, but keep in mind with squared differences you need to locate the global minimum difference to find a match, while for correlation, you're looking for a global maximum.

Follow along with this [tutorial](http://opencv-python-tutroals.readthedocs.io/en/latest/py_tutorials/py_imgproc/py_template_matching/py_template_matching.html) provided by OpenCV to try some different template matching techniques. The function you write should work like this:

```py
def find_matches(img, template_list):
    # Iterate over the list of templates
    # Use cv2.matchTemplate() to search the image for each template
    # NOTE: You can use any of the cv2.matchTemplate() search methods
    # Use cv2.minMaxLoc() to extract the location of the best match in each case
    # Compile a list of bounding box corners as output
    # Return the list of bounding boxes
```

**However**, the point of this exercise is not to discover why template matching works for vehicle detection, but rather, why it doesn't! So, after you have a working implementation of the `find_matches()` function, try it on the second image, `temp-matching-example-2.jpg`, which is currently commented out.

In the second image, all of the same six cars are visible (just a few seconds later in the video), but you'll find that **none** of the templates find the correct match! This is because with template matching we can only find very close matches, and changes in size or orientation of a car make it impossible to match with a template.

**So, just to be clear, you goal here is to:**

1. Write a function that takes in an image and list of templates and returns a list of bounding boxes.
2. Find that your function works well to locate the six example templates taken from the first image.
3. Try your code on the second image, and find that template matching breaks easily.

```py
import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

image = mpimg.imread('../data/bbox-example-image.jpg')
# image = mpimg.imread('temp-matching-example-2.jpg')
templist = ['cutout1.jpg', 'cutout2.jpg', 'cutout3.jpg',
            'cutout4.jpg', 'cutout5.jpg', 'cutout6.jpg']

templist = ['../data/cutouts/'+a for a in templist]

# Here is your draw_boxes function from the previous exercise
def draw_boxes(img, bboxes, color=(0, 0, 255), thick=6):
    # Make a copy of the image
    imcopy = np.copy(img)
    # Iterate through the bounding boxes
    for bbox in bboxes:
        # Draw a rectangle given bbox coordinates
        cv2.rectangle(imcopy, bbox[0], bbox[1], color, thick)
    # Return the image copy with boxes drawn
    return imcopy


# Define a function that takes an image and a list of templates as inputs
# then searches the image and returns the a list of bounding boxes
# for matched templates
def find_matches(img, template_list):
    # Make a copy of the image to draw on
    im_copy = np.copy(img)
    # Define an empty list to take bbox coords
    bbox_list = []
    # Iterate through template list
    # Read in templates one by one
    # Use cv2.matchTemplate() to search the image
    #     using whichever of the OpenCV search methods you prefer
    # Use cv2.minMaxLoc() to extract the location of the best match
    # Determine bounding box corners for the match
    # Return the list of bounding boxes
    for imp in template_list:
        im_templ = mpimg.imread(imp)
        h = im_templ.shape[0]
        w = im_templ.shape[1]

        result = cv2.matchTemplate(im_copy, im_templ, cv2.TM_SQDIFF)

        result = np.abs(result) ** 3
        val, result = cv2.threshold(result, 0.01, 0, cv2.THRESH_TOZERO)
        result8 = cv2.normalize(result, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)
        cv2.imshow("result", result8)

        minVal, maxVal, minLoc, maxLoc = cv2.minMaxLoc(result)
        bbox_list.append(((minLoc[0],minLoc[1]),(minLoc[0]+w, minLoc[1]+h)))

    return bbox_list


bboxes = find_matches(image, templist)
result = draw_boxes(image, bboxes)
plt.imshow(result)
plt.show()
```
<div style="text-align:center"><img src ='{{site.baseurl}}/assets/VDT/template_match_result1.png' /></div>

<div style="text-align:center"><img src ='{{site.baseurl}}/assets/VDT/bad_detect_tm.png' /></div>



