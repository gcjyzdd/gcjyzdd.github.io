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

## Histograms of Color

You've looked at using raw pixel intensities as features and now we'll look at histograms of pixel intensity (color histograms) as features.

I'll use the template shown below from the last exercise as an example. This is a blown up version of the image, but If you want to try this yourself with the actual cutout image it's [here](https://s3.amazonaws.com/udacity-sdc/Vehicle_Tracking/cutout1.jpg):

<div style="text-align:center"><img src ='{{site.baseurl}}/assets/VDT/cutout1.jpg' /></div>

You can construct histograms of the R, G, and B channels like this:

```py
import matplotlib.image as mpimg
import numpy as np

# Read in the image
image = mpimg.imread('cutout1.jpg')

# Take histograms in R, G, and B
rhist = np.histogram(image[:,:,0], bins=32, range=(0, 256))
ghist = np.histogram(image[:,:,1], bins=32, range=(0, 256))
bhist = np.histogram(image[:,:,2], bins=32, range=(0, 256))
```

With `np.histogram()`, you don't actually have to specify the number of bins or the range, but here I've arbitrarily chosen 32 bins and specified `range=(0, 256)` in order to get orderly bin sizes. `np.histogram()` returns a tuple of two arrays. In this case, for example, `rhist[0]` contains the counts in each of the bins and `rhist[1]` contains the bin edges (so it is one element longer than `rhist[0]`).

To look at a plot of these results, we can compute the bin centers from the bin edges. Each of the histograms in this case have the same bins, so I'll just use the `rhist` bin edges:

```py
# Generating bin centers
bin_edges = rhist[1]
bin_centers = (bin_edges[1:]  + bin_edges[0:len(bin_edges)-1])/2
```

And then plotting up the results in a bar chart:

```py
# Plot a figure with all three bar charts
fig = plt.figure(figsize=(12,3))
plt.subplot(131)
plt.bar(bin_centers, rhist[0])
plt.xlim(0, 256)
plt.title('R Histogram')
plt.subplot(132)
plt.bar(bin_centers, ghist[0])
plt.xlim(0, 256)
plt.title('G Histogram')
plt.subplot(133)
plt.bar(bin_centers, bhist[0])
plt.xlim(0, 256)
plt.title('B Histogram')
```

Which gives us this result:

<div style="text-align:center"><img src ='{{site.baseurl}}/assets/VDT/rgb-histogram-plot.jpg' /></div>


These, collectively, are now our feature vector for this particular cutout image. We can concatenate them in the following way:

```py
hist_features = np.concatenate((rhist[0], ghist[0], bhist[0]))
```

Having a function that does all these steps might be useful for the project so for this next exercise, your goal is to write a function that takes an image and computes the RGB color histogram of features given a particular number of bins and pixels intensity range, and returns the concatenated RGB feature vector, like this:

```py
# Define a function to compute color histogram features  
def color_hist(img, nbins=32, bins_range=(0, 256)):
    # Compute the histogram of the RGB channels separately
    # Concatenate the histograms into a single feature vector
    # Return the feature vector
```

```py
import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

image = mpimg.imread('../data/cutouts/cutout1.jpg')


# Define a function to compute color histogram features
def color_hist(img, nbins=32, bins_range=(0, 256)):
    # Compute the histogram of the RGB channels separately
    rhist = np.histogram(image[:, :, 0], bins=32, range=(0, 256))
    ghist = np.histogram(image[:, :, 1], bins=32, range=(0, 256))
    bhist = np.histogram(image[:, :, 2], bins=32, range=(0, 256))

    bin_edges = rhist[1]

    # Generating bin centers
    bin_centers = (bin_edges[1:] + bin_edges[0:len(bin_edges) - 1]) / 2
    # Concatenate the histograms into a single feature vector
    hist_features = np.concatenate((rhist[0], ghist[0], bhist[0]))
    # Return the individual histograms, bin_centers and feature vector
    return rhist, ghist, bhist, bin_centers, hist_features


rh, gh, bh, bincen, feature_vec = color_hist(image, nbins=32, bins_range=(0, 256))

# Plot a figure with all three bar charts
if rh is not None:
    fig = plt.figure(figsize=(12, 3))
    plt.subplot(131)
    plt.bar(bincen, rh[0])
    plt.xlim(0, 256)
    plt.title('R Histogram')
    plt.subplot(132)
    plt.bar(bincen, gh[0])
    plt.xlim(0, 256)
    plt.title('G Histogram')
    plt.subplot(133)
    plt.bar(bincen, bh[0])
    plt.xlim(0, 256)
    plt.title('B Histogram')
    fig.tight_layout()

    plt.show()
else:
    print('Your function is returning None for at least one variable...')
```    

## Histogram Comparison

Let's look at the color histogram features for two totally different images. The first image is of a red car and the second a blue car. The red car's color histograms are displayed on the first row and the blue car's are displayed on the second row below. Here we are just looking at 8 bins per RGB channel.

If we had to, we could differentiate the two images based on the differences in histograms alone. As expected the image of the red car has a greater intensity of total bin values in the `R Histogram 1` (Red Channel) compared to the blue car's `R Histogram 2`. In contrast the blue car has a greater intensity of total bin values in `B Histogram 2` (Blue Channel) than the red car's `B Histogram 1` features.

Differentiating images by the intensity and range of color they contain can be helpful for looking at car vs non-car images.

<div style="text-align:center"><img src ='{{site.baseurl}}/assets/VDT/hist-compare.jpg' /></div>

## Explore Color Spaces

You can study the distribution of color values in an image by plotting each pixel in some color space. Here's a code snippet that you can use to generate 3D plots:

```py
import cv2
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def plot3d(pixels, colors_rgb,
        axis_labels=list("RGB"), axis_limits=((0, 255), (0, 255), (0, 255))):
    """Plot pixels in 3D."""

    # Create figure and 3D axes
    fig = plt.figure(figsize=(8, 8))
    ax = Axes3D(fig)

    # Set axis limits
    ax.set_xlim(*axis_limits[0])
    ax.set_ylim(*axis_limits[1])
    ax.set_zlim(*axis_limits[2])

    # Set axis labels and sizes
    ax.tick_params(axis='both', which='major', labelsize=14, pad=8)
    ax.set_xlabel(axis_labels[0], fontsize=16, labelpad=16)
    ax.set_ylabel(axis_labels[1], fontsize=16, labelpad=16)
    ax.set_zlabel(axis_labels[2], fontsize=16, labelpad=16)

    # Plot pixel values with colors given in colors_rgb
    ax.scatter(
        pixels[:, :, 0].ravel(),
        pixels[:, :, 1].ravel(),
        pixels[:, :, 2].ravel(),
        c=colors_rgb.reshape((-1, 3)), edgecolors='none')

    return ax  # return Axes3D object for further manipulation


# Read a color image
img = cv2.imread("000275.png")

# Select a small fraction of pixels to plot by subsampling it
scale = max(img.shape[0], img.shape[1], 64) / 64  # at most 64 rows and columns
img_small = cv2.resize(img, (np.int(img.shape[1] / scale), np.int(img.shape[0] / scale)), interpolation=cv2.INTER_NEAREST)

# Convert subsampled image to desired color space(s)
img_small_RGB = cv2.cvtColor(img_small, cv2.COLOR_BGR2RGB)  # OpenCV uses BGR, matplotlib likes RGB
img_small_HSV = cv2.cvtColor(img_small, cv2.COLOR_BGR2HSV)
img_small_rgb = img_small_RGB / 255.  # scaled to [0, 1], only for plotting

# Plot and show
plot3d(img_small_RGB, img_small_rgb)
plt.show()

plot3d(img_small_HSV, img_small_rgb, axis_labels=list("HSV"))
plt.show()
```

### Analyze video frames

Use this to first explore some video frames, and see if you can locate clusters of colors that correspond to the sky, trees, specific cars, etc. Here are some sample images for you to use (these are taken from the [KITTI Vision Benchmark Suite](http://www.cvlibs.net/datasets/kitti/)):

<div style="text-align:center"><img src ='{{site.baseurl}}/assets/VDT/000275.png' /></div>

<div style="text-align:center"><img src ='{{site.baseurl}}/assets/VDT/001240.png' /></div>

<div style="text-align:center"><img src ='{{site.baseurl}}/assets/VDT/000528.png' /></div>

### Analyze vehicle and non-vehicle images

You might've noticed that it is hard to distinguish between the class of pixels you are interested in (vehicles, in this case) from the background. So it may be more beneficial to plot pixels from vehicle and non-vehicle images separately. See if you can identify any trends using these samples:

<div style="text-align:center"><img src ='{{site.baseurl}}/assets/VDT/25.png' /></div>
<div style="text-align:center"><img src ='{{site.baseurl}}/assets/VDT/31.png' /></div>
<div style="text-align:center"><img src ='{{site.baseurl}}/assets/VDT/53.png' /></div>
<div style="text-align:center"><img src ='{{site.baseurl}}/assets/VDT/8.png' /></div>
<div style="text-align:center"><img src ='{{site.baseurl}}/assets/VDT/2.png' /></div>
<div style="text-align:center"><img src ='{{site.baseurl}}/assets/VDT/3.png' /></div>

Try experimenting with different color spaces such as LUV or HLS to see if you can find a way to consistently separate vehicle images from non-vehicles. It doesn't have to be perfect, but it will help when combined with other kinds of features fed into a classifier.

## Spatial Binning of Color

<div style="text-align:center"><img src ='{{site.baseurl}}/assets/VDT/spatial-binning.jpg' /></div>

You saw earlier in the lesson that template matching is not a particularly robust method for finding vehicles unless you know exactly what your target object looks like. However, raw pixel values are still quite useful to include in your feature vector in searching for cars.

While it could be cumbersome to include three color channels of a full resolution image, you can perform spatial binning on an image and still retain enough information to help in finding vehicles.

As you can see in the example above, even going all the way down to 32 x 32 pixel resolution, the car itself is still clearly identifiable by eye, and this means that the relevant features are still preserved at this resolution.

A convenient function for scaling down the resolution of an image is OpenCV's `cv2.resize()`. You can use it to scale a color image or a single color channel like this (you can find the original image here):

```py
import cv2
import matplotlib.image as mpimg

image = mpimg.imread('test_img.jpg')
small_img = cv2.resize(image, (32, 32))
print(small_img.shape)
(32, 32, 3)
```

If you then wanted to convert this to a one dimensional [feature vector](https://en.wikipedia.org/wiki/Feature_vector), you could simply say something like:

```py
feature_vec = small_img.ravel()
print(feature_vec.shape)
(3072,)
```

Ok, but 3072 elements is still quite a few features! Could you get away with even lower resolution? I'll leave that for you to explore later when you're training your classifier.

Now that you've played with color spaces a bit, it's probably a good time to write a function that allows you to convert any test image into a feature vector that you can feed your classifier. Your goal in this exercise is to write a function that takes an image, a color space conversion, and the resolution you would like to convert it to, and returns a feature vector. Something like this:

```py
# Define a function that takes an image, a color space, 
# and a new image size
# and returns a feature vector
def bin_spatial(img, color_space='RGB', size=(32, 32)):
    # Convert image to new color space (if specified)
    # Use cv2.resize().ravel() to create the feature vector
    # Return the feature vector
```

## Data Exploration

For the exercises throughout the rest of this lesson, we'll use a relatively small labeled dataset to try out feature extraction and training a classifier. Before we get on to extracting HOG features and training a classifier, let's explore the dataset a bit. This dataset is a subset of the data you'll be starting with for the project.

There's no need to download anything at this point, but if you want to, you can download this subset of images for [vehicles](https://s3.amazonaws.com/udacity-sdc/Vehicle_Tracking/vehicles_smallset.zip) and [non-vehicles](https://s3.amazonaws.com/udacity-sdc/Vehicle_Tracking/non-vehicles_smallset.zip), or if you prefer you can directly grab the larger project dataset for [vehicles](https://s3.amazonaws.com/udacity-sdc/Vehicle_Tracking/vehicles.zip) and [non-vehicles](https://s3.amazonaws.com/udacity-sdc/Vehicle_Tracking/non-vehicles.zip).

These datasets are comprised of images taken from the [GTI vehicle image database](http://www.gti.ssr.upm.es/data/Vehicle_database.html), the [KITTI vision benchmark suite](http://www.cvlibs.net/datasets/kitti/), and examples extracted from the project video itself. In this exercise, you can explore the data to see what you're working with.

You are also welcome and encouraged to explore the recently released [Udacity labeled dataset](https://github.com/udacity/self-driving-car/tree/master/annotations). Each of the Udacity datasets comes with a labels.csv file that gives bounding box corners for each object labeled.

<div style="text-align:center"><img src ='{{site.baseurl}}/assets/VDT/car-not-car-examples.jpg' /></div>

Here, I've provided you with the code to extract the car/not-car image filenames into two lists. Write a function that takes in these two lists and returns a dictionary with the keys "n_cars", "n_notcars", "image_shape", and "data_type", like this:

```py
# Define a function to return some characteristics of the dataset 
def data_look(car_list, notcar_list):
    data_dict = {}
    # Define a key in data_dict "n_cars" and store the number of car images
    # Define a key "n_notcars" and store the number of notcar images
    # Read in a test image, either car or notcar
    # Define a key "image_shape" and store the test image shape 3-tuple
    # Define a key "data_type" and store the data type of the test image.
    # Return data_dict
    return data_dict
```

```py
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
import cv2
import glob

# from skimage.feature import hog
# from skimage import color, exposure
# images are divided up into vehicles and non-vehicles

images = glob.glob('../data/vehicles_smallset/cars1/*.jpeg')
images.extend(glob.glob('../data/non-vehicles_smallset/notcars1/*.jpeg'))
cars = []
notcars = []

for image in images:
    if 'image' in image or 'extra' in image:
        notcars.append(image)
    else:
        cars.append(image)


# Define a function to return some characteristics of the dataset
def data_look(car_list, notcar_list):
    data_dict = {}
    # Define a key in data_dict "n_cars" and store the number of car images
    data_dict["n_cars"] = len(car_list)
    # Define a key "n_notcars" and store the number of notcar images
    data_dict["n_notcars"] = len(notcar_list)
    # Read in a test image, either car or notcar
    tmp_im = mpimg.imread(car_list[0])
    # Define a key "image_shape" and store the test image shape 3-tuple
    data_dict["image_shape"] = tmp_im.shape
    # Define a key "data_type" and store the data type of the test image.
    data_dict["data_type"] = tmp_im.dtype

    # Return data_dict
    return data_dict


data_info = data_look(cars, notcars)

print('Your function returned a count of',
      data_info["n_cars"], ' cars and',
      data_info["n_notcars"], ' non-cars')
print('of size: ', data_info["image_shape"], ' and data type:',
      data_info["data_type"])
# Just for fun choose random car / not-car indices and plot example images
car_ind = np.random.randint(0, len(cars))
notcar_ind = np.random.randint(0, len(notcars))

# Read in car / not-car images
car_image = mpimg.imread(cars[car_ind])
notcar_image = mpimg.imread(notcars[notcar_ind])

# Plot the examples
fig = plt.figure()
plt.subplot(121)
plt.imshow(car_image)
plt.title('Example Car Image')
plt.subplot(122)
plt.imshow(notcar_image)
plt.title('Example Not-car Image')
plt.show()
```
## scikit-image HOG

Now that we've got a dataset let's extract some HOG features!

<div style="text-align:center"><img src ='{{site.baseurl}}/assets/VDT/scikit-image-logo.png' /></div>

The [scikit-image](http://scikit-image.org/) package has a built in function to extract Histogram of Oriented Gradient features. The documentation for this function can be found [here](http://scikit-image.org/docs/dev/api/skimage.feature.html?highlight=feature%20hog#skimage.feature.hog) and a brief explanation of the algorithm and tutorial can be found [here](http://scikit-image.org/docs/dev/auto_examples/features_detection/plot_hog.html).

The scikit-image `hog()` function takes in a single color channel or grayscaled image as input, as well as various parameters. These parameters include `orientations`, `pixels_per_cell` and `cells_per_block`.

The number of `orientations` is specified as an integer, and represents the number of orientation bins that the gradient information will be split up into in the histogram. Typical values are between 6 and 12 bins.

The `pixels_per_cell` parameter specifies the cell size over which each gradient histogram is computed. This paramater is passed as a 2-tuple so you could have different cell sizes in x and y, but cells are commonly chosen to be square.

The `cells_per_block` parameter is also passed as a 2-tuple, and specifies the local area over which the histogram counts in a given cell will be normalized. Block normalization is not necessarily required, but generally leads to a more robust feature set.

There is another optional power law or "gamma" normalization scheme set by the flag `transform_sqrt`. This type of normalization may help reduce the effects of shadows or other illumination variation, but will cause an error if your image contains negative values (because it's taking the square root of image values).

<div style="text-align:center"><img src ='{{site.baseurl}}/assets/VDT/hog-visualization.jpg' /></div>

This is where things get a little confusing though. Let's say you are computing HOG features for an image like the one shown above that is $$64\times64$$ pixels. If you set `pixels_per_cell=(8, 8)` and `cells_per_block=(2, 2)` and `orientations=9`. How many elements will you have in your HOG feature vector for the entire image?

You might guess the number of orientations times the number of cells, or $$9\times8\times8 = 576$$, but that's not the case if you're using block normalization! In fact, the HOG features for all cells in each block are computed at each block position and the block steps across and down through the image cell by cell.

So, the actual number of features in your final feature vector will be the total number of block positions multiplied by the number of cells per block, times the number of orientations, or in the case shown above: $$7\times7\times2\times2\times9 = 17647$$.

For the example above, you would call the `hog()` function on a single color channel img like this:

```py
from skimage.feature import hog
pix_per_cell = 8
cell_per_block = 2
orient = 9

features, hog_image = hog(img, orientations=orient,
                          pixels_per_cell=(pix_per_cell, pix_per_cell), 
                          cells_per_block=(cell_per_block, cell_per_block), 
                          visualise=True, feature_vector=False,
                          block_norm="L2-Hys")
```

The `visualise=True` flag tells the function to output a visualization of the HOG feature computation as well, which we're calling `hog_image` in this case. If we take a look at a single color channel for a random car image, and its corresponding HOG visulization, they look like this:

<div style="text-align:center"><img src ='{{site.baseurl}}/assets/VDT/car-and-hog.jpg' /></div>

The HOG visualization is not actually the feature vector, but rather, a representation that shows the dominant gradient direction within each cell with brightness corresponding to the strength of gradients in that cell, much like the "star" representation in the last video.

If you look at the `features` output, you'll find it's an array of shape $$7\times7\times2\times2\times97$$. This corresponds to the fact that a grid of $$7\times7$$ blocks were sampled, with $$2\times2$$ cells in each block and 9 orientations per cell. You can unroll this array into a feature vector using `features.ravel()`, which yields, in this case, a one dimensional array of length 1764.

Alternatively, you can set the `feature_vector=True` flag when calling the `hog()` function to automatically unroll the features. In the project, it could be useful to have a function defined that you could pass an image to with specifications for `orientations`, `pixels_per_cell`, and `cells_per_block`, as well as flags set for whether or not you want the feature vector unrolled and/or a visualization image, so let's write it!

```py
# Define a function to return HOG features and visualization
def get_hog_features(img, orient, pix_per_cell, cell_per_block, vis=False, feature_vec=True):
    if vis == True:
        # Use skimage.hog() to get both features and a visualization
        return features, hog_image
    else:      
        # Use skimage.hog() to get features only
        return features
```

**Note**: you could also include a keyword to set the `tranform_sqrt` flag but for this exercise you can just leave this at the default value of `transform_sqrt=False`.
        
## Combine and Normalize Features

<div style="text-align:center"><img src ='{{site.baseurl}}/assets/VDT/scaled-features-vis.jpg' /></div>

Now that you've got several feature extraction methods in your toolkit, you're almost ready to train a classifier, but first, as in any machine learning application, you need to normalize your data. Python's `sklearn` package provides you with the `StandardScaler()` method to accomplish this task. To read more about how you can choose different normalizations with the `StandardScaler()` method, check out the [documentation](http://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.StandardScaler.html).

To apply `StandardScaler()` you need to first have your data in the right format, as a numpy array where each row is a single feature vector. I will often create a list of feature vectors, and then convert them like this:

```py
import numpy as np
feature_list = [feature_vec1, feature_vec2, ...]
# Create an array stack, NOTE: StandardScaler() expects np.float64
X = np.vstack(feature_list).astype(np.float64)
```

You can then fit a scaler to X, and scale it like this:

```py
from sklearn.preprocessing import StandardScaler
# Fit a per-column scaler
X_scaler = StandardScaler().fit(X)
# Apply the scaler to X
scaled_X = X_scaler.transform(X)
```

Now, `scaled_X` contains the normalized feature vectors. In this next exercise, I've provided the feature scaling step for you, but I need you to provide the feature vectors. I've also provided versions of the `bin_spatial()` and `color_hist()` functions you wrote in previous exercises.

Your goal in this exercise is to write a function that takes in a list of image filenames, reads them one by one, then applies a color conversion (if necessary) and uses `bin_spatial()` and `color_hist()` to generate feature vectors. Your function should then concatenate those two feature vectors and append the result to a list. After cycling through all the images, your function should return the list of feature vectors. Something like this:

```py
# Define a function to extract features from a list of images
# Have this function call bin_spatial() and color_hist()
def extract_features(imgs, cspace='RGB', spatial_size=(32, 32),
                        hist_bins=32, hist_range=(0, 256)):
    # Create a list to append feature vectors to
    features = []
    # Iterate through the list of images
        # Read in each one by one
        # apply color conversion if other than 'RGB'
        # Apply bin_spatial() to get spatial color features
        # Apply color_hist() to get color histogram features
        # Append the new feature vector to the features list
    # Return list of feature vectors
    return features
```

## Parameter Tuning

### SVM Hyperparameters

In the SVM lesson, Katie mentioned optimizing the Gamma and C parameters.

Successfully tuning your algorithm involves searching for a kernel, a gamma value and a C value that minimize prediction error. To tune your SVM vehicle detection model, you can use one of scikit-learn's parameter tuning algorithms.

When tuning SVM, remember that you can only tune the C parameter with a linear kernel. For a non-linear kernel, you can tune C and gamma.

### Parameter Tuning in Scikit-learn

Scikit-learn includes two algorithms for carrying out an automatic parameter search:

* [GridSearchCV](http://scikit-learn.org/stable/modules/generated/sklearn.model_selection.GridSearchCV.html#sklearn.model_selection.GridSearchCV)
* [RandomizedSearchCV](http://scikit-learn.org/stable/modules/generated/sklearn.model_selection.RandomizedSearchCV.html#sklearn.model_selection.RandomizedSearchCV)

`GridSearchCV` exhaustively works through multiple parameter combinations, cross-validating as it goes. The beauty is that it can work through many combinations in only a couple extra lines of code.

For example, if I input the values C:[0.1, 1, 10] and gamma:[0.1, 1, 10], gridSearchCV will train and cross-validate every possible combination of (C, gamma): (0.1, 0.1), (0.1, 1), (0.1, 10), (1, .1), (1, 1), etc.

`RandomizedSearchCV` works similarly to `GridSearchCV` except `RandomizedSearchCV` takes a random sample of parameter combinations. `RandomizedSearchCV` is faster than `GridSearchCV` since `RandomizedSearchCV` uses a subset of the parameter combinations.

### Cross-validation with `GridSearchCV`

`GridSearchCV` uses 3-fold cross validation to determine the best performing parameter set. GridSearchCV will take in a training set and divide the training set into three equal partitions. The algorithm will train on two partitions and then validate using the third partition. Then `GridSearchCV` chooses a different partition for validation and trains with the other two partitions. Finally, `GridSearchCV` uses the last remaining partition for cross-validation and trains with the other two partitions.

By default, `GridSearchCV` uses accuracy as an error metric by averaging the accuracy for each partition. So for every possible parameter combination, `GridSearchCV` calculates an accuracy score. Then `GridSearchCV` will choose the parameter combination that performed the best.

### Scikit-learn Cross Validation Example

Here's an example from the sklearn [documentation](http://scikit-learn.org/stable/modules/generated/sklearn.grid_search.GridSearchCV.html) for implementing GridSearchCV:

```py
parameters = {'kernel':('linear', 'rbf'), 'C':[1, 10]}
svr = svm.SVC()
clf = grid_search.GridSearchCV(svr, parameters)
clf.fit(iris.data, iris.target)
```

Let's break this down line by line.

```py
parameters = {'kernel':('linear', 'rbf'), 'C':[1, 10]} 
```

A dictionary of the parameters, and the possible values they may take. In this case, they're playing around with the kernel (possible choices are 'linear' and 'rbf'), and C (possible choices are 1 and 10).

Then a 'grid' of all the following combinations of values for (kernel, C) are automatically generated:


 ('rbf', 1)      ('rbf', 10)    

 ('linear', 1)	 ('linear', 10) 

Each is used to train an SVM, and the performance is then assessed using cross-validation.

```py
svr = svm.SVC() 
```

This looks kind of like creating a classifier, just like we've been doing since the first lesson. But note that the "clf" isn't made until the next line--this is just saying what kind of algorithm to use. Another way to think about this is that the "classifier" isn't just the algorithm in this case, it's algorithm plus parameter values. Note that there's no monkeying around with the kernel or C; all that is handled in the next line.

```py
clf = grid_search.GridSearchCV(svr, parameters) 
```

This is where the first bit of magic happens; the classifier is being created. We pass the algorithm (svr) and the dictionary of parameters to try (parameters) and it generates a grid of parameter combinations to try.

```py
clf.fit(iris.data, iris.target) 
```

And the second bit of magic. The fit function now tries all the parameter combinations, and returns a fitted classifier that's automatically tuned to the optimal parameter combination. You can now access the parameter values via `clf.best_params_`.

## Color Classify

<div style="text-align:center"><img src ='{{site.baseurl}}/assets/VDT/car-color-and-hist.jpg' /></div>

Now we'll try training a classifier on our dataset. First, we'll see how well it does just using spatially binned color and color histograms.

To do this, we'll use the functions you defined in previous exercises, namely, `bin_spatial()`, `color_hist()`, and `extract_features()`. We'll then read in our car and non-car images, extract the color features for each, and scale the feature vectors to zero mean and unit variance.

All that remains is to define a labels vector, shuffle and split the data into training and testing sets, and finally, define a classifier and train it!

Our labels vector `y` in this case will just be a binary vector indicating whether each feature vector in our dataset corresponds to a car or non-car (1's for cars, 0's for non-cars). Given lists of car and non-car features (the output of `extract_features()`) we can define a labels vector like this:

```py
import numpy as np
# Define a labels vector based on features lists
y = np.hstack((np.ones(len(car_features)), 
              np.zeros(len(notcar_features))))
```

Next, we'll stack and scale our feature vectors like before:


```py
from sklearn.preprocessing import StandardScaler
# Create an array stack of feature vectors
X = np.vstack((car_features, notcar_features)).astype(np.float64)
# Fit a per-column scaler
X_scaler = StandardScaler().fit(X)
# Apply the scaler to X
scaled_X = X_scaler.transform(X)
```

And now we're ready to shuffle and split the data into training and testing sets. To do this we'll use the Scikit-Learn `train_test_split()` function, but it's worth noting that recently, this function moved from the `sklearn.cross_validation` package (in `sklearn` version $$<=$$0.17) to the `sklearn.model_selection` package (in `sklearn` version $$>=$$0.18).

In the quiz editor we're still running `sklearn` v0.17, so we'll import it like this:

```py
from sklearn.cross_validation import train_test_split
# But, if you are using scikit-learn >= 0.18 then use this:
# from sklearn.model_selection import train_test_split
```

`train_test_split()` performs both the shuffle and split of the data and you'll call it like this (here choosing to initialize the shuffle with a different random state each time):

```py
# Split up data into randomized training and test sets
rand_state = np.random.randint(0, 100)
X_train, X_test, y_train, y_test = train_test_split(
    scaled_X, y, test_size=0.2, random_state=rand_state)
```

Warning: when dealing with image data that was extracted from video, you may be dealing with sequences of images where your target object (vehicles in this case) appear almost identical in a whole series of images. In such a case, even a randomized train-test split will be subject to overfitting because images in the training set may be nearly identical to images in the test set. For the subset of images used in the next several quizzes, this is not a problem, but to optimize your classifier for the project, you may need to worry about time-series of images!

--------

Now, you're ready to define and train a classifier! Here we'll try a Linear Support Vector Machine. To define and train your classifier it takes just a few lines of code:

```py
from sklearn.svm import LinearSVC
# Use a linear SVC (support vector classifier)
svc = LinearSVC()
# Train the SVC
svc.fit(X_train, y_train)
```

Then you can check the accuracy of your classifier on the test dataset like this:

```py
print('Test Accuracy of SVC = ', svc.score(X_test, y_test))
```

Or you can make predictions on a subset of the test data and compare directly with ground truth:

```py
print('My SVC predicts: ', svc.predict(X_test[0:10].reshape(1, -1)))
print('For labels: ', y_test[0:10])
```

Play with the parameter values `spatial` and `histbin` in the exercise below to see how the classifier accuracy and training time vary with the feature vector input.

## HOG Classify

Alright, so classification by color features alone is pretty effective! Now let's try classifying with HOG features and see how well we can do.

<div style="text-align:center"><img src ='{{site.baseurl}}/assets/VDT/car-and-hog.jpg' /></div>

**NOTE**: if you copy the code from the exercise below onto your local machine, but are running sklearn version >= 0.18 you will need to change from calling:

```py
from sklearn.cross_validation import train_test_split
```
to:
```py
from sklearn.model_selection import train_test_split
```

In the exercise below, you're given all the code to extract HOG features and train a linear SVM. There is no right or wrong answer, but your mission, should you choose to accept it, is to play with the parameters `colorspace`, `orient`, `pix_per_cell`, `cell_per_block`, and `hog_channel` to get a feel for what combination of parameters give the best results.

**Note**: `hog_channel` can take values of 0, 1, 2, or "ALL", meaning that you extract HOG features from the first, second, third, or all color channels respectively.

## How many windows?

<div style="text-align:center"><img src ='{{site.baseurl}}/assets/VDT/car-identified.jpg' /></div>

To implement a sliding window search, you need to decide what size window you want to search, where in the image you want to start and stop your search, and how much you want windows to overlap. So, let's try an example to see how many windows we would be searching given a particular image size, window size, and overlap.

Suppose you have an image that is 256 x 256 pixels and you want to search windows of a size 128 x 128 pixels each with an overlap of 50% between adjacent windows in both the vertical and horizontal dimensions. Your sliding window search would then look like this:

<div style="text-align:center"><img src ='{{site.baseurl}}/assets/VDT/sliding-window.jpg' /></div>

### Test

1280x960, sample window 64x64, overlapping 50%, it needs to search

(1280/(64 * 0.5) - 1) * (960/(64 * 0.5) - 1=1131)

times.

## Sliding Window Implementation

<div style="text-align:center"><img src ='{{site.baseurl}}/assets/VDT/sliding-window.jpg' /></div>

In the last exercise, you saw how the number of windows scales with image size, window size, and overlap. In the project it will be useful to have a function to perform a sliding window search on an image, so let's write one! This will just be the first iteration, where you don't actually need to do anything besides plot a rectangle at each window position.

So, your goal here is to write a function that takes in an image, start and stop positions in both x and y (imagine a bounding box for the entire search region), window size (x and y dimensions), and overlap fraction (also for both x and y). Your function should return a list of bounding boxes for the search windows, which will then be passed to draw `draw_boxes()` function.

## Multi-scale Windows

<div style="text-align:center"><img src ='{{site.baseurl}}/assets/VDT/Screenshot from 2018-01-29 20-59-18.png' /></div>

Multi-scale windows:

* search ROI
* set min and max window size
* small sizes near horizontal, large near bottom

## Search and Classify

<div style="text-align:center"><img src ='{{site.baseurl}}/assets/VDT/bbox-example-image.jpg' /></div>

Now you're able to run a sliding window search on an image and you've trained a classifier... time to combine both steps and search for cars!

You already have all the tools you need to do this from the previous exercises. Just train your classifier, then run your sliding window search, extract features, and predict whether each window contains a car or not. You'll probably find some false positives, but soon we'll deal with removing them.

In the `lesson_functions.py` tab on the quiz editor, you'll find all the functions we've defined so far in the lesson, including `get_hog_features()`, `bin_spatial()`, `color_hist()`, `extract_features()`, `slide_window()`, and `draw_boxes()`. These are now all imported for use in the quiz with this command:

```py
from lesson_functions import *
```

Two new functions are defined: `single_img_features()` and `search_windows()`. You can use these to search over all the windows defined by your `slide_windows()`, extract features at each window position, and predict with your classifier on each set of features.

We have limited the sample size to 500 each of car and not-car training images for the classifier to avoid quiz evaluator timeout, but if you search a very large number of windows or create huge feature vectors, you still may run into timeout issues. If you want to explore more, download the data and try the code on your local machine.

You can download the subset of data used in this lesson for vehicles and non-vehicles, or if you prefer, you can directly grab the larger project dataset for vehicles and non-vehicles.

In this exercise, experiment with different color and gradient feature sets, different search window sizes and overlap to get an idea of how your classifier performs with different training features. Getting false positives in the skies and treetops? Try restricting your search area on the image with `y_start_stop` in the `slide_window()` function. What combination of features works best?

```py
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
import cv2
import glob
import time
from sklearn.svm import LinearSVC
from sklearn.preprocessing import StandardScaler
from skimage.feature import hog
from lesson_functions import *
# NOTE: the next import is only valid for scikit-learn version <= 0.17
# for scikit-learn >= 0.18 use:
# from sklearn.model_selection import train_test_split
from sklearn.cross_validation import train_test_split

# Define a function to extract features from a single image window
# This function is very similar to extract_features()
# just for a single image rather than list of images
def single_img_features(img, color_space='RGB', spatial_size=(32, 32),
                        hist_bins=32, orient=9, 
                        pix_per_cell=8, cell_per_block=2, hog_channel=0,
                        spatial_feat=True, hist_feat=True, hog_feat=True):    
    #1) Define an empty list to receive features
    img_features = []
    #2) Apply color conversion if other than 'RGB'
    if color_space != 'RGB':
        if color_space == 'HSV':
            feature_image = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
        elif color_space == 'LUV':
            feature_image = cv2.cvtColor(img, cv2.COLOR_RGB2LUV)
        elif color_space == 'HLS':
            feature_image = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)
        elif color_space == 'YUV':
            feature_image = cv2.cvtColor(img, cv2.COLOR_RGB2YUV)
        elif color_space == 'YCrCb':
            feature_image = cv2.cvtColor(img, cv2.COLOR_RGB2YCrCb)
    else: feature_image = np.copy(img)      
    #3) Compute spatial features if flag is set
    if spatial_feat == True:
        spatial_features = bin_spatial(feature_image, size=spatial_size)
        #4) Append features to list
        img_features.append(spatial_features)
    #5) Compute histogram features if flag is set
    if hist_feat == True:
        hist_features = color_hist(feature_image, nbins=hist_bins)
        #6) Append features to list
        img_features.append(hist_features)
    #7) Compute HOG features if flag is set
    if hog_feat == True:
        if hog_channel == 'ALL':
            hog_features = []
            for channel in range(feature_image.shape[2]):
                hog_features.extend(get_hog_features(feature_image[:,:,channel], 
                                    orient, pix_per_cell, cell_per_block, 
                                    vis=False, feature_vec=True))      
        else:
            hog_features = get_hog_features(feature_image[:,:,hog_channel], orient, 
                        pix_per_cell, cell_per_block, vis=False, feature_vec=True)
        #8) Append features to list
        img_features.append(hog_features)

    #9) Return concatenated array of features
    return np.concatenate(img_features)

# Define a function you will pass an image 
# and the list of windows to be searched (output of slide_windows())
def search_windows(img, windows, clf, scaler, color_space='RGB', 
                    spatial_size=(32, 32), hist_bins=32, 
                    hist_range=(0, 256), orient=9, 
                    pix_per_cell=8, cell_per_block=2, 
                    hog_channel=0, spatial_feat=True, 
                    hist_feat=True, hog_feat=True):

    #1) Create an empty list to receive positive detection windows
    on_windows = []
    #2) Iterate over all windows in the list
    for window in windows:
        #3) Extract the test window from original image
        test_img = cv2.resize(img[window[0][1]:window[1][1], window[0][0]:window[1][0]], (64, 64))      
        #4) Extract features for that window using single_img_features()
        features = single_img_features(test_img, color_space=color_space, 
                            spatial_size=spatial_size, hist_bins=hist_bins, 
                            orient=orient, pix_per_cell=pix_per_cell, 
                            cell_per_block=cell_per_block, 
                            hog_channel=hog_channel, spatial_feat=spatial_feat, 
                            hist_feat=hist_feat, hog_feat=hog_feat)
        #5) Scale extracted features to be fed to classifier
        test_features = scaler.transform(np.array(features).reshape(1, -1))
        #6) Predict using your classifier
        prediction = clf.predict(test_features)
        #7) If positive (prediction == 1) then save the window
        if prediction == 1:
            on_windows.append(window)
    #8) Return windows for positive detections
    return on_windows
    
    
# Read in cars and notcars
images = glob.glob('*.jpeg')
cars = []
notcars = []
for image in images:
    if 'image' in image or 'extra' in image:
        notcars.append(image)
    else:
        cars.append(image)

# Reduce the sample size because
# The quiz evaluator times out after 13s of CPU time
sample_size = 500
cars = cars[0:sample_size]
notcars = notcars[0:sample_size]

### TODO: Tweak these parameters and see how the results change.
color_space = 'RGB' # Can be RGB, HSV, LUV, HLS, YUV, YCrCb
orient = 9  # HOG orientations
pix_per_cell = 8 # HOG pixels per cell
cell_per_block = 2 # HOG cells per block
hog_channel = 0 # Can be 0, 1, 2, or "ALL"
spatial_size = (16, 16) # Spatial binning dimensions
hist_bins = 16    # Number of histogram bins
spatial_feat = True # Spatial features on or off
hist_feat = True # Histogram features on or off
hog_feat = True # HOG features on or off
y_start_stop = [None, None] # Min and max in y to search in slide_window()

car_features = extract_features(cars, color_space=color_space, 
                        spatial_size=spatial_size, hist_bins=hist_bins, 
                        orient=orient, pix_per_cell=pix_per_cell, 
                        cell_per_block=cell_per_block, 
                        hog_channel=hog_channel, spatial_feat=spatial_feat, 
                        hist_feat=hist_feat, hog_feat=hog_feat)
notcar_features = extract_features(notcars, color_space=color_space, 
                        spatial_size=spatial_size, hist_bins=hist_bins, 
                        orient=orient, pix_per_cell=pix_per_cell, 
                        cell_per_block=cell_per_block, 
                        hog_channel=hog_channel, spatial_feat=spatial_feat, 
                        hist_feat=hist_feat, hog_feat=hog_feat)

X = np.vstack((car_features, notcar_features)).astype(np.float64)                        
# Fit a per-column scaler
X_scaler = StandardScaler().fit(X)
# Apply the scaler to X
scaled_X = X_scaler.transform(X)

# Define the labels vector
y = np.hstack((np.ones(len(car_features)), np.zeros(len(notcar_features))))


# Split up data into randomized training and test sets
rand_state = np.random.randint(0, 100)
X_train, X_test, y_train, y_test = train_test_split(
    scaled_X, y, test_size=0.2, random_state=rand_state)

print('Using:',orient,'orientations',pix_per_cell,
    'pixels per cell and', cell_per_block,'cells per block')
print('Feature vector length:', len(X_train[0]))
# Use a linear SVC 
svc = LinearSVC()
# Check the training time for the SVC
t=time.time()
svc.fit(X_train, y_train)
t2 = time.time()
print(round(t2-t, 2), 'Seconds to train SVC...')
# Check the score of the SVC
print('Test Accuracy of SVC = ', round(svc.score(X_test, y_test), 4))
# Check the prediction time for a single sample
t=time.time()

image = mpimg.imread('bbox-example-image.jpg')
draw_image = np.copy(image)

# Uncomment the following line if you extracted training
# data from .png images (scaled 0 to 1 by mpimg) and the
# image you are searching is a .jpg (scaled 0 to 255)
#image = image.astype(np.float32)/255

windows = slide_window(image, x_start_stop=[None, None], y_start_stop=y_start_stop, 
                    xy_window=(96, 96), xy_overlap=(0.5, 0.5))

hot_windows = search_windows(image, windows, svc, X_scaler, color_space=color_space, 
                        spatial_size=spatial_size, hist_bins=hist_bins, 
                        orient=orient, pix_per_cell=pix_per_cell, 
                        cell_per_block=cell_per_block, 
                        hog_channel=hog_channel, spatial_feat=spatial_feat, 
                        hist_feat=hist_feat, hog_feat=hog_feat)                       

window_img = draw_boxes(draw_image, hot_windows, color=(0, 0, 255), thick=6)                    

plt.imshow(window_img)

if 1:
    dist_pickle = {}
    dist_pickle["svc"] = svc
    dist_pickle["scaler"] = X_scaler
    dist_pickle["orient"] = orient
    dist_pickle["pix_per_cell"] = pix_per_cell
    dist_pickle["cell_per_block"] = cell_per_block
    dist_pickle["spatial_size"] = spatial_size
    dist_pickle["hist_bins"] = hist_bins
    pickle.dump(dist_pickle, open("../data/svc_pickle.p", "wb"))
```

## Hog Sub-sampling Window Search

Now lets explore a more efficient method for doing the sliding window approach, one that allows us to only have to extract the Hog features once. The code below defines a single function `find_cars` that's able to both extract features and make predictions.

The `find_cars` only has to extract hog features once and then can be sub-sampled to get all of its overlaying windows. Each window is defined by a scaling factor where a scale of 1 would result in a window that's 8 x 8 cells then the overlap of each window is in terms of the cell distance. This means that a `cells_per_step = 2` would result in a search window overlap of 75%. Its possible to run this same function multiple times for different scale values to generate multiple-scaled search windows.

<div style="text-align:center"><img src ='{{site.baseurl}}/assets/VDT/hog-sub.jpg' /></div>

```py
# hog_subsample.py

import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
import pickle
import cv2
from sklearn.svm import LinearSVC
from lesson_functions import *

dist_pickle = pickle.load(open("../data/svc_pickle.p", "rb"))  # svc_pickle_complete
svc = dist_pickle["svc"]
X_scaler = dist_pickle["scaler"]
orient = dist_pickle["orient"]
pix_per_cell = dist_pickle["pix_per_cell"]
cell_per_block = dist_pickle["cell_per_block"]
spatial_size = dist_pickle["spatial_size"]
hist_bins = dist_pickle["hist_bins"]

print(orient, pix_per_cell, cell_per_block, spatial_size, hist_bins)

img = mpimg.imread('../data/bbox-example-image.jpg')


# Define a single function that can extract features using hog sub-sampling and make predictions
def find_cars(img, ystart, ystop, scale, svc, X_scaler, orient, pix_per_cell, cell_per_block, spatial_size, hist_bins):
    draw_img = np.copy(img)
    img = img.astype(np.float32) / 255

    box_list = []

    img_tosearch = img[ystart:ystop, :, :]
    ctrans_tosearch = convert_color(img_tosearch, conv='RGB2YCrCb')  # RGB2YCrCb
    if scale != 1:
        imshape = ctrans_tosearch.shape
        ctrans_tosearch = cv2.resize(ctrans_tosearch, (np.int(imshape[1] / scale), np.int(imshape[0] / scale)))

    ch1 = ctrans_tosearch[:, :, 0]
    ch2 = ctrans_tosearch[:, :, 1]
    ch3 = ctrans_tosearch[:, :, 2]

    # Define blocks and steps as above
    nxblocks = (ch1.shape[1] // pix_per_cell) - cell_per_block + 1
    nyblocks = (ch1.shape[0] // pix_per_cell) - cell_per_block + 1
    nfeat_per_block = orient * cell_per_block ** 2

    # 64 was the orginal sampling rate, with 8 cells and 8 pix per cell
    window = 64
    nblocks_per_window = (window // pix_per_cell) - cell_per_block + 1
    cells_per_step = 2  # Instead of overlap, define how many cells to step
    nxsteps = (nxblocks - nblocks_per_window) // cells_per_step + 1
    nysteps = (nyblocks - nblocks_per_window) // cells_per_step + 1

    # Compute individual channel HOG features for the entire image
    hog1 = get_hog_features(ch1, orient, pix_per_cell, cell_per_block, feature_vec=False)
    hog2 = get_hog_features(ch2, orient, pix_per_cell, cell_per_block, feature_vec=False)
    hog3 = get_hog_features(ch3, orient, pix_per_cell, cell_per_block, feature_vec=False)

    for xb in range(nxsteps):
        for yb in range(nysteps):
            ypos = yb * cells_per_step
            xpos = xb * cells_per_step
            # Extract HOG for this patch
            hog_feat1 = hog1[ypos:ypos + nblocks_per_window, xpos:xpos + nblocks_per_window].ravel()
            hog_feat2 = hog2[ypos:ypos + nblocks_per_window, xpos:xpos + nblocks_per_window].ravel()
            hog_feat3 = hog3[ypos:ypos + nblocks_per_window, xpos:xpos + nblocks_per_window].ravel()
            hog_features = np.hstack((hog_feat1, hog_feat2, hog_feat3))

            xleft = xpos * pix_per_cell
            ytop = ypos * pix_per_cell

            # Extract the image patch
            subimg = cv2.resize(ctrans_tosearch[ytop:ytop + window, xleft:xleft + window], (64, 64))

            # Get color features
            spatial_features = bin_spatial(subimg, size=spatial_size)
            hist_features = color_hist(subimg, nbins=hist_bins)

            # Scale features and make a prediction
            test_features = X_scaler.transform(
                np.hstack((spatial_features, hist_features, hog_features)).reshape(1, -1))
            # test_features = X_scaler.transform(np.hstack((shape_feat, hist_feat)).reshape(1, -1))
            test_prediction = svc.predict(test_features)

            if test_prediction == 1:
                xbox_left = np.int(xleft * scale)
                ytop_draw = np.int(ytop * scale)
                win_draw = np.int(window * scale)
                cv2.rectangle(draw_img, (xbox_left, ytop_draw + ystart),
                              (xbox_left + win_draw, ytop_draw + win_draw + ystart), (0, 0, 255), 6)

                box_list.append([(xbox_left, ytop_draw + ystart), (xbox_left + win_draw, ytop_draw + win_draw + ystart)])
    return draw_img, box_list


ystart = 400
ystop = 656
scale = 1.5

out_img, box_list = find_cars(img, ystart, ystop, scale, svc, X_scaler, orient, pix_per_cell, cell_per_block, spatial_size,
                    hist_bins)

plt.imshow(out_img)
plt.show()

pickle.dump(box_list, open("../data/bbox_pickle.p", "wb"))
```

```py
# lesson_functions.py
import numpy as np
import cv2
from skimage.feature import hog


def convert_color(img, conv='RGB2YCrCb'):
    if conv == 'RGB2YCrCb':
        return cv2.cvtColor(img, cv2.COLOR_RGB2YCrCb)
    if conv == 'BGR2YCrCb':
        return cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)
    if conv == 'RGB2LUV':
        return cv2.cvtColor(img, cv2.COLOR_RGB2LUV)


def get_hog_features(img, orient, pix_per_cell, cell_per_block,
                     vis=False, feature_vec=True):
    # Call with two outputs if vis==True
    if vis == True:
        features, hog_image = hog(img, orientations=orient,
                                  pixels_per_cell=(pix_per_cell, pix_per_cell),
                                  cells_per_block=(cell_per_block, cell_per_block),
                                  transform_sqrt=False,
                                  visualise=vis, feature_vector=feature_vec,
                                  block_norm='L2-Hys')
        return features, hog_image
    # Otherwise call with one output
    else:
        features = hog(img, orientations=orient,
                       pixels_per_cell=(pix_per_cell, pix_per_cell),
                       cells_per_block=(cell_per_block, cell_per_block),
                       transform_sqrt=False,
                       visualise=vis, feature_vector=feature_vec,
                       block_norm='L2-Hys')
        return features


def bin_spatial(img, size=(32, 32)):
    color1 = cv2.resize(img[:, :, 0], size).ravel()
    color2 = cv2.resize(img[:, :, 1], size).ravel()
    color3 = cv2.resize(img[:, :, 2], size).ravel()
    return np.hstack((color1, color2, color3))


def color_hist(img, nbins=32):  # bins_range=(0, 256)
    # Compute the histogram of the color channels separately
    channel1_hist = np.histogram(img[:, :, 0], bins=nbins)
    channel2_hist = np.histogram(img[:, :, 1], bins=nbins)
    channel3_hist = np.histogram(img[:, :, 2], bins=nbins)
    # Concatenate the histograms into a single feature vector
    hist_features = np.concatenate((channel1_hist[0], channel2_hist[0], channel3_hist[0]))
    # Return the individual histograms, bin_centers and feature vector
    return hist_features
```


## Multiple Detections & False Positives

Here are six consecutive frames from the project video and I'm showing all the bounding boxes for where my classifier reported positive detections. You can see that overlapping detections exist for each of the two vehicles, and in two of the frames, I find a false positive detection on the guardrail to the left. In this exercise, you'll build a heat-map from these detections in order to combine overlapping detections and remove false positives.

<div style="text-align:center"><img src ='{{site.baseurl}}/assets/VDT/screen-shot-2017-01-29-at-2.52.00-pm.png' /></div>

To make a heat-map, you're simply going to add "heat" (+=1) for all pixels within windows where a positive detection is reported by your classifier. The individual heat-maps for the above images look like this:

<div style="text-align:center"><img src ='{{site.baseurl}}/assets/VDT/screen-shot-2017-01-29-at-7.11.18-pm.png' /></div>

In the exercise below, you are provided with a list of bounding boxes for the detections in the images shown above. Rather than consider heat-maps for each individual image, you'll write a function that adds "heat" to a map for a list of bounding boxes.

```py
def add_heat(heatmap, bbox_list):
    # Iterate through list of bboxes
    for box in bbox_list:
        # Add += 1 for all pixels inside each bbox
        # Assuming each "box" takes the form ((x1, y1), (x2, y2))
        heatmap[box[0][1]:box[1][1], box[0][0]:box[1][0]] += 1

    # Return updated heatmap
    return heatmap
```

If your classifier is working well, then the "hot" parts of the map are where the cars are, and by imposing a threshold, you can reject areas affected by false positives. So let's write a function to threshold the map as well.

```py
def apply_threshold(heatmap, threshold):
    # Zero out pixels below the threshold
    heatmap[heatmap <= threshold] = 0
    # Return thresholded map
    return heatmap
```
    
In practice, you will want to integrate a heat map over several frames of video, such that areas of multiple detections get "hot", while transient false positives stay "cool". You can then simply threshold your heatmap to remove false positives.

Once you have a thresholded heat-map, there are many ways you could go about trying to figure out how many cars you have in each frame and which pixels belong to which cars, but one of the most straightforward solutions is to use the `label()` function from `scipy.ndimage.measurements`. You'll use it like this:

```py
from scipy.ndimage.measurements import label
labels = label(heatmap)
```

Now `labels` is a 2-tuple, where the first item is an array the size of the `heatmap` input image and the second element is the number of labels (cars) found.

In the above series of images the same false positive window appeared twice, so if I set a threshold of 2 (meaning set all values <= 2 to 0) and then run the `labels()` function. I get a result that 2 cars were identified (the two islands in the heat-map) and the labeled image where pixels are set to 0 for background, 1 for car number 1, and 2 for car number 2.

```py
heatmap = threshold(heatmap, 2)
labels = label(heatmap)
print(labels[1], 'cars found')
plt.imshow(labels[0], cmap='gray')
```

2 cars found    

<div style="text-align:center"><img src ='{{site.baseurl}}/assets/VDT/screen-shot-2017-01-29-at-5.23.31-pm.png' /></div>

Next, you can take your `labels` image and put bounding boxes around the labeled regions. You could do this in the following manner:

```py
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
import cv2

def draw_labeled_bboxes(img, labels):
    # Iterate through all detected cars
    for car_number in range(1, labels[1]+1):
        # Find pixels with each car_number label value
        nonzero = (labels[0] == car_number).nonzero()
        # Identify x and y values of those pixels
        nonzeroy = np.array(nonzero[0])
        nonzerox = np.array(nonzero[1])
        # Define a bounding box based on min/max x and y
        bbox = ((np.min(nonzerox), np.min(nonzeroy)), (np.max(nonzerox), np.max(nonzeroy)))
        # Draw the box on the image
        cv2.rectangle(img, bbox[0], bbox[1], (0,0,255), 6)
    # Return the image
    return img

# Read in the last image above
image = mpimg.imread('img105.jpg')
# Draw bounding boxes on a copy of the image
draw_img = draw_labeled_bboxes(np.copy(image), labels)
# Display the image
plt.imshow(draw_img)
```

And the output looks like this:

<div style="text-align:center"><img src ='{{site.baseurl}}/assets/VDT/screen-shot-2017-01-29-at-6.53.54-pm.png' /></div>

## Tips and Tricks for the Project

<div style="text-align:center"><img src ='{{site.baseurl}}/assets/VDT/multi-vehicle-bbox.jpg' /></div>

### Extract HOG features just once for the entire region of interest in each full image/video frame

In one of the previous exercises you extracted HOG features from each individual window as you searched across the image, but it turns out this is rather inefficient. To speed things up, extract HOG features just once for the entire region of interest (i.e. lower half of each frame of video) and subsample that array for each sliding window. To do this, apply `skimage.feature.hog()` with the flag `feature_vec=False`, like this:

```py
from skimage.feature import hog
orient = 9
pix_per_cell = 8
cell_per_block = 2

feature_array = hog(img, orientations=orient, pixels_per_cell=(pix_per_cell, pix_per_cell), cells_per_block=(cell_per_block, cell_per_block), visualise=False, feature_vector=False)
```

The output `feature_array` will have a shape of `(n_yblocks, n_xblocks, 2, 2, 9)`, where `n_yblocks` and `n_xblocks` are determined by the shape of your region of interest (i.e. how many blocks fit across and down your image in x and y).

So, for example, if you used `cells_per_block=2` in extracting features from the 64x64 pixel training images, then you would want to extract subarrays of shape `(7, 7, 2, 2, 9)` from `feature_array` and then use `np.ravel()` to unroll the feature vector.

### Make sure your images are scaled correctly

The training dataset provided for this project ( vehicle and non-vehicle images) are in the `.png` format. Somewhat confusingly, `matplotlib image` will read these in on a scale of 0 to 1, but `cv2.imread()` will scale them from 0 to 255. Be sure if you are switching between `cv2.imread()` and `matplotlib image` for reading images that you scale them appropriately! Otherwise your feature vectors can get screwed up.

To add to the confusion, `matplotlib image` will read `.jpg` images in on a scale of 0 to 255 so if you are testing your pipeline on `.jpg` images remember to scale them accordingly. And if you take an image that is scaled from 0 to 1 and change color spaces using `cv2.cvtColor()` you'll get back an image scaled from 0 to 255. So just be sure to be consistent between your training data features and inference features!

### Be sure to normalize your training data

Use `sklearn.preprocessing.StandardScaler()` to normalize your feature vectors for training your classifier as described in this lesson. Then apply the same scaling to each of the feature vectors you extract from windows in your test images.

### Random shuffling of data

When dealing with image data that was extracted from video, you may be dealing with sequences of images where your target object (vehicles in this case) appear almost identical in a whole series of images. In such a case, even a randomized train-test split will be subject to overfitting because images in the training set may be nearly identical to images in the test set.

For the project vehicles dataset, the `GTI*` folders contain time-series data. In the KITTI folder, you may see the same vehicle appear more than once, but typically under significantly different lighting/angle from other instances.

While it is possible to achieve a sufficiently good result on the project without worrying about time-series issues, if you really want to optimize your classifier, you should devise a train/test split that avoids having nearly identical images in both your training and test sets. This means extracting the time-series tracks from the GTI data and separating the images manually to make sure train and test images are sufficiently different from one another.

---

### Project Details

For this project, we provide you with a labeled dataset and your job is to decide what features to extract, then train a classifier and ultimately track vehicles in a video stream. Here are links to the labeled data for vehicle and non-vehicle examples to train your classifier. These example images come from a combination of the GTI vehicle image database, the KITTI vision benchmark suite, and examples extracted from the project video itself.

Udacity recently released a labeled dataset of our own, which you are encouraged to take advantage of to augment your training data. You can find the Udacity data here. In each of the folders containing images there's a csv file containing all the labels and bounding boxes. To add vehicle images to your training data, you'll need to use the csv files to extract the bounding box regions and scale them to the same size as the rest of the training images.

The project video will be the same one as for the Advanced Lane Finding Project. The reason for this is that, assuming you already have a working implementation of lane finding for this video, once your vehicle detection pipeline works, you can add it to your lane finding pipeline and do both analyses simultaneously! You can use the test images from the lane finding project to start with, or extract other frames from the video to work on.

