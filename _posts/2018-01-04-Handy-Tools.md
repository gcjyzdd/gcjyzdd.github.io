---
layout: post
date:   2018-01-04 16:29
title: "Handy Tools"
categories: Tools
---

## Download Youtube

[pytube](https://github.com/nficano/pytube): A lightweight, dependency-free Python library (and command-line utility) for downloading YouTube Videos.

### Usage

```py
from pytube import YouTube
YouTube('https://www.youtube.com/watch?time_continue=25&v=siAMDK8C_x8').streams.first().download()
```

### Install
```sh
pip install pytube
```
---
## PDF

### Combine PDFs

`pdftk` is a very comvenient tool to process PDFs. 

### Install:
```sh
sudo apt install pdftk
```

### Usage

```sh
>>>ls
b11496-10.pdf  b11496-14.pdf  b11496-2.pdf   b11496-6.pdf   
b11496-11.pdf  b11496-15.pdf  b11496-3.pdf   b11496-7.pdf   
b11496-12.pdf  b11496-16.pdf  b11496-4.pdf   b11496-8.pdf   
b11496-13.pdf  b11496-1.pdf   b11496-5.pdf   b11496-9.pdf 
>>>pdftk b11496-{1..16}.pdf cat output Matlab-Java-Programming.pdf
>>>ls
b11496-10.pdf  b11496-15.pdf  b11496-4.pdf  b11496-9.pdf
b11496-11.pdf  b11496-16.pdf  b11496-5.pdf  Matlab-Java-Programming.pdf
b11496-12.pdf  b11496-1.pdf   b11496-6.pdf
b11496-13.pdf  b11496-2.pdf   b11496-7.pdf
b11496-14.pdf  b11496-3.pdf   b11496-8.pdf
```

### Crop PDF

```sh
pdfcrop --bbox '40 100 580 760' zips2013.pdf /tmp/out.pdf
```

---
## tmux

### Basic Usage

Split windows vertically:
```
<ctrl-B>%
```

Split windows horizontally:
```
<ctrl-B>"
```

Switch tabs:
```
<ctrl-B>leftarrow[rightarrow, etc]
```

Resize a tab:
```
<ctrl-B><ctrl [left,up,right]>
```

Close a tab:
```
<ctrl-B>x
```

Create a new window:
```
<ctrl-B>c
```

Switch to previous window:
```
<ctrl-B>p
```

Switch to next window:
```
<ctrl-B>n
```

Close a window:
```
<ctrl-B>&
```

[Scroll](https://superuser.com/questions/209437/how-do-i-scroll-in-tmux):

`Ctrl-b` then `[` then you can use your normal navigation keys to scroll around (eg. `Up Arrow` or `PgDn`). Press `q` to quit scroll mode.

---
## Crop Video using FFMPEG

Use the crop filter:
```
ffmpeg -i in.mp4 -filter:v "crop=out_w:out_h:x:y" out.mp4
```

Where the options are as follows:

* `out_w` is the width of the output rectangle
* `out_h` is the height of the output rectangle
* `x` and `y` specify the top left corner of the output rectangle

[Ref](https://video.stackexchange.com/questions/4563/how-can-i-crop-a-video-with-ffmpeg)


## Cut videos using FFMPEG

Try using this. It is the fastest and best ffmpeg-way I have figure it out:

```sh
 ffmpeg -i input.mp4 -ss 00:01:00 -to 00:02:00 -c copy output.mp4
```
 
**This command trims your video in seconds!**

I have explained it on my blog [here](http://blog.georgechalhoub.com/2017/03/trimming-videos-via-ffmpeg.html):

* `i`: This specifies the input file. In that case, it is (input.mp4). 
* `ss`: Used with -i, this seeks in the input file (input.mp4) to position. 
* `00:01:00`: This is the time your trimmed video will start with. 
* `to`: This specifies duration from start (00:01:40) to end (00:02:12). 
* `00:02:00`: This is the time your trimmed video will end with. 
* `c copy`: This is an option to trim via stream copy. (NB: Very fast) 

The timing format is: hh:mm:ss

Please note that the current highly upvoted answer is outdated and the trim would be extremely slow. For more information, look at this official ffmpeg [article](https://trac.ffmpeg.org/wiki/Seeking#Cuttingsmallsections).

[Ref](https://stackoverflow.com/questions/18444194/cutting-the-videos-based-on-start-and-end-time-using-ffmpeg)


## Compress videos using ffmpeg

```
ffmpeg -i input.mp4 -vcodec h264 -acodec mp2 output.mp4
```

## Change video format

```sh
ffmpeg -i example.mov example.mp4 -hide_banner
```

## Remove audio of a video

```sh
cat cvtOGV.sh 
       
ffmpeg -i $1 \
   -c:v libx264 -preset veryslow -crf 22 \
   -c:a aac -b:a 160k -strict -2 \
   $2       

```

## Combine Videos

3 videos to 2x2:

```sh
#!/bin/bash
# combine three videos into one

ffmpeg \
    -i $1  -i $2 -i $3\
    -filter_complex 'nullsrc=size=1280x960 [base];[0:v] setpts=PTS-STARTPTS, scale=640x480 [upperleft];[1:v] setpts=PTS-STARTPTS, scale=640x480 [rightupper];[2:v] setpts=PTS-STARTPTS, scale=640x480 [leftlower];[base][upperleft] overlay=shortest=1 [tmp1];[tmp1][rightupper] overlay=shortest=1:x=640 [tmp2];[tmp2][leftlower] overlay=shortest=1:y=480'\
 -c:v libx264 -c:a libmp3lame -qscale:a 2 -ac 2 -ar 44100 $4
```

2 videos to 1x2:

```sh
#!/bin/bash
# combine two videos into one

ffmpeg \
    -i $1  -i $2\
    -filter_complex 'nullsrc=size=1920x720 [base];[0:v] setpts=PTS-STARTPTS, scale=960x720 [upperleft];[1:v] setpts=PTS-STARTPTS, scale=960x720 [rightupper];[base][upperleft] overlay=shortest=1 [tmp1];[tmp1][rightupper] overlay=shortest=1:x=960'\
 -c:v libx264 -c:a libmp3lame -qscale:a 2 -ac 2 -ar 44100 $3
```

[ref](https://trac.ffmpeg.org/wiki/Create%20a%20mosaic%20out%20of%20several%20input%20videos)


---
## sed - replace file in place

Replace 

```
![]({{"assets/Screenshot from 2017-12-28 15-01-32.png" | absolute_url}}) 
```
with 
```
![]({{site.baseurl}}/assets/Screenshot from 2017-12-28 15-01-32.png)
```
using `sed`:

{% raw %}
~~~sh
sed -i 's/{{"assets/{{site.baseurl}}\/assets/g' tmp.txt
sed -i 's/"\s|\sabsolute_url}}//g' tmp.txt 
~~~
{% endraw %}

Replace
{% raw %}
~~~html
<div style="text-align:center"><img src ='{{"assets/Screenshot from 2017-12-29 22-46-33.png" | absolute_url}}' /></div>
~~~
{% endraw %}
with

{% raw %}
~~~html
<div style="text-align:center"><img src ='{{site.baseurl}}/assets/Screenshot from 2017-12-29 22-46-33.png' /></div>`
~~~
{% endraw %}
using:

``` sh
sed -i 's/"assets/site\.baseurl}}\/assets/g'
sed -i 's/"\s|\sabsolute_url}}//g'
```

## Notepad++

1. Shift highlighted lines to the right one tab length by pressing the tab key. 
2. Shift them to the left by pressing shift-tab.
3. Markdown render: [MarkdownViewerPlusPlus](https://github.com/nea/MarkdownViewerPlusPlus/releases)

## Visual Studio

To format a selection: `Ctrl`+`K`, `Ctrl`+`F`

To format a document: `Ctrl`+`K`, `Ctrl`+`D`

## PyCharm

Auto-indent code:

```
Ctrl A
Ctrl Alt L
```

## OBS

Screen recording.

## Vim

Add a '*' at the end of each line:

```sh
:%s/$/\*/g
```

## GDB with Eclipse

Add flags `-g -O0` to compilation. 

```
cat CmakeLists.txt
project(bfgs_mpc)

cmake_minimum_required (VERSION 3.5)


#Flags for compiler
IF(("${CMAKE_CXX_COMPILER_ID}" STREQUAL "GNU") OR ("${CMAKE_CXX_COMPILER_ID}" STREQUAL "Clang"))
    set(CMAKE_CXX_FLAGS "-std=c++11 -g -O0")
ENDIF()

```

Set build configuration (`build location`) for Eclipse:

<div style="text-align:center"><img src ='{{site.baseurl}}/assets/tools/Screenshot from 2018-04-28 15-07-49.png' /></div>

## Solve unresolved symbols XXX with Eclipse

1. Go to Project -> Properties -> C/C++ General -> Preprocessor Include Paths, Macros, etc. -> Providers -> CDT GCC built-in compiler settings, deactivate Use global provider shared between projects and add the command line argument -std=c++11.
2. Project -> C++ Index -> Rebuild

## Ubuntu switch display manager from lightdm to gdm


```sh
# configure gdm3
sudo dpkg-reconfigure gdm3
sudo service lightdm stop
sudo service gdm3 start
```

Install tweak to set details for gnome.

