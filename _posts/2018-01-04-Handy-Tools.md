---
layout: post
date:   2018-01-04 16:29
title: "Handy Tools"
categories: Tools
---

# Download Youtube

[pytube](https://github.com/nficano/pytube): A lightweight, dependency-free Python library (and command-line utility) for downloading YouTube Videos.

## Usage

```py
from pytube import YouTube
YouTube('https://www.youtube.com/watch?time_continue=25&v=siAMDK8C_x8').streams.first().download()
```

## Install
```sh
pip install pytube
```
---
# PDF

## Combine PDFs

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

## Crop PDF

---
# tmux

## Basic Usage

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
# Crop Video using FFMPEG

Use the crop filter:
```
ffmpeg -i in.mp4 -filter:v "crop=out_w:out_h:x:y" out.mp4
```

Where the options are as follows:

* `out_w` is the width of the output rectangle
* `out_h` is the height of the output rectangle
* `x` and `y` specify the top left corner of the output rectangle

[Ref](https://video.stackexchange.com/questions/4563/how-can-i-crop-a-video-with-ffmpeg)


# sed - replace file in place

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

