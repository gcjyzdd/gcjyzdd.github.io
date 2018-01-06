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
<ctrl-B>n

Close a window:
```
<ctrl-B>&
```

