---
layout: post
date:   2018-01-04 16:29
title: "Handy Tools"
categories: Tools
---

# Download Youtube

[pytube](https://github.com/nficano/pytube): A lightweight, dependency-free Python library (and command-line utility) for downloading YouTube Videos.

```py
from pytube import YouTube
YouTube('https://www.youtube.com/watch?time_continue=25&v=siAMDK8C_x8').streams.first().download()
```

Install
```sh
pip install pytube
```

