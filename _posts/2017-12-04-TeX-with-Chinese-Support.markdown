---
layout: post
title:  "TeX with Chinese support"
date:   2017-12-04
categories: TeX
---

Use TeX with Chinese Support
-----------------------------

The simple way to use Chinese is to use the `ctex` package and encode the text with UTF8.

The following is an example:


```
\documentclass{letter}

\usepackage[UTF8]{ctex}

\begin{document}
测试
\end{document}
```

On Ubuntu system with texlive installed, this could cause an error if compile with `pdflatex`. To solve this error, compile tex file with `xelatex`.

Below is the output display:

<div style="text-align:center"><img src="{{site.baseurl}}/assets/tex_with_chinese.png" /></div>

![Output]({{site.baseurl}}/assets/tex_with_chinese.png)
