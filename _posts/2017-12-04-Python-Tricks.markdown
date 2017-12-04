---
layout: post
title:  "Python Tricks"
date:   2017-12-04 22:20
categories: Python Numpy
---

### Table of Contents
[Numpy](#Numpy)

# Numpy

## Transform a row to a column

Transform a row to a column:

```
import numpy as np


# a row
a = np.array([1, 2, 3])

# transform a to a column vector
a[:, None]
```

