---
layout: post
date:   2018-03-08 13:06
categories: Matlab Simulink
title: Matlab/Simulink Notes
---

## Simulink Models

### Probe

<div style="text-align:center"><img width='50%' src ='{{site.baseurl}}/assets/tass/probe_block.png' /></div>

### Selector

<div style="text-align:center"><img width='100%' src ='{{site.baseurl}}/assets/tass/selector.png' /></div>

### Rate Limiter

<div style="text-align:center"><img width='100%' src ='{{site.baseurl}}/assets/tass/rate_limiter.png' /></div>

### Detect Change

<div style="text-align:center"><img width='100%' src ='{{site.baseurl}}/assets/tass/detect_change.png' /></div>


### Interpreted Matlab Function

<div style="text-align:center"><img width='100%' src ='{{site.baseurl}}/assets/tass/interpreted_fcn.png' /></div>

## Warnings

Dsiable warning of increasing array size in loop:

```matlab
output_postfix =[num2str(plot_idx) '_' output_postfix]; %#ok<AGROW>
```

## Matlab

### Enumeration to char/string

```matlab
>> a=char(report.processFuns.TSR_ENUM_GT.Electronic5)
a =
Electronic5
>> class(a)
ans =
char
```

### Stop debug mode

```
dbquit
dbquit all
```

