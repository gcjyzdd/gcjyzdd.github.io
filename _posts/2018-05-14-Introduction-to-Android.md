---
layout: post
date:   2018-05-14 21:01
categories: Android
---

<script src="https://cdnjs.cloudflare.com/ajax/libs/mathjax/2.7.0/MathJax.js?config=TeX-AMS-MML_HTMLorMML" type="text/javascript"></script>

## Building Layout Part 1

### TextView

Attributes:
 * text
 * background color. `Hex Colors`.
 * text color
 * layout width, height. `dp` density-independent pixles.
 * text font size. `sp` scale-invariant pixel

Example:

```xml
<TextView
  android:text="I got it"
  android:background="#9C27B0"
  android:textColor="#DCE775"
  android:layout_width="wrap_content"
  android:layout_height="wrap_content"
  android:textSize="45sp" />
```

### Simple ImageView

```xml
<ImageView
  android:src="@drawable/cake"
  android:layout_width="wrap_content"
  android:layout_height="wrap_content"
  android:scaleType="center"/>
```
where `drawable` is the image folder.

**scale type**: `centerCrop`.

## Building Layout Part 2

### ViewGroups

<div style="text-align:center"><img src ='{{site.baseurl}}/assets/android/intro/Screenshot from 2018-05-14 22-20-23.png' /></div>


* ViewGroups
* Root View
* Parent
* Child
* Sibling

<div style="text-align:center"><img src ='{{site.baseurl}}/assets/android/intro/Screenshot from 2018-05-14 22-24-46.png' /></div>

Quiz:

<div style="text-align:center"><img src ='{{site.baseurl}}/assets/android/intro/Screenshot from 2018-05-14 22-28-16.png' /></div>

### Types of ViewGroups

**LinearLayout**:

* Verticle column
* Horizontal row

**RelativeLayout**:

* Relative to parent
* Relative to other children

```xml
<LinearLayout
    android:orientation="vertical"
    android:layout_width="wrap_content"
    android:layout_height="wrap_content">
 
    <TextView
        android:text="Guest List"
        android:layout_width="wrap_content"
        android:layout_height="wrap_content" />
 
    <TextView
        android:text="Kunal"
        android:layout_width="wrap_content"
        android:layout_height="wrap_content" />
 
</LinearLayout>
```

<div style="text-align:center"><img src ='{{site.baseurl}}/assets/android/intro/Screenshot from 2018-05-14 22-34-37.png' /></div>

### LinearLayout

Namespace declaration.

Quiz:

<div style="text-align:center"><img src ='{{site.baseurl}}/assets/android/intro/Screenshot from 2018-05-14 22-46-33.png' /></div>

Code:

```xml
<LinearLayout
    xmlns:android="http://schemas.android.com/apk/res/android"
    android:orientation="vertical"
    android:background="#FF0000"
    android:layout_width="wrap_content"
    android:layout_height="wrap_content">

    <TextView
        android:text="Guest List"
        android:layout_width="wrap_content"
        android:layout_height="wrap_content"
        android:textColor="#FFFFFF"
        android:textSize="24sp"  />

    <TextView
        android:text="Kunal"
        android:layout_width="wrap_content"
        android:layout_height="wrap_content"
        android:textColor="#FFFFFF"
        android:textSize="24sp"  />
    
    <TextView
        android:text="Ted"
        android:layout_width="wrap_content"
        android:layout_height="wrap_content"
        android:textColor="#FFFFFF"
        android:textSize="24sp"  />

    <LinearLayout
        android:orientation="horizontal"
    	android:background="#00FF00"
    	android:layout_width="wrap_content"
    	android:layout_height="wrap_content">
        
        <TextView
        android:text="Ted"
        android:layout_width="wrap_content"
        android:layout_height="wrap_content"
        android:textColor="#FFFFFF"
        android:textSize="24sp"  />
        
        <TextView
        android:text="Marcel"
        android:layout_width="wrap_content"
        android:layout_height="wrap_content"
        android:textColor="#FFFFFF"
        android:textSize="24sp"  />
    
    
    </LinearLayout>
    
    <TextView
        android:text="Ted"
        android:layout_width="wrap_content"
        android:layout_height="wrap_content"
        android:textColor="#FFFFFF"
        android:textSize="24sp"  />
    
</LinearLayout>
```

### Width and Height

Three ways:
* Fixed dp
* wrap_content
* **Match parent**

<div style="text-align:center"><img src ='{{site.baseurl}}/assets/android/intro/Screenshot from 2018-05-14 22-50-00.png' /></div>

<div style="text-align:center"><img src ='{{site.baseurl}}/assets/android/intro/Screenshot from 2018-05-14 22-50-58.png' /></div>

### Evenly Spacing Out Children

**layout_weight**

### Layout Weight

Hangouts:

<div style="text-align:center"><img src ='{{site.baseurl}}/assets/android/intro/Screenshot from 2018-05-14 23-15-25.png' /></div>

Maps:

<div style="text-align:center"><img src ='{{site.baseurl}}/assets/android/intro/Screenshot from 2018-05-14 23-16-57.png' /></div>

Gmail:

<div style="text-align:center"><img src ='{{site.baseurl}}/assets/android/intro/Screenshot from 2018-05-14 23-18-19.png' /></div>

Quiz:

<div style="text-align:center"><img src ='{{site.baseurl}}/assets/android/intro/Screenshot from 2018-05-14 23-21-51.png' /></div>

### Relative Layout

<div style="text-align:center"><img src ='{{site.baseurl}}/assets/android/intro/Screenshot from 2018-05-14 23-24-12.png' /></div>

Relative to parent:

<div style="text-align:center"><img src ='{{site.baseurl}}/assets/android/intro/Screenshot from 2018-05-14 23-26-40.png' /></div>



