---
layout: post
date:   2018-05-14 21:01
categories: Android
---

<script src="https://cdnjs.cloudflare.com/ajax/libs/mathjax/2.7.0/MathJax.js?config=TeX-AMS-MML_HTMLorMML" type="text/javascript"></script>

## Lesson 1: Building Layout Part 1

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

## Lesson 2: Building Layout Part 2

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

* layout_alignParentTop
* layout_alignParentBottom
* layout_alignParentLeft
* layout_alignParentRight
* layout_centerHorizontal
* layout_centerVertical

<div style="text-align:center"><img src ='{{site.baseurl}}/assets/android/intro/Screenshot from 2018-05-14 23-26-40.png' /></div>

### Relative Layout Quiz

```xml
<RelativeLayout
       xmlns:android="http://schemas.android.com/apk/res/android"
       android:layout_width="match_parent"
       android:layout_height="match_parent"
       android:padding="16dp">
 
   <TextView
        android:text="I’m in this corner"
        android:layout_height="wrap_content"
        android:layout_width="wrap_content"
        android:layout_alignParentBottom="true"
        android:layout_alignParentLeft="true" />
 
    <TextView
        android:text="No, up here"
        android:layout_height="wrap_content"
        android:layout_width="wrap_content"
        android:layout_alignParentTop="true"
        android:layout_alignParentLeft="true" />
 
    <TextView
        android:text="Wait, I’m here"
        android:layout_height="wrap_content"
        android:layout_width="wrap_content"
        android:layout_alignParentBottom="true"
        android:layout_alignParentRight="true" />
 
    <TextView
        android:text="Actually, I’m here"
        android:layout_height="wrap_content"
        android:layout_width="wrap_content"
        android:layout_alignParentTop="true"
        android:layout_alignParentRight="true" />
 
</RelativeLayout>
```

<div style="text-align:center"><img src ='{{site.baseurl}}/assets/android/intro/Screenshot from 2018-05-15 20-44-16.png' /></div>

### Relative to Other Views

<div style="text-align:center"><img src ='{{site.baseurl}}/assets/android/intro/Screenshot from 2018-05-15 20-47-50.png' /></div>

<div style="text-align:center"><img src ='{{site.baseurl}}/assets/android/intro/Screenshot from 2018-05-15 20-50-29.png' /></div>

**Quiz**:

```xml
<RelativeLayout
    xmlns:android="http://schemas.android.com/apk/res/android"
    android:layout_width="match_parent"
    android:layout_height="match_parent">

    <TextView
        android:id="@+id/lyla_text_view"
        android:layout_width="wrap_content"
        android:layout_height="wrap_content"
        android:layout_alignParentBottom="true"
        android:layout_alignParentLeft="true"
        android:textSize="24sp"
        android:text="Lyla" />

    <TextView
        android:id="@+id/me_text_view"
        android:layout_width="wrap_content"
        android:layout_height="wrap_content"
        android:layout_alignParentBottom="true"
        android:layout_toRightOf="@id/lyla_text_view"
        android:textSize="24sp"
        android:text="Me" />

    <TextView
        android:id="@+id/natalie_text_view"
        android:layout_width="wrap_content"
        android:layout_height="wrap_content"
        android:layout_above="@id/lyla_text_view"
        android:textSize="24sp"
        android:text="Natalie" />

    <TextView
        android:id="@+id/jennie_text_view"
        android:layout_width="wrap_content"
        android:layout_height="wrap_content"
        android:layout_alignParentBottom="true"
        android:layout_alignParentRight="true"
        android:textSize="24sp"
        android:text="Jennie" />

    <TextView
        android:id="@+id/omoju_text_view"
        android:layout_width="wrap_content"
        android:layout_height="wrap_content"
        android:layout_alignParentRight="true"
        android:layout_above="@id/jennie_text_view"
        android:textSize="24sp"
        android:text="Omoju" />

    <TextView
        android:id="@+id/amy_text_view"
        android:layout_width="wrap_content"
        android:layout_height="wrap_content"
        android:layout_alignParentRight="true"
        android:layout_above="@id/omoju_text_view"
        android:textSize="24sp"
        android:text="Amy" />

    <TextView
        android:id="@+id/ben_text_view"
        android:layout_width="wrap_content"
        android:layout_height="wrap_content"
        android:layout_alignParentTop="true"
        android:layout_centerHorizontal="true"
        android:textSize="24sp"
        android:text="Ben" />

    <TextView
        android:id="@+id/kunal_text_view"
        android:layout_width="wrap_content"
        android:layout_height="wrap_content"
        android:layout_alignParentTop="true"
        android:layout_toLeftOf="@id/ben_text_view"
        android:textSize="24sp"
        android:text="Kunal" />

    <TextView
        android:id="@+id/kagure_text_view"
        android:layout_width="wrap_content"
        android:layout_height="wrap_content"
        android:layout_alignParentTop="true"
        android:layout_toRightOf="@id/ben_text_view"
        android:textSize="24sp"
        android:text="Kagure" />

</RelativeLayout>
```

<div style="text-align:center"><img src ='{{site.baseurl}}/assets/android/intro/Screenshot from 2018-05-15 20-57-56.png' /></div>

[Documentation](https://developer.android.com/reference/android/widget/RelativeLayout.LayoutParams?utm_source=udacity&utm_medium=course&utm_campaign=android_basics)

### List Item with RelativeLayout

<div style="text-align:center"><img src ='{{site.baseurl}}/assets/android/intro/Screenshot from 2018-05-15 21-05-07.png' /></div>

```xml
<RelativeLayout
    xmlns:android="http://schemas.android.com/apk/res/android"
    android:layout_width="match_parent"
    android:layout_height="match_parent"
    android:orientation="vertical" >

    <ImageView
        android:id="@+id/image_view"
        android:layout_width="56dp"
        android:layout_height="56dp"
        android:scaleType="centerCrop"
        android:layout_alignParentTop="true"
        android:layout_alignParentLeft="true"
        android:src="@drawable/ocean" />

    <TextView
        android:id="@+id/text1"
        android:layout_width="wrap_content"
        android:layout_height="wrap_content"
        android:text="Pebble Beach"
        android:layout_alignParentTop="true"
        android:layout_toRightOf="@id/image_view"
        android:textAppearance="?android:textAppearanceMedium" />

    <TextView
        android:id="@+id/text2"
        android:layout_width="wrap_content"
        android:layout_height="wrap_content"
        android:text="California"
        android:layout_toRightOf="@id/image_view"
        android:layout_below="@id/text1"
        android:textAppearance="?android:textAppearanceSmall" />

    <TextView
        android:layout_width="wrap_content"
        android:layout_height="wrap_content"
        android:text="10 miles away"
        android:layout_toRightOf="@id/image_view"
        android:layout_below="@id/text2"
        android:textAppearance="?android:textAppearanceSmall" />

</RelativeLayout>
```

### Padding vs. Margin

<div style="text-align:center"><img src ='{{site.baseurl}}/assets/android/intro/Screenshot from 2018-05-15 21-11-08.png' /></div>

<div style="text-align:center"><img src ='{{site.baseurl}}/assets/android/intro/Screenshot from 2018-05-15 21-12-10.png' /></div>

<div style="text-align:center"><img src ='{{site.baseurl}}/assets/android/intro/Screenshot from 2018-05-15 21-13-33.png' /></div>

<div style="text-align:center"><img src ='{{site.baseurl}}/assets/android/intro/Screenshot from 2018-05-15 21-18-10.png' /></div>

<div style="text-align:center"><img src ='{{site.baseurl}}/assets/android/intro/Screenshot from 2018-05-15 21-20-53.png' /></div>

**Quiz**:

<div style="text-align:center"><img src ='{{site.baseurl}}/assets/android/intro/Screenshot from 2018-05-15 21-31-12.png' /></div>

```xml
<LinearLayout
    xmlns:android="http://schemas.android.com/apk/res/android"
    android:layout_width="match_parent"
    android:layout_height="match_parent"
    android:orientation="vertical">

    <ImageView
        android:src="@drawable/ocean"
        android:layout_width="match_parent"
        android:layout_height="0dp"
        android:layout_weight="1"
        android:scaleType="centerCrop"/>

    <TextView
        android:text="You're invited!"
        android:layout_width="match_parent"
        android:layout_height="wrap_content"
        android:textColor="@android:color/white"
        android:textSize="45sp"
        android:paddingLeft="16dp"
        android:paddingTop="16dp"
        android:background="#009688"/>

    <TextView
        android:text="Bonfire at the beach"
        android:layout_width="match_parent"
        android:layout_height="wrap_content"
        android:textColor="@android:color/white"
        android:textSize="24sp"
        android:paddingLeft="16dp"
        android:paddingTop="8dp"
        android:paddingBottom="16dp"
        android:background="#009688"/>

</LinearLayout>
```

## Lesson 3: Practice Set

### Discussion about Constraint Layout

Before starting on the next video, we want to bring to your attention a new feature in new Android Studio - Constraint Layout - that may require you to make some code modifications to the steps you see in the following videos.

**Keeping up with the changes**

Google is constantly improving the Android platform and adding new features. This is great for you as a developer, but it makes learning harder sometimes. Recently Google released `ConstraintLayout`; a tool that makes it super fast to create responsive UIs with many different types of components. `ConstraintLayout` is a great tool to have on the sleeve, but for this class, we will use `RelativeLayout`, `LinearLayout simpler`.

All of this matters to you because the new project templates in Android Studio now use ConstraintLayout as default, which makes the code you see on your computer a bit different from what is on the screen.

**Current Layout File**

In the new versions of Android Studio, after choosing the `Empty Activity` template, the layout file `app/src/main/res/layout/activity_main.xml` will look like this:

```xml
<?xml version="1.0" encoding="utf-8"?>
<android.support.constraint.ConstraintLayout
    xmlns:android="http://schemas.android.com/apk/res/android"
    xmlns:app="http://schemas.android.com/apk/res-auto"
    xmlns:tools="http://schemas.android.com/tools"
    android:id="@+id/activity_main"
    android:layout_width="match_parent"
    android:layout_height="match_parent"
    tools:context="com.udacity.myapplication.MainActivity">

    <TextView
        android:layout_width="wrap_content"
        android:layout_height="wrap_content"
        android:text="Hello World!"
        app:layout_constraintLeft_toLeftOf="@+id/activity_main"
        app:layout_constraintTop_toTopOf="@+id/activity_main"
        app:layout_constraintRight_toRightOf="@+id/activity_main"
        app:layout_constraintBottom_toBottomOf="@+id/activity_main" />

</android.support.constraint.ConstraintLayout>
```

Note the use of `ConstraintLayout`, and that `TextView` has a list of limiters that position it within `ConstraintLayout`.

**Modify the Layout File**

Unlike the above code, our videos and start code assume that the template looks more like the following, using as the root of the view a `RelativeLayout`:

```xml
<?xml version="1.0" encoding="utf-8"?>
<RelativeLayout
    xmlns:android="http://schemas.android.com/apk/res/android"
    xmlns:tools="http://schemas.android.com/tools"
    android:layout_width="match_parent"
    android:layout_height="match_parent"
    android:paddingBottom="@dimen/activity_vertical_margin"
    android:paddingLeft="@dimen/activity_horizontal_margin"
    android:paddingRight="@dimen/activity_horizontal_margin"
    android:paddingTop="@dimen/activity_vertical_margin"
    tools:context="com.udacity.myapplication.MainActivity">

    <TextView
        android:layout_width="wrap_content"
        android:layout_height="wrap_content"
        android:text="Hello World!" />
</RelativeLayout>
```

When you create your new project, go to `app/src/main/res/layout/activity_main.xml` and copy and paste the above code. Then you're ready to go!

**Learn More About Constraint Layout**

If you want to understand more about the great features that `ConstraintLayout` provides, check out the documentation at: [https://developer.android.com/studio/write/layout-editor.html](https://developer.android.com/studio/write/layout-editor.html)

Additionally, for those wanting a hands-on demo using Android Studio Layout Editor with ConstraintLayout, here's a [Code Lab](https://codelabs.developers.google.com/codelabs/constraint-layout/index.html?index=..%2F..%2Findex#5). Note that this is important information, but is beyond the current scope of this course. 

### RelativeLayout LinearLayout

Steps:

1. Select the views
2. Position the views
3. Style the views

<div style="text-align:center"><img src ='{{site.baseurl}}/assets/android/intro/Screenshot from 2018-05-15 22-16-59.png' /></div>

<div style="text-align:center"><img src ='{{site.baseurl}}/assets/android/intro/Screenshot from 2018-05-15 22-17-40.png' /></div>

<div style="text-align:center"><img src ='{{site.baseurl}}/assets/android/intro/Screenshot from 2018-05-15 22-20-41.png' /></div>











