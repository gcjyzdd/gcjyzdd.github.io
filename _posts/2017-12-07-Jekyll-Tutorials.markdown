---
layout: "post"
title:  "Jekyll Tutorials"
date:   2017-12-07 21:10 + 0100
categories: Linux Serive
author: "C. Guan"
permalink: /:year/:categories/:title.html
---

Create new project
------------------

Use `jekyll new` to create a new project:

```
jekyll new ga_blog
```

It creates several default files in the folder `ga_blog`. 


Serve the website
----------------------

At the first time, use the command

```
bundle exec jekyll serve
```

to serve the website. Later just use `jekyll serve` for short.

Subdirectories
---------------

And we can create subdirectories inside the `_post` folder to efficiently manage contents. Jekyll handles all the files inside `_post` automatically.

Drafts
------

Create a folder named `_drafts` under the root of the project. Files inside `_drafts` are not shown when serving the website. However, you can use `--draft` option to explictly show the drafts:

```
jekyll serve --draft
```

In addition, when wpermalink: "/my-post/post#1"riting a draft post, it's not necessary to proceed with the date in the file name.


New pages
----------

You can create new pages like `About me` by just placing a file inside the root directory of the project.

For example, 

```
touch donate.md

---
layout: "page"
title: Donate
---

Donate to our site.
```

Permalinks
------------

Defaulty, jekyll uses the date and category to create the url of pages. Thus, if you chage the date of your post later, the original url will be gone. You don't want this if someone else saved your post link. To change this, add `permalink` to the front matter:

```
---
permalink: "/my-post/post1"
---
```

We can also use variables to represent the url, for example 

```
---
permalink: "/:category/:month/:year/:title.html"
---

```

Front Matter Defaults
----------------------

Default front matter values can be set in the `_config.yml` file. For example,

```
defaults:
  -
     scope:
         path: ""  
         type: "posts"
     values:
         layout: "post"
         author: "C. Guan"
         permalink: "/:year/:categories/:title.html"
```

where `scope` specifies which those values are applied to.

Themes
----------

Search `jekyll-theme` in [rubygems.org](http://rubygems.org) to select your favoriate theme.

For example, in Gemfile, append

```
gem "jekyll-theme-hacker"
```

And run `bundle install` to install the theme. And run `bundle exec jekyll serve` to serve the site.

## Layout

Create folder `_layouts`. Overwrite `post` by creating `post.html` in the folder. 

```
<h1>This is a post</h1>
<hr>

{{content}}

```


