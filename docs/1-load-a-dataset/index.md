---
date: '2025-10-25T10:06:00+03:30'
draft: false
title: 'Load an Image Classification Dataset'
description: "A tutorial about how to load an Image Classification Dataset"
weight: 20
tags: [ "Deep-learning", "Keras", "PyTorch", "Python", "Kaggle", "Google-colab" ]
image: "load-a-dataset.webp"
code: "https://github.com/LiterallyTheOne/deep-learning-with-keras/blob/master/src/1-load-a-dataset/a1-load-a-dataset.ipynb"
---

# Load an Image Classification Dataset

# Introduction

In the previous tutorial, we learned how about **Keras**, **Google Colab**, and **Kaggle**.
Our task was to select an **Image Classification Dataset** from **Kaggle**.
In this tutorial, we are going to load this dataset and make a ready to give it to a model.

## Get data from Kaggle

The easiest and the recommended way to download a dataset from **Kaggle** is to use a package called **Kagglehub**.
**Kaggle** itself has developed this package and made it super easy to use.
You can learn more about this package in their
[GitHub Repository](https://github.com/Kaggle/kagglehub).

Now, how to use this package to download a dataset.
In the dataset that you have selected, click on the **Download** button in the top right corner of the page.
A window will pop up that has a code snippet on it.
You should copy that code and use it in your own code.
For [Tom and Jerry Image classification](https://www.kaggle.com/datasets/balabaskar/tom-and-jerry-image-classification),
the is like this:

```python
import kagglehub

# Download latest version
path = kagglehub.dataset_download("balabaskar/tom-and-jerry-image-classification")

print("Path to dataset files:", path)
```

The code above, will automatically download the dataset and returns its path.
We said that we wanted a structure like below:

```text
class_a/
...a_image_1.jpg
...a_image_2.jpg
class_b/
...b_image_1.jpg
...b_image_2.jpg
```

We know that this dataset has this structure and if you looked at the dataset in **Kaggle**, you have noticed that
it is in `tom_and_jerry/tom_and_jerry` directory.
But get more familiar with the **jupyter notebook** commands, let's find it with taking the list of the path that we
are currently on.

```shell
!ls {path}


"""
--------
output: 

challenges.csv   ground_truth.csv tom_and_jerry
"""
```

As you can see, we have `tom_and_jerry` directory.
Now, let's take the list of this directory.

```shell
!ls {path}/tom_and_jerry


"""
--------
output: 

tom_and_jerry
"""
```

As you can see, we have another `tom_and_jerry` directory.
Let's take the list of it to see what's inside of it.

```shell
!ls {path}/tom_and_jerry/tom_and_jerry


"""
--------
output: 

jerry       tom         tom_jerry_0 tom_jerry_1
"""
```

And as you can see, we have reached to the structure that we wanted.
Let's put this path in a variable called `data_path`, to be able to use it later.

```python
from pathlib import Path

data_path = Path(path) / "tom_and_jerry/tom_and_jerry"
```

Your dataset might have subdirectories like `train`, `validation` and `test`.
If it was like this put the `train` directory in the `data_path` and store the other ones in their respective directory.
For example, `val_path` for validation and `test_path` for test.