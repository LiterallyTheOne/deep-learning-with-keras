---
date: '2025-10-18T08:59:00+03:30'
draft: false
title: 'Introduction'
description: "Introduction to Deep Learning with Keras"
weight: 10
tags: [ "Deep-learning", "Keras", "Python", "Kaggle" ]
image: "intro.webp"
code: "https://github.com/LiterallyTheOne/deep-learning-with-keras/blob/master/src/0-intro/a1-hello-world.ipynb"
---

# Introduction

## Keras

**Keras** is a high-level API for building and training **Deep Learning Models**.
It is designed to be a stand-alone project.
But with the help of **TensorFlow**, **PyTorch**, and **Jax**,
It can run on top of different hardware (e.g., **CPU**, **GPU**).

The fascinating thing about **Keras** is that it is super easy to get started with.
You can train and test a model with only a few lines of code.
It is a perfect way to learn **Deep Learning** concepts by practically seeing their effects.

## Google Colab

There are so many ways available to run a **Deep Learning** code.
One of the fastest and easiest way that doesn't require any installation, is **Google Colab**.
[Google colab](https://colab.research.google.com/) is a free could-based platform that
is powered by `jupyter notebook`.
All the packages that we want for this tutorial is already installed in **Google Colab**.
Also, every code that we run in this tutorial can be run on this platform.
So, I highly recommend you to start with **Google Colab**.
After you have become more comfortable with the packages and concepts,
switch to a local platform like your personal computer.

All the codes that we talk about in this tutorial is available in the **GitHub**.
Each tutorial has a link to its respective code, which you can find it at the bottom of each page.
To load and run the codes in **Google Colab**, you can follow these steps.

* Open [Google colab](https://colab.research.google.com/)
* From **files** select **Open Notebook**
* Go to the **GitHub** section
* Copy the **URL** of the code
* Select the **.ipynb** file that you want

Here is an example of loading this tutorial's code:

![Colab GitHub](colab-github.webp)

## Hello World

Here is a **Hello World** example that we are gradually going to complete it step by step.

```python
# Setup
import os

os.environ["KERAS_BACKEND"] = "torch"

# Imports
from keras.datasets import mnist
import keras
from keras import layers

# Prepare the Data
(train_images, train_labels), (test_images, test_labels) = mnist.load_data()

train_images = train_images.reshape((60000, 28 * 28))
train_images = train_images.astype("float32") / 255
test_images = test_images.reshape((10000, 28 * 28))
test_images = test_images.astype("float32") / 255

# Define the model
model = keras.Sequential([
    layers.Dense(512, activation="relu"),
    layers.Dense(10, activation="softmax")
])

model.compile(optimizer="adam",
              loss="sparse_categorical_crossentropy",
              metrics=["accuracy"])

# Train the model
model.fit(train_images, train_labels, epochs=5, batch_size=128)

# Test the model
test_loss, test_acc = model.evaluate(test_images, test_labels)

"""
--------
output: 

Epoch 1/5
469/469 ━━━━━━━━━━━━━━━━━━━━ 2s 5ms/step - accuracy: 0.9259 - loss: 0.2622
Epoch 2/5
469/469 ━━━━━━━━━━━━━━━━━━━━ 2s 5ms/step - accuracy: 0.9685 - loss: 0.1092
Epoch 3/5
469/469 ━━━━━━━━━━━━━━━━━━━━ 2s 5ms/step - accuracy: 0.9797 - loss: 0.0710
Epoch 4/5
469/469 ━━━━━━━━━━━━━━━━━━━━ 2s 5ms/step - accuracy: 0.9852 - loss: 0.0515
Epoch 5/5
469/469 ━━━━━━━━━━━━━━━━━━━━ 2s 5ms/step - accuracy: 0.9901 - loss: 0.0363
313/313 ━━━━━━━━━━━━━━━━━━━━ 1s 3ms/step - accuracy: 0.9801 - loss: 0.0616
"""

```

In the code above, we have trained and tested a model on a dataset called **MNIST**.
In the future, we are going deeper into each step, but for now, here is a simple explanation of each of them.
At first, we set up the backend of our **Keras**.
We set that to `torch`, but you can set that to either `tensorflow` or `jax`.
Then we imported the necessary modules.
After that, we downloaded
[MNIST](https://keras.io/api/datasets/mnist/).
[MNIST](https://keras.io/api/datasets/mnist/) contains of $28 \times 28$ images of handwritten
digits between $0$ and $9$.
Then, we normalize our data.
After that, we defined a simple model and compiled the model with the proper `optimizer`, `loss`, and `metrics`.
With the `fit` function, we train our model.
And finally, we test our model with the evaluate function.
As you can see in the output, our model's `accuracy` and `loss` are shown in each training step and testing step.
We have gotten `99%` accuracy on our training data and `98%` accuracy on our test data.