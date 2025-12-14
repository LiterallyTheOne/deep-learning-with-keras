---
date: '2025-12-14T09:18:00+03:30'
draft: true
title: 'Callbacks'
description: "A tutorial about the callbacks in Keras, including Early Stopping and Model Checkpoint"
weight: 80
tags: [ "Deep-learning", "Keras", "PyTorch", "Python", "Kaggle", "Google-colab", "matplotlib", "Tensorboard" ]
image: "callbacks.webp"
code: "https://github.com/LiterallyTheOne/deep-learning-with-keras/blob/master/src/7-callbacks"
---

# Callbacks

## Introduction

In the previous tutorial, we have learned about **preprocessing and data augmentation** techniques in Keras.
In this tutorial, we learn about **Callbacks** and explore some of the most important ones in **Keras**.

## Callbacks

**Callback** in **Keras** is a function that we pass it to our **fit** function.
**Keras** calls that function automatically in the specific moment.
We have learned about **TensorBoard Callback** before.
We used to create a **TensorBoard Callback** and pass it to our fit function as below:

```python
log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
tensorboard_callback = keras.callbacks.TensorBoard(log_dir=log_dir)

history = model.fit(
    ...,
    callbacks=[tensorboard_callback],
)
```

In this tutorial, we are going to learn about another two important **CallBacks**,
called: **EarlyStopping** and **ModelCheckpoint**.

## Early Stopping

## Model Checkpoint

## Your turn

## Conclusion
