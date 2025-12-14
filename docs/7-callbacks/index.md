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
from keras.callbacks import TensorBoard

log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
tensorboard_callback = TensorBoard(log_dir=log_dir)

history = model.fit(
    ...,
    callbacks=[tensorboard_callback],
)

```

In this tutorial, we are going to learn about another two important **CallBacks**,
called: **EarlyStopping** and **ModelCheckpoint**.

## Early Stopping

**EarlyStopping** is a callback that stops the training procedure if there is no improvement.
Here is an example of **EarlyStopping**:

```python
from keras.callbacks import EarlyStopping

early_stopping = EarlyStopping(
    monitor="val_loss",
    patience=5,
    verbose=1,
)


history = model.fit(
    ...,
    epochs=200,
    callbacks=[tensorboard_callback, early_stopping],
)

```

In the code above, we have created an object of **EarlyStopping**.
We set our **EarlyStopping** to monitor our validation loss (`"val_loss"`).
Then, we told it to wait for 5 epochs, if there was no improvement seen on those epochs, stop the training.
Also, we set the `verbose` to `1`, to be able to have a report of the procedure.
As you can see, we have added it to our callbacks argument in `fit` function and increased our `epochs` to `200`.

## Model Checkpoint

## Your turn

## Conclusion
