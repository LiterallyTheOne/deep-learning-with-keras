---
date: '2025-11-18T10:06:00+03:30'
draft: false
title: 'Plot and Tensorboard'
description: "A tutorial about how to plot our training history using matplotlib and Tensorboard"
weight: 40
tags: [ "Deep-learning", "Keras", "PyTorch", "Python", "Kaggle", "Google-colab", "matplotlib", "Tensorboard" ]
image: "plot-and-tensorboard.webp"
code: "https://github.com/LiterallyTheOne/deep-learning-with-keras/blob/master/src/3-plot-and-tensorboard/a1-plot-and-tensorboard.ipynb"
---

# Plot and TensorBoard

## Introduction

In the previous tutorial, we learned about **model** and **Transfer Learning**.
Here is the summary of the code that we have implemented so far.

```python
import os

os.environ["KERAS_BACKEND"] = "torch"
from pathlib import Path

from matplotlib import pyplot as plt

import torch
from torch.utils.data import random_split, DataLoader

from torchvision.datasets import ImageFolder
from torchvision import transforms

import keras
from keras import layers
from keras.applications import MobileNetV2

import kagglehub

import datetime

# Load the Dataset

path = kagglehub.dataset_download("balabaskar/tom-and-jerry-image-classification")

data_path = Path(path) / "tom_and_jerry/tom_and_jerry"

trs = transforms.Compose(
    [
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ]
)

all_data = ImageFolder(data_path, transform=trs)

g1 = torch.Generator().manual_seed(20)
train_data, val_data, test_data = random_split(all_data, [0.7, 0.2, 0.1], g1)

train_loader = DataLoader(train_data, batch_size=12, shuffle=True)
val_loader = DataLoader(val_data, batch_size=12, shuffle=False)
test_loader = DataLoader(test_data, batch_size=12, shuffle=False)

# Create the model

base_model = MobileNetV2(include_top=False, input_shape=(224, 224, 3))

base_model.trainable = False

model = keras.Sequential(
    [
        layers.Input(shape=(3, 224, 224)),
        layers.Permute((2, 3, 1)),
        base_model,
        layers.Flatten(),
        layers.Dense(4, activation="softmax"),
    ]
)

model.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"],
)

# Train the model

history = model.fit(train_loader, epochs=5, validation_data=val_loader)

# Evaluate the model

loss, accuracy = model.evaluate(test_loader)

print("loss:", loss)
print("accuracy:", accuracy)
```


As you can see, in the code above, when we were training our model using `.fit` function, we were storing its result in a variable
called `history`.
In this tutorial, we will learn more about `history` and how to plot its results.
Also, we will learn about a very powerful tool for plotting and seeing the results during training,
called **TensorBoard**.

## Your turn

## Conclusion
