---
date: '2025-12-02T08:27:00+03:30'
draft: true
title: 'Preprocessing and Data Augmentation'
description: "A Tutorial about the concept of preprocessing and data augmentation in Keras"
weight: 80
tags: [ "Deep-learning", "Keras", "PyTorch", "Python", "Kaggle", "Google-colab", "matplotlib", "Tensorboard" ]
image: "preprocessing-and-augmentation.webp"
code: "https://github.com/LiterallyTheOne/deep-learning-with-keras/blob/master/src/6-preprocessing-and-augmentation"
---

# Preprocessing and Augmentation

## Introduction

In the previous tutorial, we have learned about the basic layers used in CNNs.
In this tutorial, we are going to learn about **preprocessing** and **augmentation** layers in **Keras**.

> [Preprocessing layers in Keras](https://keras.io/api/layers/preprocessing_layers/)

## Preprocessing

**Data preprocessing** is a set of steps that we take before feeding the data to our model.
These steps help us to have clean, consistent, and meaningful inputs.
Also, they help the model to have a better accuracy, convergence speed, and generalization.
When we were loading our dataset, we used two transformations: `Resize` and `ToTensor`.
These two functions were related to the **PyTorch** and we were using them on the `ImageFolder`.
Now, we are going to learn about some preprocessing layers in **Keras**.

## Resizing

`Resizing` layer is a layer that resizes its input to match the given size.
Here is an example that we resized an image with the size of $1920 \times 1080$
to $224 \times 224$.

```python
from keras.layers import Resizing

resizing_layer = Resizing(224, 224)

input_image = np.random.randint(0, 256, (1, 1920, 1080, 3))

result_image = resizing_layer(input_image)

print(f"Input's shape: {input_image.shape}")
print(f"Result's shape: {result_image.shape}")

"""
--------
output: 

Input's shape: (1, 1920, 1080, 3)
Result's shape: torch.Size([1, 224, 224, 3])
"""
```

## Rescaling

`Rescaling` layer is a layer that rescales its input to the given scale.
In the example below, we have made a `Rescaling` layer with the scale of $\frac{1}{255}$.

```python
from keras.layers import Rescaling

rescaling_layer = Rescaling(1 / 255)

input_image = np.random.randint(0, 256, (1, 224, 224, 3))

result_image = rescaling_layer(input_image)

print(f"Input's max: {input_image.max()}")
print(f"Input's min: {input_image.min()}")
print(f"Result's max: {result_image.max()}")
print(f"Result's min: {result_image.min()}")

"""
--------
output: 

Input's max: 255
Input's min: 0
Result's max: 1.0
Result's min: 0.0
"""
```

## Augmentation

## RandomFlip

## RandomRotation

## RandomZoom

## RandomTranslation

## RandomContrast

## RandomBrightness

## RandomCrop

## Your turn

## Conclusion
