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

**Data Augmentation** is a technique in machine learning that artificially expands our training dataset by applying
different transformations.
**Data Augmentation** is extremely useful when we don't have enough data or our data is not balanced.
It helps us with the generalization and prevents the model from overfitting.
We have so many different **augmentation** techniques for different use-cases.
Let's get to know how to use some of them in **Keras**.
You can see the output of all the examples in
[this notebook](https://github.com/LiterallyTheOne/deep-learning-with-keras/blob/main/src/6-preprocessing-and-augmentation/a2-augmentation.ipynb)

## RandomFlip

`RandomFlip`, technically, has a $50%$ chance to flip its input in the given mode.
Modes can be:

* `horizontal`
* `vertical`
* `horizontal_and_vertical`

Here is an example that only flips horizontally:

```python
from keras.layers import RandomFlip

random_flip_layer = RandomFlip("horizontal")

```

## RandomRotation

`RandomRotation` rotates its input with the given factor.
The range of the rotation would be: $[-factor * \pi, +factor * \pi]$  
For example, if we put the factor to $0.2$, it would rotate the input in the range of
$$[-0.2 * \pi, +0.2 * \pi]
= [-0.2 * 180^\circ, 0.2 * 180^\circ]
=\boxed{[-36^\circ, 36^\circ]}
$$
Here is an example of this layer:

```python
from keras.layers import RandomRotation

random_rotation_layer = RandomRotation(0.2)

```

## RandomZoom

`RandomZoom` zooms in or out respect to the `height_factor` and `width_factor`.
Here is an example of this layer:

```python
from keras.layers import RandomZoom

random_zoom_layer = RandomZoom(0.4, 0.2)

```

## RandomTranslation

`RandomZoom` moves the image respect to the `height_factor` and `width_factor`.
Here is an example of this layer:

```python
from keras.layers import RandomTranslation

random_translation_layer = RandomTranslation(0.2, 0.2)

```

## RandomContrast

`RandomContrast` changes the contrast respect to the given `factor`.
Here is an example of this layer:

```python
from keras.layers import RandomContrast

random_contrast_layer = RandomContrast(0.4)
```

## RandomBrightness

## RandomCrop

## Your turn

## Conclusion
