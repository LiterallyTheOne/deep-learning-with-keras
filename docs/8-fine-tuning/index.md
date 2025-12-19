---
date: '2025-11-26T08:35:00+03:30'
draft: True
title: "Fine-tuning"
description: "A tutorial about how to apply fine-tuning"
weight: 90
tags: [ "Deep-learning", "Keras", "PyTorch", "Python", "Kaggle", "Google-colab", "matplotlib", "Tensorboard" ]
image: "fine-tuning.webp"
code: "https://github.com/LiterallyTheOne/deep-learning-with-keras/blob/master/src/8-fine-tuning"
---

# Fine-tuning

## Introduction

In the previous tutorials, we learned about transfer learning and how to use the advantage of
a pre-trained model on our dataset.
In this Tutorial, we learn about how to apply fine-tuning and see the results.
Also, we introduce how to use other platforms like [Kaggle models](https://www.kaggle.com/models/)
to access to new models.

## Fine-tuning

**Fine-tuning** is a technique in **Deep Learning** that we use to adapt our pretrained model with the
new **Dataset**.
In the previous tutorials, we worked with **transfer learning**.
**Fine-tuning** is pretty similar to **transfer learning**.
The only difference is that we unfreeze some of the last layers to our **base_model** in order to train them.
Here is an example:

```python
base_model = MobileNetV2(include_top=False, input_shape=(224, 224, 3))

for layer in base_model.layers[:-5]:
    layer.trainable = False
```

In the code above, we froze the starting layers of our `base_model` and left the last $5$ layers as `trainable`.
Now, let's print the `base_model` summary with `show_trainable=True` like below:

```python
print(base_model.summary(show_trainable=True))

"""
--------
output: 

"""

```

As you can see, the last $5$ layers, are trainable.
The only thing that we should do, is to train our model like before.

## Kaggle models

## Share your model

## Your turn

## Conclusion
