---
date: '2025-11-12T10:06:00+03:30'
draft: false
title: 'Model and Transfer Learning'
description: "A tutorial about how we can define a model and use transfer learning on image classification datasets"
weight: 30
tags: [ "Deep-learning", "Keras", "PyTorch", "Python", "Kaggle", "Google-colab" ]
image: "model-and-transfer-learning.webp"
code: "https://github.com/LiterallyTheOne/deep-learning-with-keras/blob/master/src/2-model-and-transfer-learning/a1-model-and-transfer-learning.ipynb"
---

# Model and Transfer Learning

## Introduction

In the previous tutorial, we have loaded our selected **Kaggle** dataset into **train**, **validation**, and **test**
subsets.
Then, we have made a `DataLoader` for each subset.
The summary of our code looks like below:

```python
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
```

As you might have noticed, we only changed the scale of our `Resize` transform to (224, 224) to 
make it one of the standard sizes for images in deep learning.
In this tutorial, we will learn about how to define a model in **Keras**.
Then, we will improve our results using a technique called **Transfer Learning**

## Model in Keras

There are 3 ways to define a model in **Keras**.

* Sequential
* Functional
* Subclassing


All these three ways have their own use-cases.
`Sequential` is one of the cleanest and best ways of defining a model which we will
learn it in this session.
As the name suggests, it would take a sequence of layers.
Then, pass the data through them in order and generate the output.
To define it in **Keras**, we can use this code.

```python
keras.Sequential(
    [
    ],
)
```

It requires a list of layers, which we are going to talk about them very shortly.

> **Sources**:
> * <https://keras.io/api/models/model/>
> * <https://keras.io/api/models/sequential/>
> * <https://keras.io/guides/sequential_model/> 

## Input layer

**Input layer** is the layer that we use to tell **Keras** what the shape of our input is.
For example, for the shape of `(3, 224, 224)`, we can
use the code below:

```python
input_layer = keras.layers.Input(shape=(3, 224, 224))
```

So let's add it to our sequential model: 

```python
model = keras.Sequential(
    [
        keras.layers.Input(shape=(3, 224, 224)),
    ],
)
```

## Dense layer

`Dense layer` (fully connected layer) is a layer that
all the neurons of this layer is connected to the neurons
of the previous layer.
To define a `Dense layer` in `Keras` we can simpy use
`keras.layers.Dense`.
It requires the number of the neurons.
Also, we can optionally give it the activation function.
For example, if we want to have 10 neurons with the `ReLU` activation,
we can use the code below:

```python
dense_layer = keras.layers.Dense(10, activation="relu")
```

> **Source**: https://keras.io/api/layers/core_layers/dense/ 

## Output layer

`Output layer` is the layer that we use to generate our output respect to our problem.
In **classification** problems we mostly use `Dense layer` with `softmax` as its activation.
For example, if we have $4$ classes we can define an output layer like below:

```python
keras.layers.Dense(4, activation="softmax"),
```

Now, let's add it to our sequential model:

```python
model = keras.Sequential(
    [
        keras.layers.Input(shape=(3, 224, 224)),
        keras.layers.Dense(4, activation="softmax"),
    ],
)
```

## Flatten layer

`Flatten layer` is simply flatten the output of the previous layer. 
If we have `5` data that their shape is `(8, 9)`, the output
of a `flatten layer` would be `5` data with the shape of `(72,)`.
To use a flatten layer we can use the code below:

```python
flatten_layer = keras.layers.Flatten()
```

Since the output of our input layer is `(3, 224, 224)`, we should flatten
this output to give it to our `dense layer`.
So let's add our `flatten layer` to our sequential model like this.

```python
model = keras.Sequential(
    [
        keras.layers.Input(shape=(3, 224, 224)),
        keras.layers.Flatten(),
        keras.layers.Dense(4, activation="softmax"),
    ],
)
```

> **Source**: https://keras.io/api/layers/reshaping_layers/flatten/ 

## Compile

`compile` is the function that we use to determine our `loss function`, `optimizer` and `metrics`.
These functions are necessary in the training procedure, and we are going to talk about them individually
in the next tutorials.
But for now, we can use the code below to compile our model.

```python
model.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"],
)
```

## Your turn

## Conclusion
