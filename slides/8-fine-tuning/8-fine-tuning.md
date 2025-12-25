---
marp: true
theme: uncover
footer: By Ramin Zarebidoky (LiterallyTheOne)
header: Deep Learning with Keras Tutorial, Fine-tuning
size: 16:9
---


<style scoped>
p {
  color: cyan;
}
</style>

<!-- _header: "" -->
<!-- _footer: "" -->

# Deep Learning with Keras

By LiterallyTheOne

## 8: Fine-tuning

![bg right:33% w:400](qr-code-1.webp)

---
<!-- paginate: true -->

## Introduction

* Previous Tutorial: Callbacks
* This Tutorial: Fine-tuning

---

## Fine-tuning

* Deep learning technique
* Adapt to the new dataset
* Similar to Transfer learning
* Unfreeze some of the last layers

---

## Fine-tuning example

```python
base_model = MobileNetV2(include_top=False, input_shape=(224, 224, 3))

for layer in base_model.layers[:-4]:
    layer.trainable = False

```

---

## Fine-tuning example result

<style scoped>
  pre {
    font-size: 12px; /* Adjust this value to your desired size */
  }
</style>

```python
print(base_model.summary(show_trainable=True))

"""
--------
output: 

  ...
├───────────────────┼─────────────────┼───────────┼────────────────┼───────┤
| block_16_project  │ (None, 7, 7,    │   307,200 │ block_16_dept… │   N   │
│ (Conv2D)          │ 320)            │           │                │       │
├───────────────────┼─────────────────┼───────────┼────────────────┼───────┤
| block_16_project… │ (None, 7, 7,    │     1,280 │ block_16_proj… │   Y   │
│ (BatchNormalizat… │ 320)            │           │                │       │
├───────────────────┼─────────────────┼───────────┼────────────────┼───────┤
│ Conv_1 (Conv2D)   │ (None, 7, 7,    │   409,600 │ block_16_proj… │   Y   │
│                   │ 1280)           │           │                │       │
├───────────────────┼─────────────────┼───────────┼────────────────┼───────┤
│ Conv_1_bn         │ (None, 7, 7,    │     5,120 │ Conv_1[0][0]   │   Y   │
│ (BatchNormalizat… │ 1280)           │           │                │       │
├───────────────────┼─────────────────┼───────────┼────────────────┼───────┤
│ out_relu (ReLU)   │ (None, 7, 7,    │         0 │ Conv_1_bn[0][… │   -   │
│                   │ 1280)           │           │                │       │
└───────────────────┴─────────────────┴───────────┴────────────────┴───────┘

Total params: 2,257,984 (8.61 MB)
Trainable params: 412,800 (1.57 MB)
Non-trainable params: 1,845,184 (7.04 MB)

"""

```

---

## Underfitting

* Not training well on training data
* Model
  * Not learning the pattern

---

## Underfitting: simple model

* Reason:
  * Model is too simple
* Solution:
  * Choose a more complex model
  * More trainable layers

---

## Underfitting: Regularization

* Reason:
  * To much regularization
* Solution:
  * Choose correct regularization techniques

---

## Underfitting: input features

* Reason:
  * Incorrect input features
  * Example: House pice -> not having the size
* Solution:
  * Gather the correct features

---

## Overfitting

* Training: good
* Validation and Test: bad
* Model
  * Understood the pattern
  * Including the noise

---

## Overfitting: Complex model

* Reason:
  * Model is too complex
* Solution:
  * Simpler model
  * lower trainable layers

---

## Overfitting: Over training

* Reason:
  * Model is trained for so long
* Solution:
  * EarlyStopping

---

## Overfitting: Regularization

* Reason
  * Not enough Regularization
* Solution:
  * Add the correct Regularization techniques

---

## Generalize our classifier

```python
model = keras.Sequential(
    [
        ...,
        layers.GlobalAveragePooling2D(),
        layers.Dropout(0.5),
        layers.Dense(128, activation="relu"),
        layers.Dropout(0.5),
        layers.Dense(4, activation="softmax"),
    ]
)
```

---

## Link to the tutorial and materials

![w:400](qr-code-1.webp)
