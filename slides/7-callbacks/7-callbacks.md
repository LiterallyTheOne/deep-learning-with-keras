---
marp: true
theme: uncover
footer: By Ramin Zarebidoky (LiterallyTheOne)
header: Deep Learning with Keras Tutorial, Callbacks
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

## 7: Callbacks

![bg right:33% w:400](qr-code-1.webp)

---
<!-- paginate: true -->

## Introduction

* Previous Tutorial: Preprocessing and Data Augmentation
* This Tutorial: Callbacks

---

## Callbacks

* A function
* Pass to Fit function
* Gets called in a specific moment

---

## TensorBoard

```python
from keras.callbacks import TensorBoard

log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
tensorboard_callback = TensorBoard(log_dir=log_dir)

history = model.fit(
    ...,
    callbacks=[tensorboard_callback],
)

```

---

## EarlyStopping

* Stops the training
* No improvement

---

## EarlyStopping Example

```python
from keras.callbacks import EarlyStopping

early_stopping = EarlyStopping(
    monitor="val_loss",
    patience=5,
    restore_best_weights=True,
    verbose=1,
)

history = model.fit(
    ...,
    epochs=200,
    callbacks=[tensorboard_callback, early_stopping],
)
```

---

## ModelCheckpoint

* Saves the model
* During training
* Helps us
  * Not to loose the model
  * Have the previous weights to continue from them

---

## ModelCheckpoint Example

```python
from keras.callbacks import ModelCheckpoint

model_checkpoint = ModelCheckpoint(
    filepath="checkpoints/best_model.weights.h5",
    monitor="val_loss",
    save_best_only=True,
    save_weights_only=True,
    verbose=1,
)

history = model.fit(
    ...,
    callbacks=[tensorboard_callback, early_stopping, model_checkpoint],
)
```

---

## Save only Weights

* If `save_weights_only=True`:
  * file name should end with: `.weights.h5`
* Helps us:
  * Capacity
  * Multi-platform

---

## Load the best weights

```python
model.load_weights("checkpoints/best_model.weights.h5")
```

---

## Link to the tutorial and materials

![w:400](qr-code-1.webp)
