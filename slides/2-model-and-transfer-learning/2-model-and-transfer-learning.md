---
marp: true
theme: uncover
class: invert
footer: By Ramin Zarebidoky (LiterallyTheOne)
header: Deep Learning with Keras Tutorial, Model and Transfer Learning
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

## 2: Model and Transfer Learning

![bg right:33% w:400](qr-code-1.webp)

---
<!-- paginate: true -->

## Introduction

* Previous Tutorial: Load a dataset Correctly
* This Tutorial: Model and Transfer Learning

---

## Your Turn

<style scoped>
  {
    font-size: 30px; /* Adjust this value to your desired size */
  }
</style>

* Load your dataset in 3 subsets: **train**, **validation**, and **test**.
* Choose another model other than `MobileNetV2` as your base model.
  * You can use this link to see the other models
  * <https://keras.io/api/applications/>
* Set the input layer according to your data
* Set the output layer according to the number of the classes
* Use the transfer learning technique correctly
* Train your model on your train subset
  * You should fill `validation_data` argument
  * 5 epochs is enough
* Report your result on your test subset

---

## Link to the tutorial and materials

![w:400](qr-code-1.webp)
