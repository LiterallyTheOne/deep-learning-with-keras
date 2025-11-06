---
marp: true
theme: uncover
class: invert
footer: By Ramin Zarebidoky (LiterallyTheOne)
header: Deep Learning with Keras Tutorial, Load a dataset
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

## 1: Load an Image Classification Dataset

![bg right:33% w:400](qr-code-1.webp)

---
<!-- paginate: true -->

## Introduction

* Previous Tutorial: Select a Kaggle dataset
* This Tutorial: Load it correctly

---

## Get a Dataset from Kaggle

* Kagglehub
* Download: top right of the page of the dataset

---

## Kagglehub example

```python
import kagglehub

# Download latest version
path = kagglehub.dataset_download("balabaskar/tom-and-jerry-image-classification")

print("Path to dataset files:", path)
```

---

## Directory structure

```text
class_a/
...a_image_1.jpg
...a_image_2.jpg
class_b/
...b_image_1.jpg
...b_image_2.jpg
```

---

## Find the Image structure using ls

```shell
!ls {path}


"""
--------
output: 

challenges.csv   ground_truth.csv tom_and_jerry
"""
```

---

## Moving forward

```shell
!ls {path}/tom_and_jerry


"""
--------
output: 

tom_and_jerry
"""
```

---

## Find the directory that we want

```shell
!ls {path}/tom_and_jerry/tom_and_jerry


"""
--------
output: 

jerry       tom         tom_jerry_0 tom_jerry_1
"""
```

---

## Store it in a variable

```python
from pathlib import Path

data_path = Path(path) / "tom_and_jerry/tom_and_jerry"
```

---

## ImageFolder

* Load an Image Classification Dataset
* Needs the structure below

```text
main_directory/
...class_a/
......a_image_1.jpg
......a_image_2.jpg
...class_b/
......b_image_1.jpg
......b_image_2.jpg
```

---

## ImageFolder example

```python
from torchvision.datasets import ImageFolder

all_data = ImageFolder(data_path)
```

---

## Show one sample

```python
from matplotlib import pyplot as plt

for image, label in all_data:
    plt.figure()
    plt.imshow(image)
    print(label)
    break

"""
--------
output: 

0
"""
```

![bg right:40% w:500](../../docs/1-load-a-dataset/tom-and-jerry-example.webp)

---

## Transforms

* Transform images to our standard
* Torchvision

---

## Resize Transform

<style scoped>
  pre {
    font-size: 15px; /* Adjust this value to your desired size */
  }
</style>

```python
from torchvision import transforms

# Without resize transform

all_data = ImageFolder(data_path)

for image, label in all_data:
    print(f"image size without resize transform: {image.size}")
    break

# With resize transform

transform = transforms.Resize((90, 160))

all_data = ImageFolder(data_path, transform=transform)

for image, label in all_data:
    print(f"image size with resize transform: {image.size}")
    break

"""
--------
output: 

image size without resize transform: (1280, 720)
image size with resize transform: (160, 90)
"""
```

---

## ToTensor

* Changes PIL images to `Tensor`
* The way that model accepts our data

---

## Combine two Transforms

<style scoped>
  pre {
    font-size: 19px; /* Adjust this value to your desired size */
  }
</style>

```python
trs = transforms.Compose(
    [
        transforms.Resize((90, 160)),
        transforms.ToTensor(),
    ]
)

all_data = ImageFolder(data_path, transform=trs)

for image, label in all_data:
    print(type(image))
    print(image.shape)
    break

"""
--------
output: 

<class 'torch.Tensor'>
torch.Size([3, 90, 160])
"""
```

---

## Split into Train, Validation, and Test

* Some Datasets don't have these subsets
* We should make them manually
* `random_split`

---

<style scoped>
  pre {
    font-size: 19px; /* Adjust this value to your desired size */
  }
</style>

## Example of splitting

```python
import torch
from torch.utils.data import random_split

g1 = torch.Generator().manual_seed(20)
train_data, val_data, test_data = random_split(all_data, [0.7, 0.2, 0.1], g1)

print(f"all_data's size: {len(all_data)}")
print(f"train_data's size: {len(train_data)}")
print(f"val_data's size: {len(val_data)}")
print(f"test_data's size: {len(test_data)}")

"""
--------
output: 

all_data's size: 5478
train_data's size: 3835
val_data's size: 1096
test_data's size: 547
"""

```

---

## DataLoader

* Takes the loaded dataset
* Helps us to apply Deep Learning techniques
  * `batch_size`
  * `shuffle`

---

## DataLoader for each subset

```python
train_loader = DataLoader(train_data, batch_size=12, shuffle=True)
val_loader = DataLoader(val_data, batch_size=12, shuffle=False)
test_loader = DataLoader(test_data, batch_size=12, shuffle=False)
```

---

## Show a batch of Data

```python
fig, axes = plt.subplots(3, 4)

axes_ravel = axes.ravel()

for images, labels in train_loader:
    for i, (image, label) in enumerate(zip(images, labels)):
        axes_ravel[i].imshow(transforms.ToPILImage()(image))
        axes_ravel[i].set_axis_off()
        axes_ravel[i].set_title(f"{label}")
    break
```

![bg right:40% w:500](../../docs/1-load-a-dataset/batch-tom-and-jerry.webp)

---

## Your Turn

* get your Kaggle dataset.
* use the ImageFolder to load that dataset and show one of its images.
* if you don’t have any of the train, validation, and test subsets, make them using random_split.
* load those three subsets using DataLoader and set a batch_size for them.
* show a batch of your data.

---

## Link to the tutorial and materials

![w:400](qr-code-1.webp)
