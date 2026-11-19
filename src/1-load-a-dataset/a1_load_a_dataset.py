# %%
# -------------------[ Setup ]-------------------

import os

os.environ["KERAS_BACKEND"] = "torch"

# %%
# -------------------[ Imports ]-------------------
from pathlib import Path

import kagglehub
import torch
from matplotlib import pyplot as plt
from torch.utils.data import DataLoader, random_split
from torchvision.datasets import ImageFolder
from torchvision.transforms import Compose, Resize, ToPILImage, ToTensor

# %%
# -------------------[ Load our dataset ]-------------------
# Download latest version
path = kagglehub.dataset_download("balabaskar/tom-and-jerry-image-classification")

print("Path to dataset files:", path)

# %%
# -------------------[ Find src directory ]-------------------
# !ls {path}/tom_and_jerry/tom_and_jerry

data_dir = Path(path) / "tom_and_jerry/tom_and_jerry"

# %%
# -------------------[ Image folder ]-------------------
all_data = ImageFolder(data_dir)

image, label = next(iter(all_data))

# %%
# -------------------[ Show the sample ]-------------------
plt.figure()
plt.imshow(image)
plt.title(f"label: {label}")
plt.show()

# %%
# -------------------[ Show size ]-------------------
print(image.size)

# %%
# -------------------[ Resize transform ]-------------------

# height, width
transform = Resize((90, 160))

all_data = ImageFolder(data_dir, transform=transform)

image, label = next(iter(all_data))

# %%
print(image.size)

# %%
# -------------------[ Transform compose ]-------------------
compose = Compose(
    [
        Resize((90, 160)),
        ToTensor(),
    ]
)

all_data = ImageFolder(data_dir, transform=compose)

image, label = next(iter(all_data))


# %%
print(type(image))
print(image.shape)

# %%
# -------------------[ Split to train, validation, test ]-------------------

g1 = torch.Generator().manual_seed(20)
train_data, val_data, test_data = random_split(all_data, [0.7, 0.2, 0.1], g1)

# %%

image, label = next(iter(train_data))  # type: ignore

# %%
plt.figure()
plt.title(f"label: {label}")
plt.imshow(ToPILImage()(image))


# %%
# -------------------[ DataLoader ]-------------------
train_loader = DataLoader(train_data, 12, shuffle=True)
val_loader = DataLoader(val_data, 12, shuffle=False)
test_loader = DataLoader(test_data, 1, shuffle=False)

# %%
images, labels = next(iter(train_loader))

# %%
# -------------------[ Show a batch of data ]-------------------
fig, axes = plt.subplots(3, 4)
axes_ravel = axes.ravel()

for image, label, ax in zip(images, labels, axes_ravel):
    ax.imshow(ToPILImage()(image))
    ax.set_title(f"label: {label}")
    ax.set_axis_off()

# %%
# -------------------[ Ramin Zarebidoky ]-------------------
# -------------------[ LiterallyTheOne.github.io ]-------------------
