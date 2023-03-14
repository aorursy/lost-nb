#!/usr/bin/env python
# coding: utf-8



# basic
import numpy as np
import pandas as pd

# visualize
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image

# algorithms
from collections import Counter
from collections import defaultdict
import cv2
import math

## tools
import os
from tqdm import tqdm
from pathlib import Path
import json

## config ##
get_ipython().run_line_magic('matplotlib', 'inline')
plt.rcParams["font.size"] = 15
pd.set_option("display.max_rows", 101)
print(os.listdir("/kaggle/input/severstal-steel-defect-detection/"))




input_base_path = "/kaggle/input/severstal-steel-defect-detection/"
train = pd.read_csv(input_base_path + "train.csv")
sample = pd.read_csv(input_base_path + "sample_submission.csv")




dfs = pd.concat([sample.ImageId_ClassId.str.split('_', expand=True), sample], axis=1)     .drop('ImageId_ClassId', axis=1)
dfs.rename(columns={0: 'file_name', 1: 'defect_class'}, inplace=True)




dft = pd.concat([train.ImageId_ClassId.str.split('_', expand=True), train], axis=1)     .drop('ImageId_ClassId', axis=1)
dft.rename(columns={0: 'file_name', 1: 'defect_class'}, inplace=True)




dft




## image counts of each category
print('all:', len(dft.file_name.unique()))
print('defects:', len(dft.dropna().file_name.unique()))
print('non defects:', len(dft.file_name.unique()) - len(dft.dropna().file_name.unique()))




## how many defects each class exists
Counter(dft.dropna().defect_class)




## how many defects each image has (0: 5902)
Counter(Counter(dft.dropna().file_name).values())




## test images and its size

test_sizes = defaultdict(int)
TEST_PATH = Path(input_base_path) / 'test_images'

for path in TEST_PATH.iterdir():
    img = Image.open(path)
    test_sizes[img.size] += 1
test_sizes




## train_images and its size

train_sizes = defaultdict(int)
TRAIN_PATH = Path(input_base_path) / 'train_images'

for path in TRAIN_PATH.iterdir():
    img = Image.open(path)
    train_sizes[img.size] += 1
train_sizes




# IMG_SIZE_ = (1600, 256)
IMG_SIZE = (256, 1600) # for np.array
IMG_SIZE3 = *IMG_SIZE, 3
TRAIN_SIZE = 12568




## defect_class with colors
PALLET = [(249, 192, 12), (0, 185, 241), (114, 0, 218), (249,50,12)] # class = 1, 2, 3, 4

fig, ax = plt.subplots(1, 4, figsize=(10, 3))
for i, color in enumerate(PALLET):
    ax[i].axis('off')
    ax[i].imshow(np.ones((20, 20, 3), dtype=np.uint8) * PALLET[i])
    ax[i].set_title("class color: {}".format(i+1))
fig.suptitle("each class colors")

plt.show()




pixels = dft.EncodedPixels
pixels.loc[0]




# 1 channel mask image-like np.array
def gen_mask(pxls):
    mask = np.zeros(IMG_SIZE[0] * IMG_SIZE[1], dtype=np.uint8)

    ary = np.array(pxls.split(), dtype = int)
    ary.resize((len(ary) // 2, 2))

    for s, e in ary:
        mask[s-1:s+e] += 1

    return mask.reshape(IMG_SIZE, order='F')




def reflect_defects_contours(img, encoded_pixels, color):
    mask = gen_mask(encoded_pixels)
    contours, _ = cv2.findContours(mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
    for i in range(0, len(contours)):
        cv2.polylines(img, contours[i], True, color, 2)

    return img




pixels = dft.EncodedPixels

def show_defects(idx):
    idx = idx // 4 * 4 # adjust index to the class 1 defect
    fig, ax = plt.subplots(figsize=(15, 15))
    file_name = dft.loc[idx].file_name
    img = cv2.imread(str(TRAIN_PATH / file_name))
    for j, color in enumerate(PALLET):
        encoded_pixels = pixels.loc[idx + j]
        if encoded_pixels is np.nan:
            continue
        reflect_defects_contours(img, encoded_pixels, color)
    ax.imshow(img)




# example for show_defects function
show_defects(0)




# ------------------------------------------------------------------------------- #




dft




defected_idxs = defaultdict(list)
for idx, cls in dft.dropna().defect_class.items():
    defected_idxs[cls].append(idx)




for k, v in defected_idxs.items():
    print(k, len(v))




# class-1

for i in defected_idxs['1'][:5]:
    show_defects(i)




# class-2

for i in defected_idxs['2'][15:20]:
    show_defects(i)




# class-3

for i in defected_idxs['3'][15:20]:
    show_defects(i)




# class-4

for i in defected_idxs['4'][15:20]:
    show_defects(i)






