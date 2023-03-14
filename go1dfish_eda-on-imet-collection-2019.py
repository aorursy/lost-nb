#!/usr/bin/env python
# coding: utf-8



import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import seaborn as sns
sns.set_palette("husl")
import os
print(os.listdir("../input/"))
import warnings
warnings.filterwarnings('ignore')
import gc
from pathlib import Path
from PIL import Image
from IPython.display import clear_output
from tqdm import tqdm_notebook as tqdm




labels_df = pd.read_csv("../input/labels.csv")




labels_df.head()




labels_df.tail()




print(f"labels.csv have {labels_df.shape[0]} attributes_name.")




kind_dict = {}
for i in range(len(labels_df)):
    kind, name = labels_df.attribute_name[i].split("::")
    if(kind in kind_dict.keys()):
        kind_dict[kind] += 1
    else:
        kind_dict[kind] = 1
for key, val in kind_dict.items():
    print("The number of {} is {}({:.2%})".format(key, val, val/len(labels_df)))




label_dict = labels_df.attribute_name.to_dict()




train_df = pd.read_csv("../input/train.csv")
train_df.head()




test_path = Path("../input/test/")
test_num = len(list(test_path.glob("*.png")))
train_num = len(train_df)
fig, ax = plt.subplots()
sns.barplot(y=["train", "test"], x=[train_num, test_num])
ax.set_title("The amount of data")
clear_output()




id_len_dict = {}
id_num_dict = {}
for i in range(train_df.shape[0]):
    ids = list(map(int, train_df.attribute_ids[i].split()))
    id_len = len(ids)
    if(id_len in id_len_dict.keys()):
        id_len_dict[id_len] += 1
    else:
        id_len_dict[id_len] = 1
    for num in ids:
        if(num in id_num_dict.keys()):
            id_num_dict[num] += 1
        else:
            id_num_dict[num] = 1




fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
sns.barplot(x=list(id_len_dict.keys()), y=list(id_len_dict.values()), ax=ax1)
ax1.set_title("The number of attribute_id per image")
ax2.bar(list(id_num_dict.keys()), list(id_num_dict.values()))
ax2.set_title("Appearance frequency of attribute")
ax2.set_xticks(np.linspace(0, max(id_num_dict.keys()), 10, dtype='int'))
clear_output()




id_len_list = sorted(id_len_dict.items(), key=lambda x: -x[1])
print("The number of attribute_id per image\n")
print("{0:9s}{1:20s}".format("label num".rjust(9), "amount".rjust(20)))
for i in id_len_list:
    print("{0:9d}{1:20d}".format(i[0], i[1]))




id_num_list = sorted(id_num_dict.items(), key=lambda x: -x[1])
print("Top 10 high appearance frequency attitude\n")
print("{0:4s}{1:15s}{2:30s}".format("rank".rjust(4), "num".rjust(15), "attitude_name".rjust(30)))
for i in range(10):
    print("{0:3d}.{1:15d}{2:30s}".format(i+1, id_num_list[i][1], (label_dict[id_num_list[i][0]]).rjust(30)))




id_num_list = sorted(id_num_dict.items(), key=lambda x: x[1])
print("Top 16 low appearance frequency attitude\n")
print("{0:4s}{1:15s}{2:30s}".format("rank".rjust(4), "num".rjust(15), "attitude_name".rjust(50)))
for i in range(16):
    print("{0:3d}.{1:15d}{2:50s}".format(i+1, id_num_list[i][1], (label_dict[id_num_list[i][0]]).rjust(50)))




train_path = Path("../input/train/")
fig, ax = plt.subplots(3, figsize=(10, 20))
for i, index in enumerate(np.random.randint(0, len(train_df), 3)):
    path = (train_path / (train_df.id[index] + ".png"))
    img = np.asarray(Image.open(str(path)))
    ax[i].imshow(img)
    ids = list(map(int, train_df.attribute_ids[index].split()))
    for num, attribute_id in enumerate(ids):
        x_pos = img.shape[1] + 100
        y_pos = (img.shape[0] - 100) / len(ids) * num + 100
        ax[i].text(x_pos, y_pos, label_dict[attribute_id], fontsize=20)




test_path = Path("../input/test/")
test_img_paths = list(test_path.glob("*.png"))
fig, ax = plt.subplots(3, figsize=(10, 20))
for i, path in enumerate(np.random.choice(test_img_paths, 3)):
    img = np.asarray(Image.open(str(path)))
    ax[i].imshow(img)




def check_area_size(folder_path):
    area_list = []
    max_width = None
    min_width = None
    max_height = None
    min_height = None
    img_paths = list(folder_path.glob("*.png"))
    for path in tqdm(img_paths):
        img = np.asarray(Image.open(str(path)))
        shape = img.shape
        area_list.append(shape[0]*shape[1])
        if(max_width is None):
            max_width = (shape[1], path)
            min_width = (shape[1], path)
            max_height = (shape[0], path)
            min_height = (shape[0], path)
        else:
            if(max_width[0] < shape[1]):
                max_width = (shape[1], path)
            elif(min_width[0] > shape[1]):
                min_width = (shape[1], path)
            if(max_height[0] < shape[0]):
                max_height = (shape[0], path)
            elif(min_height[0] > shape[0]):
                min_height = (shape[0], path)
    return area_list, max_width, min_width, max_height, min_height




train_area_list, train_max_width, train_min_width, train_max_height, train_min_height    = check_area_size(train_path)
clear_output()




print("test max area size is {}".format(max(train_area_list)))
print("test min area size is {}".format(min(train_area_list)))
print("Max area is {:.2f} times min area".format(max(train_area_list)/ min(train_area_list)))




print(f"max train image width is {train_max_width[0]}")
img = np.asarray(Image.open(str(train_max_width[1])))
plt.imshow(img)
plt.show()




print(f"min train image width is {train_min_width[0]}")
img = np.asarray(Image.open(str(train_min_width[1])))
plt.imshow(img)
plt.show()




print(f"max train image height is {train_max_height[0]}")
img = np.asarray(Image.open(str(train_max_height[1])))
plt.imshow(img)
plt.show()




print(f"min train image height is {train_min_height[0]}")
img = np.asarray(Image.open(str(train_min_height[1])))
plt.imshow(img)
plt.show()




test_area_list, test_max_width, test_min_width, test_max_height, test_min_height    = check_area_size(test_path)
clear_output()




print("test max area size is {}".format(max(test_area_list)))
print("test min area size is {}".format(min(test_area_list)))
print("Max area is {:.2f} times min area".format(max(test_area_list)/ min(test_area_list)))




fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6), sharex=True)
sns.distplot(train_area_list, kde=False, ax=ax1)
ax1.set_title("Distribution of the train image area")
sns.distplot(test_area_list, kde=False, ax=ax2)
ax2.set_title("Distribution of the test image area")
plt.show()




print(f"max test image width is {test_max_width[0]}")
img = np.asarray(Image.open(str(test_max_width[1])))
plt.imshow(img)
plt.show()




print(f"min train image width is {test_min_width[0]}")
img = np.asarray(Image.open(str(test_min_width[1])))
plt.imshow(img)
plt.show()




print(f"max train image height is {test_max_height[0]}")
img = np.asarray(Image.open(str(test_max_height[1])))
plt.imshow(img)
plt.show()




print(f"min train image height is {test_min_height[0]}")
img = np.asarray(Image.open(str(test_min_height[1])))
plt.imshow(img)
plt.show()






