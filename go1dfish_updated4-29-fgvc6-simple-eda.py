#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np # linear algebra
import pandas as pd
pd.set_option("display.max_rows", 101)
import os
print(os.listdir("../input"))
import cv2
import json
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
plt.rcParams["font.size"] = 15
import seaborn as sns
from collections import Counter
from PIL import Image
import math
import seaborn as sns


# In[2]:


input_dir = "../input/"


# In[3]:


def classid2label(class_id):
    category, *attribute = class_id.split("_")
    return category, attribute


# In[4]:


def print_dict(dictionary, name_dict):
    print("{}{}{}{}{}".format("rank".ljust(5), "id".center(8), "name".center(40), "amount".rjust(10), "ratio(%)".rjust(10)))
    all_num = sum(dictionary.values())
    for i, (key, val) in enumerate(sorted(dictionary.items(), key=lambda x: -x[1])):
        print("{:<5}{:^8}{:^40}{:>10}{:>10.3%}".format(i+1, key, name_dict[key], val, val/all_num))


# In[5]:


def print_img_with_labels(img_name, labels, category_name_dict, attribute_name_dict, ax):
    img = np.asarray(Image.open(input_dir + "train/" + img_name))
    label_interval = (img.shape[0] * 0.9) / len(labels)
    ax.imshow(img)
    for num, attribute_id in enumerate(labels):
        x_pos = img.shape[1] * 1.1
        y_pos = (img.shape[0] * 0.9) / len(labels) * (num + 2) + (img.shape[0] * 0.1)
        if(num == 0):
            ax.text(x_pos, y_pos-label_interval*2, "category", fontsize=12)
            ax.text(x_pos, y_pos-label_interval, category_name_dict[attribute_id], fontsize=12)
            if(len(labels) > 1):
                ax.text(x_pos, y_pos, "attribute", fontsize=12)
        else:
            ax.text(x_pos, y_pos, attribute_name_dict[attribute_id], fontsize=12)


# In[6]:


def print_img(img_name, ax):
    img_df = train_df[train_df.ImageId == img_name]
    labels = list(set(img_df["ClassId"].values))
    print_img_with_labels(img_name, labels, category_name_dict, attribute_name_dict, ax)


# In[7]:


def json2df(data):
    df = pd.DataFrame()
    for index, el in enumerate(data):
        for key, val in el.items():
            df.loc[index, key] = val
    return df


# In[8]:


train_df = pd.read_csv(input_dir + "train.csv")


# In[9]:


train_df.head()


# In[10]:


with open(input_dir + "label_descriptions.json") as f:
    label_description = json.load(f)


# In[11]:


print("this dataset info")
print(json.dumps(label_description["info"], indent=2))


# In[12]:


category_df = json2df(label_description["categories"])
category_df["id"] = category_df["id"].astype(int)
category_df["level"] = category_df["level"].astype(int)
attribute_df = json2df(label_description["attributes"])
attribute_df["id"] = attribute_df["id"].astype(int)
attribute_df["level"] = attribute_df["level"].astype(int)


# In[13]:


print("Category Labels")
category_df


# In[14]:


print("Attribute Labels")
attribute_df


# In[15]:


print("We have {} categories, and {} attributes.".format(len(label_description['categories']), len(label_description['attributes'])))
print("Each label　have ID, name, supercategory, and level.")


# In[16]:


train_df.head(10)


# In[17]:


image_label_num_df = train_df.groupby("ImageId")["ClassId"].count()


# In[18]:


fig, ax = plt.subplots(figsize=(25, 7))
x = image_label_num_df.value_counts().index.values
y = image_label_num_df.value_counts().values
z = zip(x, y)
z = sorted(z)
x, y = zip(*z)
index = 0
x_list = []
y_list = []
for i in range(1, max(x)+1):
    if(i not in x):
        x_list.append(i)
        y_list.append(0)
    else:
        x_list.append(i)
        y_list.append(y[index])
        index += 1
for i, j in zip(x_list, y_list):
    ax.text(i-1, j, j, ha="center", va="bottom", fontsize=13)
sns.barplot(x=x_list, y=y_list, ax=ax)
ax.set_xticks(list(range(0, len(x_list), 5)))
ax.set_xticklabels(list(range(1, len(x_list), 5)))
ax.set_title("the number of labels per image")
ax.set_xlabel("the number of labels")
ax.set_ylabel("amout");


# In[19]:


counter_category = Counter()
counter_attribute = Counter()
for class_id in train_df["ClassId"]:
    category, attribute = classid2label(class_id)
    counter_category.update([category])
    counter_attribute.update(attribute)


# In[20]:


len(counter_category)


# In[21]:


len(counter_attribute)


# In[22]:


category_name_dict = {}
for i in label_description["categories"]:
    category_name_dict[str(i["id"])] = i["name"]
attribute_name_dict = {}
for i in label_description["attributes"]:
    attribute_name_dict[str(i["id"])] = i["name"]


# In[23]:


print("Category label frequency")
print_dict(counter_category, category_name_dict)


# In[24]:


print("Attribute label frequency")
print_dict(counter_attribute, attribute_name_dict)


# In[25]:


train_df.ClassId.max()


# In[26]:


attribute_num_dict = {}
none_key = str(len(counter_attribute))
k = list(map(str, range(len(counter_attribute) + 1)))
v = [0] * (len(counter_attribute) + 1)
zipped = zip(k, v)
init_dict = dict(zipped)
for class_id in train_df["ClassId"].values:
    category, attributes = classid2label(class_id)
    if category not in attribute_num_dict.keys():
        attribute_num_dict[category] = init_dict.copy()
    if attributes == []:
        attribute_num_dict[category][none_key] += 1
        continue
    for attribute in attributes:
        attribute_num_dict[category][attribute] += 1


# In[27]:


fig, ax = plt.subplots(math.ceil(len(counter_category)/2), 2,                       figsize=(8*2, 6*math.ceil(len(counter_category)/2)), sharey=True)
for index, key in enumerate(sorted(map(int, attribute_num_dict.keys()))):
    x = list(map(int, attribute_num_dict[str(key)].keys()))
    total = sum(attribute_num_dict[str(key)].values())
    y = list(map(lambda x: x / total, attribute_num_dict[str(key)].values()))
    sns.barplot(x, y, ax=ax[index//2, index%2])
    ax[index//2, index%2].set_title("category:{}({})".format(key, category_name_dict[str(key)]))
    ax[index//2, index%2].set_xticks(list(range(0, int(none_key), 5)))
    ax[index//2, index%2].set_xticklabels(list(range(0, int(none_key), 5)))
print("the ratio of attribute per category(x=92 means no attribute)")


# In[28]:


print("The number of training image is {}.".format(len(os.listdir("../input/train/"))))
print("The number of test image is {}.".format(len(os.listdir("../input/test/"))))


# In[29]:


image_shape_df = train_df.groupby("ImageId")["Height", "Width"].first()


# In[30]:


fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 5))
ax1.hist(image_shape_df.Height, bins=100)
ax1.set_title("Height distribution")
ax2.hist(image_shape_df.Width, bins=100)
ax2.set_title("Width distribution");


# In[31]:


img_name = image_shape_df.Height.idxmin()
height, width = image_shape_df.loc[img_name, :]
print("Minimam height image is {},\n(H, W) = ({}, {})".format(img_name, height, width))
fig, ax = plt.subplots()
print_img(img_name, ax)


# In[32]:


img_name = image_shape_df.Height.idxmax()
height, width = image_shape_df.loc[img_name, :]
print("Maximum height image is {},\n(H, W) = ({}, {})".format(img_name, height, width))
fig, ax = plt.subplots()
print_img(img_name, ax)


# In[33]:


img_name = image_shape_df.Width.idxmin()
height, width = image_shape_df.loc[img_name, :]
print("Minimam width image is {},\n(H, W) = ({}, {})".format(img_name, height, width))
fig, ax = plt.subplots()
print_img(img_name, ax)


# In[34]:


img_name = image_shape_df.Width.idxmax()
height, width = image_shape_df.loc[img_name, :]
print("Maximum width image is {},\n(H, W) = ({}, {})".format(img_name, height, width))
fig, ax = plt.subplots()
print_img(img_name, ax)


# In[35]:


pallete =  [
    'Pastel1', 'Pastel2', 'Paired', 'Accent', 'Dark2',
    'Set1', 'Set2', 'Set3', 'tab10', 'tab20', 'tab20b', 'tab20c']


def make_mask_img(segment_df):
    category_num = len(counter_category)
    seg_width = segment_df.at[0, "Width"]
    seg_height = segment_df.at[0, "Height"]
    seg_img = np.full(seg_width*seg_height, category_num-1, dtype=np.uint8)
    for encoded_pixels, class_id in zip(segment_df["EncodedPixels"].values, segment_df["ClassId"].values):
        pixel_list = list(map(int, encoded_pixels.split(" ")))
        for i in range(0, len(pixel_list), 2):
            start_index = pixel_list[i] - 1
            index_len = pixel_list[i+1] - 1
            seg_img[start_index:start_index+index_len] =                int(int(class_id.split("_")[0]) / (category_num-1) * 255)
    seg_img = seg_img.reshape((seg_height, seg_width), order='F')
    return seg_img


def train_generator(df, batch_size):
    img_ind_num = df.groupby("ImageId")["ClassId"].count()
    index = df.index.values[0]
    trn_images = []
    seg_images = []
    for i, (img_name, ind_num) in enumerate(img_ind_num.items()):
        img = cv2.imread("../input/train/" + img_name)
        segment_df = (df.loc[index:index+ind_num-1, :]).reset_index(drop=True)
        index += ind_num
        if segment_df["ImageId"].nunique() != 1:
            raise Exception("Index Range Error")
        seg_img = make_mask_img(segment_df)
        
        # HWC -> CHW
        img = img.transpose((2, 0, 1))
        
        trn_images.append(img)
        seg_images.append(seg_img)
        if((i+1) % batch_size == 0):
            return trn_images, seg_images


# In[36]:


def cv2plt(img, isColor=True):
    original_img = img
    original_img = original_img.transpose(1, 2, 0)
    original_img = cv2.cvtColor(original_img, cv2.COLOR_BGR2RGB)
    return original_img


# In[37]:


original, segmented = train_generator(train_df, 6)
fig, ax = plt.subplots(3, 2, figsize=(16, 18))
for i, (img, seg) in enumerate(zip(original, segmented)):
    ax[i//2, i%2].imshow(cv2plt(img))
    seg[seg == 45] = 255
    ax[i//2, i%2].imshow(seg, cmap='tab20_r', alpha=0.6)
    ax[i//2, i%2].set_title("Sample {}".format(i))


# In[38]:


sample_df = pd.read_csv(input_dir + "sample_submission.csv")


# In[39]:


sample_df.head(20)


# In[40]:





# In[40]:





# In[40]:




