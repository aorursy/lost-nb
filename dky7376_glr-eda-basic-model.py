#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import math
import pandas as pd
import glob
import os
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image


# In[2]:


PATH = "../input/landmark-recognition-2020"
train = pd.read_csv(PATH + "/train.csv")
sample_submission = pd.read_csv(PATH + "/sample_submission.csv")


# In[3]:



imgs = train[train.landmark_id==123]['id'].values
print(f"landmark_id: {123}")
_, axs = plt.subplots(2, 5, figsize=(20, 8))
axs = axs.flatten()
for f_name,ax in zip(imgs[:10],axs):
    img = Image.open(f"{PATH}/train/{f_name[0]}/{f_name[1]}/{f_name[2]}/{f_name}.jpg")
    ax.imshow(img)
    ax.axis('off')
plt.show()


imgs = train[train.landmark_id==138982]['id'].values
print(f"landmark_id: {138982}")
_, axs = plt.subplots(2, 5, figsize=(20, 8))
axs = axs.flatten()
for f_name,ax in zip(imgs[:10],axs):
    img = Image.open(f"{PATH}/train/{f_name[0]}/{f_name[1]}/{f_name[2]}/{f_name}.jpg")
    ax.imshow(img)
    ax.axis('off')    
plt.show()


# In[4]:


print(f'Total numbers of images in training set is {train.shape[0]}')
print(f'Total numbers of images in test set is {sample_submission.shape[0]}')


# In[5]:


landmarks = train.groupby('landmark_id',as_index=False)['id'].count()    .sort_values('id',ascending=False).reset_index(drop=True)
landmarks.rename(columns={'id':'count'},inplace=True)


# In[6]:


def add_text(ax,fontsize=12):
    for p in ax.patches:
        x=p.get_bbox().get_points()[:,0]
        y=p.get_bbox().get_points()[1,1]
        ax.annotate('{}'.format(int(y)), (x.mean(), y), ha='center', va='bottom',size=fontsize)
fig, (ax1,ax2) = plt.subplots(2,1,figsize=(16,8))
sns.barplot(data=landmarks[:10],x='landmark_id',y='count',ax=ax1,color='#30a2da',
           order=landmarks[:10]['landmark_id'])
add_text(ax1,fontsize=8)
ax1.set_title('Top 50 Landmarks')
ax1.set_ylabel('Number of Images')
ax1.set_xticklabels(ax1.get_xticklabels(), rotation=40, ha="right",size=8)
sns.barplot(data=landmarks[-10:],x='landmark_id',y='count',ax=ax2,color='#fc4f30')
ax2.set_title('Bottom 50 Landmarks')
ax2.set_ylabel('Number of Images')
ax2.set_xticklabels(ax2.get_xticklabels(), rotation=40, ha="right",size=8)
plt.tight_layout()
print(f"Number of Landmarks with less than 10 images are {len(landmarks[landmarks['count']<10])}")
print(f"Number of Landmarks with less than 20 images are {len(landmarks[landmarks['count']<20])}")
plt.show()


# In[7]:


dataset = train.landmark_id.value_counts()
dataset = pd.DataFrame({"landmark_id": dataset.index, "num_of_images": dataset.values})
max_img = dataset.num_of_images.max()
min_img = dataset.num_of_images.min()
print(f'Total number of classes is: {len(dataset)}')
print(f'maximum image for a landmark class is:{max_img}, minimum image for landmark class is:{min_img}')


# In[8]:


# Plot
plt.scatter(dataset.index, dataset.num_of_images, alpha=0.5)
plt.title('No. of images Vs. Class')
plt.xlabel('Classes')
plt.ylabel('No. of images')
plt.show()


# In[9]:


num_img =50
per = len(dataset[dataset.num_of_images<num_img])/len(dataset)*100
print(f'There are {int(per)}% classes having less than {num_img} images')

