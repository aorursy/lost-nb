#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from matplotlib.patches import Rectangle
import seaborn as sns
from os import listdir
from os.path import isfile, join


# In[3]:


df = pd.read_csv('../input/global-wheat-detection/train.csv')


# In[4]:


df.head()


# In[5]:


df.info()


# In[6]:


df['source'].unique()


# In[7]:


df['bbox'] = df['bbox'].map(lambda x: x.lstrip('[').rstrip(']'))
df['bbox'] = df['bbox'].str.split(',')


# In[8]:


df['source'].hist()


# In[9]:


plt.hist(df['image_id'].value_counts(), bins=30)
plt.show()


# In[11]:


plt.figure(figsize=(10,5))
img = mpimg.imread('../input/global-wheat-detection/train/0d2948e6d.jpg')
imgplot = plt.imshow(img)

for rec_patch in df[df['image_id'] == '0d2948e6d']['bbox']:
    plt.gca().add_patch(
        Rectangle(
            (float(rec_patch[0]), float(rec_patch[1])),
            float(rec_patch[2]),
            float(rec_patch[3]),
            linewidth=2,
            edgecolor='r',
            facecolor='none'
        ))


# In[12]:


fig, axes = plt.subplots(4, 4, figsize=(15,15))

for row in range(0, 4):
    for column in range(0, 4):
        image_id = df['image_id'].sample().values[0]        
        img = mpimg.imread('../input/global-wheat-detection/train/%s.jpg' % image_id)
        imgplot = axes[row, column].imshow(img)
          
        for rec_patch in df[df['image_id'] == image_id]['bbox']:
            axes[row, column].add_patch(
                Rectangle(
                    (float(rec_patch[0]), float(rec_patch[1])),
                    float(rec_patch[2]),
                    float(rec_patch[3]),
                    linewidth=2,
                    edgecolor='r',
                    facecolor='none'
                ))


# In[ ]:





# In[13]:


df_head_count = df['image_id'].value_counts().to_frame('head_count')
df_head_count = df_head_count.rename_axis('image_id').reset_index()

def get_image_source(image_id):
    return df[df['image_id'] == image_id]['source'].unique()[0]

# df_head_count['image_id'].apply(lambda image_id: df[df['image_id'] == image_id]['source'].unique()[0])

df_head_count['source'] = df_head_count['image_id'].map(get_image_source)


# In[14]:


df_head_count.shape


# In[16]:


fig, axes = plt.subplots(3, 3, figsize=(15,15))

for row in range(0, 3):
    for column in range(0, 3):
        image_id = df_head_count[df_head_count['head_count'] == 1].sample()['image_id'].values[0]     
        img = mpimg.imread('../input/global-wheat-detection/train/%s.jpg' % image_id)
        imgplot = axes[row, column].imshow(img)
          
        for rec_patch in df[df['image_id'] == image_id]['bbox']:
            axes[row, column].add_patch(
                Rectangle(
                    (float(rec_patch[0]), float(rec_patch[1])),
                    float(rec_patch[2]),
                    float(rec_patch[3]),
                    linewidth=2,
                    edgecolor='r',
                    facecolor='none'
                ))


# In[ ]:


**Images with less than 10 wheat heads**


# In[17]:


fig, axes = plt.subplots(3, 3, figsize=(15,15))

for row in range(0, 3):
    for column in range(0, 3):
        image_id = df_head_count[df_head_count['head_count'] <= 10].sample()['image_id'].values[0]     
        img = mpimg.imread('../input/global-wheat-detection/train/%s.jpg' % image_id)
        imgplot = axes[row, column].imshow(img)
          
        for rec_patch in df[df['image_id'] == image_id]['bbox']:
            axes[row, column].add_patch(
                Rectangle(
                    (float(rec_patch[0]), float(rec_patch[1])),
                    float(rec_patch[2]),
                    float(rec_patch[3]),
                    linewidth=2,
                    edgecolor='r',
                    facecolor='none'
                ))


# In[ ]:





# In[18]:


sns.countplot(x='source', data=df_head_count)


# In[19]:


df_head_count.groupby('source')['head_count'].mean().plot.bar()


# In[ ]:




