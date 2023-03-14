#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
import glob

import numpy as np 
import pandas as pd 

import matplotlib.pyplot as plt
import seaborn as sns

import matplotlib.image as mpimg

from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report

from tensorflow.keras.utils import to_categorical
from tensorflow.keras.preprocessing.image import ImageDataGenerator, load_img
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dense, Flatten, Dropout


# In[2]:


get_ipython().system(' ls ../input/siim-isic-melanoma-classification')


# In[3]:


train = pd.read_csv('../input/siim-isic-melanoma-classification/train.csv')
train.head()


# In[4]:


test = pd.read_csv('../input/siim-isic-melanoma-classification/test.csv')
test.head()


# In[5]:


sample_submission = pd.read_csv('../input/siim-isic-melanoma-classification/sample_submission.csv')
sample_submission.head()


# In[6]:


print('No. of images in the train dataset :', train.shape[0])
print('No. of unique patients in train dataset :', train['patient_id'].nunique())


# In[7]:


print('No. of images in the test dataset :', test.shape[0])
print('No. of unique patients in test dataset :', test['patient_id'].nunique())


# In[8]:


def count_hbar(col, title, pal='Dark2'):
    df = pd.DataFrame(train[col].value_counts()).reset_index()
    
    sns.set_style('whitegrid')

    sns.barplot(data=df, x=col, y='index', palette=pal)
    
    for ind, row in df.iterrows():
        plt.text(row[col]+500, ind, row[col])
        
    
    sns.despine(bottom=True)
    plt.title(title)
    plt.xlabel('')
    plt.ylabel('')
    plt.show()


# In[9]:


count_hbar('sex', 'Gender', ['royalblue', 'deeppink'])


# In[10]:


count_hbar('anatom_site_general_challenge', 'Anatomy site of the mole')


# In[11]:


count_hbar('diagnosis', 'Diagnosis  of the mole')


# In[12]:


count_hbar('benign_malignant', 'Benign or Malignant', ['dimgray', 'orangered'])


# In[13]:


temp = train.groupby('patient_id').agg({'sex':max, 'age_approx':np.mean}).reset_index()

plt.figure(figsize=(12, 5))
sns.kdeplot(temp[temp['sex']=='male']['age_approx'], label='Male', shade=True, color='royalblue')
sns.kdeplot(temp[temp['sex']=='female']['age_approx'], label='Female', shade=True, color='deeppink')
plt.title('Age distribution Male and Female patients', 
          loc='left', fontsize=16)
plt.show()


# In[14]:


df = pd.DataFrame(train.groupby(['anatom_site_general_challenge'])['target'].mean())         .sort_values('target', ascending=False)         .reset_index() 
df['target'] = round(df['target'], 4)

plt.figure(figsize=(12, 5))
sns.set_style('darkgrid')

sns.barplot(data=df, x='target', y='anatom_site_general_challenge', palette='Set2')

for ind, row in df.iterrows():
    plt.text(row['target']+0.0001, ind+0.1, row['target'])

sns.despine(bottom=True)
plt.title('Probability of mole being a Malignant one wrt to it\'s possition on the human body', 
          loc='left', fontsize=16)
plt.xlabel('')
plt.ylabel('')
plt.show()


# In[15]:


def plot_images(diagnosis, title, n):
    temp = train[train['diagnosis']==diagnosis]
    img_ids = ['../input/siim-isic-melanoma-classification/jpeg/train/'+i+'.jpg' for i in temp['image_name'].sample(n)]

    fig, ax = plt.subplots(figsize=(24, 5))
    fig.suptitle(title, fontsize=24)
    for ind, img in enumerate(img_ids[:n]):
        plt.subplot(1, 5, ind+1)
        image = plt.imread(img) # read image
        plt.axis('off')
        plt.imshow(image)


# In[16]:


def plot_image(diagnosis, title):
    temp = train[train['diagnosis']==diagnosis]
    img_ids = ['../input/siim-isic-melanoma-classification/jpeg/train/'+i+'.jpg' for i in temp['image_name']]
    
    plt.figure(figsize = (4, 4))
    image = plt.imread(img_ids[0])
    plt.axis('off')
    plt.title(title, fontsize=16)
    plt.imshow(image)
    plt.show()


# In[17]:


plot_images('melanoma', 'Melanoma', 5)


# In[18]:


plot_images('seborrheic keratosis', 'Seborrheic Keratosis', 5)


# In[19]:


plot_images('lichenoid keratosis', 'Lichenoid Keratosis', 5)


# In[20]:


plot_images('lentigo NOS', 'Lentigo NOS', 5)


# In[21]:


plot_images('solar lentigo', 'Solar Lentigo', 5)


# In[22]:


plot_image('cafe-au-lait macule', 'Cafe-au-lait Macule')


# In[23]:


plot_image('atypical melanocytic proliferation', 'Atypical Melanocytic Proliferation')


# In[ ]:




