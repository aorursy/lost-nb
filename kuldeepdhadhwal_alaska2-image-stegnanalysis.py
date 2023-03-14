#!/usr/bin/env python
# coding: utf-8

# In[31]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import imageio
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import torch


# In[32]:


PATH = '/kaggle/input/alaska2-image-steganalysis'
model = '/kaggle/input/alaska2-image-steganalysis/e4s-srm/'

train_images = pd.Series(os.listdir(PATH + '/Cover/')).sort_values(ascending=True).reset_index(drop=True)
test_images = pd.Series(os.listdir(PATH + '/Test')).sort_values(ascending=True).reset_index(drop=True)
sample_submission = pd.read_csv(f'{PATH}/sample_submission.csv')


# In[ ]:


es4 = .alexnet(pretrained=True)


# In[27]:


sample_submission.head()


# In[24]:


fig, ax = plt.subplots(nrows=2, ncols=4, figsize=(16, 8))
k=0
for i, row in enumerate(ax):
    for j, col in enumerate(row):
        img = mpimg.imread(PATH + '/Cover/' + train_images[k])
        col.imshow(img)
        col.set_title(train_images[k])
        k=k+1
plt.suptitle('Samples from Cover Images', fontsize=14)
plt.show()


# In[26]:


fig, ax = plt.subplots(nrows=2, ncols=4, figsize=(16, 8))
k=0
for i, row in enumerate(ax):
    for j, col in enumerate(row):
        img = mpimg.imread(PATH + '/Test/' + test_images[k])
        col.imshow(img)
        col.set_title(test_images[k])
        k=k+1
plt.suptitle('Samples from Test Images', fontsize=14)
plt.show()


# In[30]:


for folder in os.listdir(PATH):
    try:
        print(f"Folder {folder} contains {len(os.listdir(PATH + '/' + folder))} images.")
    except:
        print(f'{folder}')


# In[38]:


folders = ['Cover', 'JMiPOD', 'JUNIWARD', 'UERD']
k=0
img_cover = mpimg.imread(PATH + '/' + folders[0] + '/'+ train_images[k])
img_jmipod = mpimg.imread(PATH + '/' + folders[1] + '/'+ train_images[k])
img_juniward = mpimg.imread(PATH + '/' + folders[2] + '/'+ train_images[k])
img_uerd = mpimg.imread(PATH + '/' + folders[3] + '/'+ train_images[k])

fig, ax = plt.subplots(nrows=3, ncols=4, figsize=(16, 12))

ax[0,0].imshow(img_jmipod)
ax[0,1].imshow((img_cover == img_jmipod).astype(int)[:,:,0])
ax[0,1].set_title(f'{train_images[k]} Channel 0')

ax[0,2].imshow((img_cover == img_jmipod).astype(int)[:,:,1])
ax[0,2].set_title(f'{train_images[k]} Channel 1')
ax[0,3].imshow((img_cover == img_jmipod).astype(int)[:,:,2])
ax[0,3].set_title(f'{train_images[k]} Channel 2')
ax[0,0].set_ylabel(folders[1], rotation=90, size='large', fontsize=14)


ax[1,0].imshow(img_juniward)
ax[1,1].imshow((img_cover == img_juniward).astype(int)[:,:,0])
ax[1,2].imshow((img_cover == img_juniward).astype(int)[:,:,1])
ax[1,3].imshow((img_cover == img_juniward).astype(int)[:,:,2])
ax[1,0].set_ylabel(folders[2], rotation=90, size='large', fontsize=14)

ax[2,0].imshow(img_uerd)
ax[2,1].imshow((img_cover == img_uerd).astype(int)[:,:,0])
ax[2,2].imshow((img_cover == img_uerd).astype(int)[:,:,1])
ax[2,3].imshow((img_cover == img_uerd).astype(int)[:,:,2])
ax[2,0].set_ylabel(folders[3], rotation=90, size='large', fontsize=14)

plt.suptitle('Pixel Deviation from Cover Image', fontsize=14)

plt.show()


# In[44]:


print('{} images with Cover Images '.format(train_images.nunique()))
print('{} images with Test Images '.format(test_images.nunique()))

