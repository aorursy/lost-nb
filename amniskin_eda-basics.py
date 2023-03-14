#!/usr/bin/env python
# coding: utf-8

# In[1]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import seaborn as sns
sns.set_style("darkgrid")
plt.style.use("dark_background")


# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory
COMPETITION_NAME = 'severstal-steel-defect-detection'
import os

# Any results you write to the current directory are saved as output.

df = pd.read_csv(os.path.join('/kaggle/input', COMPETITION_NAME, 'train.csv'))
df.head()


# In[2]:


img_df = df['ImageId_ClassId'].str.split('_', expand=True).rename({0: 'ImageID', 1: 'ClassID'}, axis=1)
img_df['ClassID'] = img_df['ClassID'].astype(int)
img_df.head()


# In[3]:


print(
    'The observation level is (image, class)? -- ',
    img_df.groupby('ImageID')['ClassID'].apply(lambda x: sorted(x.tolist()) == [1, 2, 3, 4]).all()
)


# In[4]:


img_df['ImageID'].unique().shape


# In[5]:


full_df = pd.concat([df, img_df], axis=1, sort=True)
full_df.head()


# In[6]:


full_df.groupby('ClassID')['EncodedPixels'].count()    .plot(kind='bar', title='Number of defects by type', figsize=(13, 8));


# In[7]:


full_df.groupby('ClassID')['EncodedPixels'].apply(lambda x: x.count() / x.shape[0])    .plot(kind='bar', title='Pct defective by type', figsize=(13, 8));


# In[8]:


def read_img(name, test=False, base_dir='../input/severstal-steel-defect-detection/'):
    subdir = 'test_images' if test else 'train_images'
    img_path = os.path.join(base_dir, subdir, name)
    img = mpimg.imread(img_path)
    return img
    
IMG_NAME = '5e581254c.jpg'
img = read_img(IMG_NAME)
display(img.shape)
base_img_shape = (img.shape[0], img.shape[1])


# In[9]:


for i in range(img.shape[-1]):
    plt.figure(figsize=(18,8))
    plt.imshow(img[:,:,i], cmap='nipy_spectral')
    plt.show()


# In[10]:


def all_eq(img):
    return np.all([
        np.array_equal(img[:,:,i], img[:,:,0]) for i in range(img.shape[-1])
    ])
all_eq(img)


# In[11]:


for i, f in enumerate(os.listdir('../input/severstal-steel-defect-detection/train_images/')):
    if not all_eq(read_img(f)):
        print('error with', f)
    if i >= 599:
        break
print('ALL DONE')


# In[12]:


def _get_encoding_mask(encoded_pixels, mask=None, shape=base_img_shape):
    if mask is None:
        mask = np.zeros(shape[0] * shape[1])
    splt_pix = list(map(int, encoded_pixels.split(' ')))
    pixel_map = list(zip(splt_pix[::2], splt_pix[1::2]))
    for start, dur in pixel_map:
        mask[start:start+dur] = 1
    return mask
tmp = _get_encoding_mask(full_df.loc[18550]['EncodedPixels'])


# In[13]:


def get_encoding_mask(img_df, shape=base_img_shape):
    n_classes = img_df.shape[0]
    mask = np.zeros(shape[0] * shape[1] * n_classes).reshape((-1, n_classes))
    for _, row in img_df.iterrows():
        if isinstance(row['EncodedPixels'], str) and len(row['EncodedPixels']) > 0:
            _get_encoding_mask(
                row['EncodedPixels'],
                mask[:, row['ClassID'] - 1],
                shape=shape
            )
    return np.swapaxes(mask.reshape((shape[1], shape[0], n_classes)), 0, 1)
mask = get_encoding_mask(full_df.query(f'ImageID == "{IMG_NAME}"'))


# In[14]:


def plot_img(img, mask=None, mask_index=None, greyscale=False):
    if not greyscale:
        img = img[:, :, 0]
    plt.figure(figsize=(18,8))
    plt.imshow(img, cmap='nipy_spectral')
    plt.grid(None)
    plt.show()
    if greyscale:  # for multiplication later
        img = img[:, :, 0]
    if mask is not None:
        if mask_index is not None:
            lst = [mask_index]
        else:
            lst = range(mask.shape[-1])
        for i in lst:
            plt.figure(figsize=(18,8))
            plt.imshow(mask[:,:,i] * img, cmap='nipy_spectral')
            plt.grid(None)
            plt.title('Mask for class: ' + str(i+1))
            plt.show()
plot_img(img, mask, mask_index=2)


# In[15]:


def load_and_plot(img_name, mask_index=None, test=False, df=full_df,
                  greyscale=False, base_dir='../input/severstal-steel-defect-detection/'):
    img = read_img(img_name, test=test, base_dir=base_dir)
    shape = (img.shape[0], img.shape[1])
    mask = get_encoding_mask(df.query(f'ImageID == "{img_name}"'),
                             shape=shape)
    plot_img(img, mask, mask_index=mask_index, greyscale=greyscale)
load_and_plot(IMG_NAME, greyscale=True)


# In[16]:


load_and_plot('18cc39190.jpg')


# In[17]:


for class_id in full_df['ClassID'].unique():
    img_name = full_df.loc[full_df['EncodedPixels'].notnull() & (full_df['ClassID'] == class_id),
                           'ImageID'].iloc[0]
    print(class_id, img_name)
    load_and_plot(img_name)


# In[18]:


class_id = 1
img_names = full_df.loc[full_df['EncodedPixels'].notnull() & (full_df['ClassID'] == class_id),
                        'ImageID'].iloc[:3]
for img_name in img_names:
    print(class_id, img_name)
    load_and_plot(img_name, mask_index=class_id - 1)


# In[19]:


class_id = 2
img_names = full_df.loc[full_df['EncodedPixels'].notnull() & (full_df['ClassID'] == class_id),
                        'ImageID'].iloc[:3]
for img_name in img_names:
    print(class_id, img_name)
    load_and_plot(img_name, mask_index=class_id - 1)


# In[20]:


class_id = 3
img_names = full_df.loc[full_df['EncodedPixels'].notnull() & (full_df['ClassID'] == class_id),
                        'ImageID'].iloc[:3]
for img_name in img_names:
    print(class_id, img_name)
    load_and_plot(img_name, mask_index=class_id - 1)


# In[21]:


class_id = 4
img_names = full_df.loc[full_df['EncodedPixels'].notnull() & (full_df['ClassID'] == class_id),
                        'ImageID'].iloc[:3]
for img_name in img_names:
    print(class_id, img_name)
    load_and_plot(img_name, mask_index=class_id - 1)


# In[22]:


load_and_plot('0025bde0c.jpg')


# In[23]:


load_and_plot('002af848d.jpg')


# In[24]:


def num_pix_classes(img_df, shape=base_img_shape):
    n_classes = img_df.shape[0]
    mask = np.zeros(shape[0] * shape[1] * n_classes).reshape((-1, n_classes))
    for _, row in img_df.iterrows():
        if isinstance(row['EncodedPixels'], str) and len(row['EncodedPixels']) > 0:
            _get_encoding_mask(
                row['EncodedPixels'],
                mask[:, row['ClassID'] - 1],
                shape=shape
            )
    vals, cnts = np.unique(mask.sum(axis=-1).astype(int), return_counts=True)
    return pd.Series(cnts, index=vals)
mask = num_pix_classes(full_df.query(f'ImageID == "{IMG_NAME}"'))
mask


# In[25]:


counts = full_df.groupby('ImageID').apply(num_pix_classes).unstack()
counts.head()


# In[26]:


tmp = counts.dropna()
(tmp[1] / (tmp.sum(axis=1))).plot(kind='hist', bins=100, figsize=(18,8),
                                  title='Number of Images vs Pct Defective')
del tmp


# In[27]:


(counts.fillna(0)[1] / (counts.sum(axis=1))).plot(kind='hist', bins=100, figsize=(18,8),
                                                  title='Number of Images vs Pct Defective');

