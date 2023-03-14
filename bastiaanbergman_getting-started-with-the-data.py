#!/usr/bin/env python
# coding: utf-8

# In[1]:


import sys
import os
import subprocess
from six import string_types
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import scipy
from skimage import io
from scipy import ndimage
from IPython.display import display
get_ipython().run_line_magic('matplotlib', 'inline')
from spectral import imshow


# In[2]:


get_ipython().system('ls -lha ../input')


# In[3]:


get_ipython().system('ls -lha ../input/test-tif-v2 | wc -l')


# In[4]:


PLANET_KAGGLE_ROOT = os.path.abspath("../input/")
PLANET_KAGGLE_JPEG_DIR = os.path.join(PLANET_KAGGLE_ROOT, 'train-jpg')
PLANET_KAGGLE_LABEL_CSV = os.path.join(PLANET_KAGGLE_ROOT, 'train_v2.csv')
assert os.path.exists(PLANET_KAGGLE_ROOT)
assert os.path.exists(PLANET_KAGGLE_JPEG_DIR)
assert os.path.exists(PLANET_KAGGLE_LABEL_CSV)


# In[5]:


labels_df = pd.read_csv(PLANET_KAGGLE_LABEL_CSV)
labels_df.head()


# In[6]:


# Build list with unique labels
label_list = []
for tag_str in labels_df.tags.values:
    labels = tag_str.split(' ')
    for label in labels:
        if label not in label_list:
            label_list.append(label)


# In[7]:


# Add onehot features for every label
for label in label_list:
    labels_df[label] = labels_df['tags'].apply(lambda x: 1 if label in x.split(' ') else 0)
# Display head
labels_df.head()


# In[8]:


# Histogram of label instances
labels_df[label_list].sum().sort_values().plot.bar()


# In[9]:


def make_cooccurence_matrix(labels):
    numeric_df = labels_df[labels]; 
    c_matrix = numeric_df.T.dot(numeric_df)
    sns.heatmap(c_matrix)
    return c_matrix

# Compute the co-ocurrence matrix
make_cooccurence_matrix(label_list)


# In[10]:


weather_labels = ['clear', 'partly_cloudy', 'haze', 'cloudy']
make_cooccurence_matrix(weather_labels)


# In[11]:


land_labels = ['primary', 'agriculture', 'water', 'cultivation', 'habitation']
make_cooccurence_matrix(land_labels)


# In[12]:


rare_labels = [l for l in label_list if labels_df[label_list].sum()[l] < 2000]
make_cooccurence_matrix(rare_labels)


# In[13]:


def sample_images(tags, n=None):
    """Randomly sample n images with the specified tags."""
    condition = True
    if isinstance(tags, string_types):
        raise ValueError("Pass a list of tags, not a single tag.")
    for lbl in label_list:
        if lbl in tags:
            condition = condition & (labels_df[lbl] == 1)
        else:
            condition = condition & (labels_df[lbl] == 0)
    if n is not None:
        return labels_df[condition].sample(n)
    else:
        return labels_df[condition]


# In[14]:


sample_images(['clear','primary'], n=10)


# In[15]:


def load_image(filename):
    '''Look through the directory tree to find the image you specified
    (e.g. train_10.tif vs. train_10.jpg)'''
    for dirname in os.listdir(PLANET_KAGGLE_ROOT):
        path = os.path.abspath(os.path.join(PLANET_KAGGLE_ROOT, dirname, filename))
        if os.path.exists(path):
            #print('Found image {}'.format(path))
            return io.imread(path)
    # if you reach this line, you didn't find the image you're looking for
    print('Load failed: could not find image {}'.format(path))
    
def sample_to_fname(sample_df, row_idx, suffix='tif'):
    '''Given a dataframe of sampled images, get the
    corresponding filename.'''
    fname = sample_df.get_value(sample_df.index[row_idx], 'image_name')
    return '{}.{}'.format(fname, suffix)

def display_sample_im(tags, n=None):
    s = sample_images(tags, n=n)
    if n is None:
        n=0
    for i in range(n):
        fname = sample_to_fname(s.iloc[i], 0)
        rgbn_image = load_image(fname)
        imshow(rgbn_image[:,:,:3])
    return rgbn_image, s
    
    


# In[16]:


im, s = display_sample_im(['primary', 'clear'], n=4);
s


# In[17]:


im_m = np.vstack([im[i*16:(i+1)*16,j*16:(j+1)*16,0].ravel() for i in range(16) for j in range(16)])
cov = np.cov(im_m)
eigen_value, eigen_vector = np.linalg.eig(cov)
eigen_value = eigen_value.reshape(-1,1)
significance_ind = eigen_value.argsort(axis=0)[::-1]
eigen_value[significance_ind[:,0]]
# The n_th eigen vector
n = 0
i = significance_ind[n,0]
feature = eigen_vector[:,i:i+1].T
finaldata = np.dot(feature,im_m).T
first_eigen_image = np.dot(feature.T,finaldata.T).T
plt.imshow(im[0:16,0:16,0]);
plt.figure()
plt.imshow(finaldata.reshape(16,16));
plt.figure()
plt.imshow(first_eigen_image)


# In[18]:


def calibrate_image(rgb_image):
   ref_stds = [41.262260630543992, 35.759466445746916, 33.383302346657047]
   ref_means = [80.198569793701168, 87.701977996826173, 76.552578582763672]
   
   # Transform test image to 32-bit floats to avoid 
   # surprises when doing arithmetic with it 
   calibrated_img = rgb_image.copy().astype('float32')

   # Loop over RGB
   for i in range(3):
       # Subtract mean 
       calibrated_img[:,:,i] = calibrated_img[:,:,i]-np.mean(calibrated_img[:,:,i])
       # Normalize variance
       calibrated_img[:,:,i] = calibrated_img[:,:,i]/np.std(calibrated_img[:,:,i])
       # Scale to reference 
       calibrated_img[:,:,i] = calibrated_img[:,:,i]*ref_stds[i] + ref_means[i]
       # Clip any values going out of the valid range
       calibrated_img[:,:,i] = np.clip(calibrated_img[:,:,i],0,255)

   # Convert to 8-bit unsigned int
   return calibrated_img.astype('uint8')


# In[19]:




