#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.


# In[ ]:


from fastai.vision import *


# In[ ]:


get_ipython().system('unzip -q /kaggle/input/tgs-salt-identification-challenge/train.zip')


# In[ ]:


path_img = "/kaggle/working/images"
path_lbl = "/kaggle/working/masks"


# In[ ]:


fnames = get_image_files(path_img)
fnames[:3]


# In[ ]:


lbl_names = get_image_files(path_lbl)
lbl_names[:3]


# In[ ]:


get_y_fn = lambda x: path_lbl + '/'+ f'{x.stem}{x.suffix}'


# In[ ]:


# Function to get label masks is running fine
x = fnames[0]       
get_y_fn(x)


# In[ ]:


# Load an image
img_f = fnames[2]
img = open_image(img_f, div=True)
img.show(figsize=(5,5))
print(img.shape)


# In[ ]:


# Load corresponding masks

mask = open_mask(get_y_fn(img_f))
mask.show(figsize=(5,5), cmap='gray')
print(mask.shape)


# In[ ]:


# Check the mask data
mask.data


# In[ ]:


# Resize all the masks by dividing by 255 and replacing the original masks
for i in fnames:
    mask = open_mask(get_y_fn(i), div=True)
    mask.save(get_y_fn(i))


# In[ ]:


print(len(fnames))


# In[ ]:


i = fnames[8]
img = open_image(i)
img.show()


# In[ ]:



mask = open_mask(get_y_fn(i))
mask.show()


# In[ ]:


mask.data


# In[ ]:


bs = 4


# In[49]:


data = (SegmentationItemList.from_folder(path_img)
       .split_by_rand_pct(0.2)
       .label_from_func(get_y_fn, classes = ['0','255'])
       .transform(get_transforms(), tfm_y=True)
       .databunch(bs = bs)
       .normalize(imagenet_stats))


# In[50]:


data.train_ds.x[1].data


# In[51]:


data.train_ds.y[1].data


# In[52]:


data.show_batch(2, cmap='gray')     # Shows 2 rows and 2 cols


# In[53]:


data.show_batch(2, figsize=(10,7), ds_type=DatasetType.Valid)   # Display valid data


# In[54]:


metrics = dice
wd = 1e-2


# In[58]:


# learn.destroy()         # If you are reusing the same learner


# In[59]:


learn = unet_learner(data, models.resnet34, metrics=metrics, wd=wd)


# In[60]:


learn.lr_find()                
learn.recorder.plot()


# In[61]:


lr = 3e-4


# In[62]:


learn.fit_one_cycle(5, slice(lr))


# In[63]:


learn.save('stage-1')
learn.load('stage-1');


# In[64]:


learn.show_results()  #rows=10, figsize=(8,9), cmap='Gray')


# In[66]:


get_ipython().run_line_magic('pinfo2', 'learn.freeze_to')


# In[67]:


learn.freeze_to(2)


# In[68]:


lrs = slice(lr/100, lr/10)
lrs


# In[69]:


learn.fit_one_cycle(3, lrs)


# In[48]:


learn.show_results()


# In[70]:


learn.summary()


# In[71]:


learn.recorder.plot_losses()


# In[ ]:




