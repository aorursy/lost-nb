#!/usr/bin/env python
# coding: utf-8

# In[1]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.


# In[2]:


from fastai.vision import *
path = Path('/kaggle/input/')
path.ls()


# In[3]:


df = pd.read_csv('../input/train_v2.csv')
df.head()


# In[4]:


tfms = get_transforms(flip_vert=True, max_lighting=0.1, max_zoom=1.05, max_warp=0.)


# In[5]:


np.random.seed(42)
src = (ImageList.from_csv(path, 'train_v2.csv', folder='train-jpg', suffix='.jpg')
       .split_by_rand_pct(0.2)
       .label_from_df(label_delim=' '))


# In[6]:


data = (src.transform(tfms, size=128)
        .databunch().normalize(imagenet_stats))


# In[7]:


data.show_batch(rows=3, figsize=(12,9))


# In[8]:


arch = models.resnet50


# In[9]:


acc_02 = partial(accuracy_thresh, thresh=0.2)
f_score = partial(fbeta, thresh=0.2)
learn = cnn_learner(data, arch, metrics=[acc_02, f_score], model_dir='/tmp/models')


# In[10]:


learn.lr_find()


# In[11]:


learn.recorder.plot()


# In[12]:


lr = 0.01


# In[13]:


learn.fit_one_cycle(5, slice(lr))


# In[14]:


learn.save('stage-1-resnet50')


# In[15]:


learn.unfreeze()


# In[16]:


learn.lr_find()
learn.recorder.plot()


# In[17]:


learn.fit_one_cycle(5, slice(1e-5,lr/5))


# In[18]:


learn.save('stage-2-resnet50')


# In[19]:


data = (src.transform(tfms, size=256)
        .databunch().normalize(imagenet_stats))

learn.data = data
data.train_ds[0][0].shape


# In[20]:


learn.freeze()


# In[21]:


learn.lr_find()
learn.recorder.plot()


# In[22]:


lr=1e-2/2


# In[23]:


learn.fit_one_cycle(5, slice(lr))


# In[24]:


learn.save('stage-1-256-rn50')


# In[25]:


learn.unfreeze()


# In[26]:


learn.fit_one_cycle(5, slice(1e-5, lr/5))


# In[27]:


learn.recorder.plot_losses()


# In[28]:


learn.save('stage-2-256-rn50')


# In[29]:





# In[29]:





# In[29]:




