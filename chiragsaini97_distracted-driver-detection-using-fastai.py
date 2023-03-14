#!/usr/bin/env python
# coding: utf-8

# In[1]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from fastai.vision import *
from fastai.metrics import error_rate

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.


# In[2]:


os.listdir("../input/")


# In[3]:


base_path = "../input/" + os.listdir("../input")[0] + "/"
os.listdir(base_path)


# In[4]:


drivers_df = pd.read_csv(base_path+"driver_imgs_list.csv")


# In[5]:


drivers_df.head()


# In[6]:


categories = {
"c0": "safe driving",
"c1": "texting - right",
"c2": 'talking on the phone - right',
"c3": "texting - left",
"c4": "talking on the phone - left",
'c5': "operating the radio",
'c6': 'drinking',
'c7': 'reaching behind',
'c8': 'hair and makeup',
'c9': 'talking to passenger'
}


# In[7]:


get_ipython().run_line_magic('pinfo2', 'ImageDataBunch.from_folder')


# In[8]:


imgs_path = base_path + "imgs/"
data = ImageDataBunch.from_folder(imgs_path, train=imgs_path+"train", valid_pct=0.2, test=imgs_path+"test",
                                    ds_tfms=get_transforms(), size=224, bs=16).normalize(imagenet_stats)


# In[9]:


data.show_batch(rows=5, figsize=(8,10))


# In[10]:


print(data.classes)
len(data.classes),data.c


# In[11]:


learn = cnn_learner(data, models.resnet34, metrics=error_rate)


# In[12]:


learn.model


# In[13]:


learn.fit_one_cycle(4)


# In[14]:


learn.model_dir='/kaggle/working/'


# In[15]:


learn.save("stage-1")


# In[16]:


interp = ClassificationInterpretation.from_learner(learn)

losses,idxs = interp.top_losses()

len(data.valid_ds)==len(losses)==len(idxs)


# In[17]:


interp.plot_top_losses(9, figsize=(15,11))


# In[18]:


interp.plot_confusion_matrix(figsize=(12,12), dpi=60)


# In[19]:


confused = interp.most_confused(min_val=2)


# In[20]:


for x in confused:
    print("Real:",categories[x[0]],", Predicted:", categories[x[1]],", Number of times it did it:", x[2])


# In[21]:


learn.lr_find()


# In[22]:


learn.recorder.plot()


# In[23]:


learn.unfreeze()
learn.fit_one_cycle(2, max_lr=slice(1e-5,1e-4))


# In[24]:


learn.save("stage-2")


# In[25]:


get_ipython().system('pip install pytorch2keras')


# In[26]:


get_ipython().system('pip install onnx')


# In[27]:


pytorch_model = learn.model_dir+"stage-2.pth"
keras_output = learn.model_dir+"learn.h5"


# In[28]:


import tensorflow as tf
import torch
import onnx


# In[29]:


# To Be Continued


# In[ ]:




