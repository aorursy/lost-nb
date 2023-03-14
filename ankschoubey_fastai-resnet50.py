#!/usr/bin/env python
# coding: utf-8

# In[1]:


from fastai import *
from fastai.vision import *

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

base_dir = Path("../input")
save_dir = Path('/kaggle/working')
model_dir= Path('/tmp/models')
base = base_dir
base.ls()


# In[2]:


train_folder, train_path, sample_sub_path, test_path= base.ls()


# In[3]:


train_folder.ls()


# In[4]:


train_folder = train_folder.ls()[0]; train_folder


# In[5]:


train_df = pd.read_csv(train_path); train_df.head()


# In[6]:


np.random.seed(42)
data_il = ImageList.from_df(train_df,train_folder)
data_il = data_il.split_by_rand_pct()
data_il = data_il.label_from_df()
data_il = data_il.transform(get_transforms(),size=32)


# In[7]:


data = data_il.databunch().normalize(imagenet_stats)


# In[8]:


data.show_batch()


# In[9]:


learn = cnn_learner(data, models.resnet50, model_dir='/temp/model',metrics=[accuracy]).to_fp16()


# In[10]:


learn.freeze()
learn.lr_find()


# In[11]:


learn.recorder.plot()


# In[12]:


learn.fit_one_cycle(3, 1e-2)


# In[13]:


learn.recorder.plot_losses()


# In[14]:


learn.save(save_dir/'stage1-64-resnet50')


# In[15]:


learn.load(save_dir/'stage1-64-resnet50')


# In[16]:


learn.unfreeze()


# In[17]:


learn.lr_find()


# In[18]:


learn.recorder.plot()


# In[19]:


learn.fit_one_cycle(2, slice(1e-8,1e-7/5))


# In[20]:


learn.recorder.plot_losses()


# In[21]:


learn.recorder.plot_metrics()


# In[22]:


learn.recorder.plot_lr()


# In[23]:


learn.save(save_dir/'stage2-32-resnet34')


# In[24]:


learn.load(save_dir/'stage2-32-resnet34')


# In[25]:


test_df = pd.read_csv("../input/sample_submission.csv")


# In[26]:


test_path = test_path.ls()[0]; test_path.ls()


# In[27]:


test_img = ImageList.from_df(test_df, path=test_path, folder='')


# In[28]:


test_img[0]


# In[29]:


learn.data.add_test(test_img)


# In[30]:


preds,y = learn.get_preds(ds_type=DatasetType.Test)


# In[31]:


test_df.has_cactus = preds.numpy()[:, 0]


# In[32]:


test_df.to_csv('submission.csv', index=False)


# In[33]:




