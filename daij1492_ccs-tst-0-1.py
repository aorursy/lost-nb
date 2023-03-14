#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import PIL
import matplotlib.pyplot as plt
import keras


# In[2]:


def read_jpg(path):
    return np.asarray(PIL.Image.open(path),dtype=np.uint8)    


# In[3]:


import os


# In[4]:


tr_ty1 = [x for x in os.listdir('../input/train/Type_1') if '.jpg' in x]


# In[5]:


for j,f in enumerate(tr_ty1):
    img = read_jpg(path='../input/train/Type_1/'+f)
    print(j,f,img.shape)


# In[6]:


# Something wrong /w this particular image
read_jpg('../input/train/Type_1/'+tr_ty1[60])


# In[7]:


read_jpg('../input/train/Type_1/'+tr_ty1[61])


# In[8]:


tr_ty1[61]


# In[9]:


import cv2
def get_image_data(fname):
    img = cv2.imread(fname)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img


# In[10]:


img = get_image_data('../input/train/Type_1/'+tr_ty1[61])


# In[11]:


plt.imshow(img)


# In[12]:


for j,f in enumerate(tr_ty1):
    img = get_image_data(fname='../input/train/Type_1/'+f)
    print(j,f,img.shape)


# In[13]:


# The following is from previous version of testing notebook


# In[14]:


len([x for x in os.listdir('../input/train/Type_3') if '.jpg' in x])


# In[15]:


[x for x in os.listdir('../input/train/Type_1') if '.jpg' not in x] + [x for x in os.listdir('../input/train/Type_2') if '.jpg' not in x] + [x for x in os.listdir('../input/train/Type_3') if '.jpg' not in x]


# In[16]:


len([x for x in os.listdir('../input/additional/Type_1') if '.jpg' in x]), len([x for x in os.listdir('../input/additional/Type_2') if '.jpg' in x]), len([x for x in os.listdir('../input/additional/Type_3') if '.jpg' in x])


# In[17]:


[x for x in os.listdir('../input/additional/Type_1') if '.jpg' not in x] + [x for x in os.listdir('../input/additional/Type_2') if '.jpg' not in x] + [x for x in os.listdir('../input/additional/Type_3') if '.jpg' not in x]


# In[18]:


test_files = os.listdir('../input/test/')
len(test_files)


# In[19]:


[x for x in test_files if 'jpg' not in x]


# In[20]:


#plt.imshow('../input/test/0.jpg') # Doesn't work directly


# In[21]:





# In[21]:


img = read_jpg('../input/test/0.jpg')


# In[22]:


plt.imshow(img)


# In[23]:


img.shape


# In[24]:


plt.imshow(read_jpg('../input/test/1.jpg'))


# In[25]:


plt.imshow(read_jpg('../input/test/2.jpg'))


# In[26]:


plt.imshow(read_jpg('../input/test/3.jpg'))


# In[27]:


subm = pd.read_csv('../input/sample_submission.csv')


# In[28]:





# In[28]:


subm.columns


# In[29]:


subm.head()


# In[30]:


0.168805	+ 0.527346	+ 0.303849


# In[31]:


subm.shape


# In[32]:


os.listdir('../config')


# In[33]:





# In[33]:


os.listdir('../lib')


# In[34]:





# In[34]:





# In[34]:





# In[34]:





# In[34]:




