#!/usr/bin/env python
# coding: utf-8

# In[ ]:


get_ipython().run_line_magic('load_ext', 'autoreload')
get_ipython().run_line_magic('autoreload', '2')
get_ipython().run_line_magic('matplotlib', 'inline')


# In[ ]:


import numpy as np
import pandas as pd
import os
from fastai.vision import *
from pathlib import Path
import matplotlib.pyplot as plt
import cv2
import torch
from sklearn.metrics import f1_score
PATH = Path('../input')
TRAIN = Path('../input/train')
print(os.listdir("../input"))
# Any results you write to the current directory are saved as output.


# In[ ]:


classes = get_ipython().getoutput('ls {TRAIN}')
print('Number of Samples per class :')
for i in classes:
    print(i,':', len(get_image_files(TRAIN/classes[0])))


# In[ ]:


# Looking at one random image
im_path = get_image_files(TRAIN/classes[0])[np.random.randint(0, 263, 1)[0]]
img = cv2.imread(str(im_path))
print(img.shape)
plt.imshow(img)


# In[ ]:


tfms = get_transforms(flip_vert=True, max_rotate=90, max_zoom=1.3)
bs = 64
sz = 224
nw = 0
data = ImageDataBunch.from_folder(TRAIN, test=PATH/'test', valid_pct=0.2, ds_tfms=tfms, bs=bs, size=sz, num_workers=nw).normalize(imagenet_stats)


# In[ ]:


data.show_batch(rows=3)


# In[ ]:


data.classes


# In[ ]:


def f1_micro(y_true, y_pred):
    return fl_score(y_true, y_pred, average='micro')


# In[ ]:


learn = create_cnn(data, models.resnet34, metrics=f1_micro)


# In[ ]:




