#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
import pandas as pd
from scipy.misc import imread
from tqdm import tqdm
import cv2
from skimage.transform import resize
import os
from sklearn.model_selection import train_test_split
from matplotlib import pyplot as plt
import gc


# In[ ]:


gc.enable()


# In[ ]:


INPUT_PATH = '../input/'
TRAIN_PATH = INPUT_PATH + 'train/'
TEST_PATH = INPUT_PATH + 'test/'


# In[ ]:


data = pd.read_csv(INPUT_PATH+"train.csv")


# In[ ]:


# https://www.kaggle.com/iafoss/pretrained-resnet34-with-rgby-0-460-public-lb#
def open_rgby(path, id_): #a function that reads RGBY image
    colors = ['red','green','blue','yellow']
    flags = cv2.IMREAD_GRAYSCALE
    img = [cv2.imread(os.path.join(path, id_ + '_' + color + '.png'), flags).astype(np.float32) / 255
           for color in colors]
    return np.stack(img, axis=-1)


# In[ ]:


get_ipython().system('mkdir train_4channel')


# In[ ]:


for idx in tqdm(data.Id.values):
    data = open_rgby(TRAIN_PATH, idx)
    with open("./train_4channel/"+idx+".npz", 'wb') as f:
        #np.save(f, data)


# In[ ]:


import glob
test_files = glob.glob(TEST_PATH + "*.png")


# In[ ]:


test_ids = list(set([x[13:].split("_")[0] for x in test_files]))


# In[ ]:


get_ipython().system('mkdir test_4channel')


# In[ ]:


for idx in tqdm(test_ids):
    data = open_rgby(TEST_PATH, idx)
    with open("./test_4channel/"+idx+".npz", 'wb') as f:
        #np.save(f, data)

