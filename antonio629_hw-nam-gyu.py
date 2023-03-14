#!/usr/bin/env python
# coding: utf-8

# In[1]:


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


# In[2]:


pip install imutils


# In[3]:



from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from imutils import paths
import numpy as np
import imutils 
import cv2 
import os
import argparse


# In[4]:


def image_to_feature_vector(image, size=(32, 32)):
    return cv2.resize(image, size).flatten()

dataset_train = "/content/input2/train/"

print("[INFO] describing images...")
imagePaths = list(paths.list_images(dataset_train))

print(len(imagePaths))

rawImages = []
labels = []
for (i, imagePath) in enumerate(imagePaths):

    image = cv2.imread(imagePath)
    label = imagePath.split(os.path.sep)[-1].split(".")[0]
 

    pixels = image_to_feature_vector(image)
    
 

    rawImages.append(pixels)
    labels.append(label)

    if i > 0 and i % 1000 == 0:
        print("[INFO] processed {}/{}".format(i, len(imagePaths)))


# In[5]:



rawImages = np.array(rawImages)
labels = np.array(labels)

print("[INFO] pixels matrix: {:.2f}MB".format(
rawImages.nbytes / (1024 * 1000.0)))
bUseCompleteDataset = False
if bUseCompleteDataset:
    (trainRI, testRI, trainRL, testRL) = train_test_split(rawImages, labels, test_size=0.25, random_state=42)
else:
    rawImages_subset = rawImages[:2000]
    labels_subset = labels[:2000]
    (trainRI, testRI, trainRL, testRL) = train_test_split(rawImages_subset, labels_subset, test_size=0.25, random_state=42)


# In[6]:


print("[INFO] evaluating raw pixel accuracy...")
neighbors = [1, 3, 5, 7, 13]

for k in neighbors:
    model = KNeighborsClassifier(n_neighbors= 5)
    a=model.fit(trainRI, trainRL)
    acc = model.score(testRI, testRL)
    print("[INFO] raw pixel accuracy: {:.2f}%".format(acc * 100))


# In[7]:


dataset_test = "/content/input2/test1/"

imagePaths_test=list(paths.list_images(dataset_test))

print(len(imagePaths_test))
rawImages_test=[]
labels_test=[]

for(i,imagePath_test) in enumerate(imagePaths_test):
    image=cv2.imread(imagePath_test)
    label=imagePath_test.split(os.path.sep)[-1].split(".")[0]

    pixels=image_to_feature_vector(image)

    rawImages_test.append(pixels)
    labels_test.append(label)


    if i>0 and i%1000==0:
      print("[INFO] processed {}/{}".format(i,len(imagePaths_test)))


# In[8]:


model = KNeighborsClassifier(n_neighbors= 5)
a=model.fit(trainRI, trainRL)

result=a.predict(rawImages_test)


# In[9]:


import pandas as pd


result=np.reshape(result,(-1,1))

print(result.shape)
df = pd.DataFrame(result, columns=["label"])
df.index.name = 'id'
df.index += 1
df = df.replace('dog',1)
df = df.replace('cat',0)
df = df.rename({'1':'id','0':'label'})
df.to_csv('results-nk-v2.csv',index=True, header=True)

