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
# for dirname, _, filenames in os.walk('/kaggle/input'):
#     for filename in filenames:
#         print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.


# In[2]:


import os
import sys
import numpy as np
import pandas as pd
import cv2
import seaborn as sns

from math import ceil
from tqdm import tqdm

from PIL import Image
from matplotlib import pyplot as plt

from sklearn.model_selection import train_test_split

from keras.applications.inception_v3 import InceptionV3, preprocess_input
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential, Model
from keras.layers import Input, Dense, GlobalAveragePooling2D, Dropout
from keras.optimizers import RMSprop, Adam, SGD
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, EarlyStopping


# In[3]:


data_path = "/kaggle/input/"
train_img_path = os.path.join(data_path,'train_images')
test_img_path = os.path.join(data_path,'test_images')
train_label_path = os.path.join(data_path,'train.csv')
test_label_path = os.path.join(data_path,'test.csv')

df_train = pd.read_csv(train_label_path)
df_test = pd.read_csv(test_label_path)

print("num of train images ", len(os.listdir(train_img_path)))
print("num of test images ",len(os.listdir(test_img_path)))


# In[ ]:





# In[4]:


import matplotlib.pyplot as plt
df_train['diagnosis'].value_counts().plot(kind = 'bar')
plt.title("Level of diagnosis")


# In[5]:


import random
samp = random.sample(df_train['id_code'].tolist(),3)
sub=130
for i in range(len(samp)):
    sub+=1
    plt.figure(figsize=(15,15))
    plt.subplot(sub)
    file_path = "../input/train_images/"+samp[i]+".png"
    img = cv2.imread(file_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    plt.imshow(img)


# In[6]:


import random
samp = random.sample(df_train['id_code'].tolist(),3)
sub=130
for i in range(len(samp)):
    sub+=1
    plt.figure(figsize=(15,15))
    plt.subplot(sub)
    file_path = "../input/train_images/"+samp[i]+".png"
    img = cv2.imread(file_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    plt.imshow(img)


# In[7]:


df_train_img=[]
train_list = df_train["id_code"].tolist()
for item in train_list:
    file_path = "../input/train_images/"+str(item)+".png"
    img = cv2.imread(file_path)
    img = cv2.resize(img,(150,150))
    #print(img)
    df_train_img.append(img)
df_train_img = np.array(df_train_img, np.float32)/255


# In[8]:


df_test_img=[]
for item in df_test["id_code"].tolist():
    file_path = "../input/test_images/"+str(item)+".png"
    img = cv2.imread(file_path)
    img = cv2.resize(img,(150,150))
    df_test_img.append(img)
df_test_img = np.array(df_test_img, np.float32)


# In[9]:


y_train = (df_train.iloc[:,1].values).astype('int32')
# from keras.utils.np_utils import to_categorical
# y_train = to_categorical(y_train)


# In[10]:


y_train


# In[11]:


from sklearn.model_selection import train_test_split
X = df_train_img
Y = y_train
x_train, x_val, y_train, y_val = train_test_split(df_train_img, y_train, test_size = 0.15, random_state = 42)


# In[12]:


# df_train_img.reshape(df_train_img.shape[0],150,150,1)


# In[13]:


# train_datagen = ImageDataGenerator(rescale = 1./255,
#                                   shear_range = 0.2,
#                                   zoom_range = 0.2,
#                                   horizontal_flip = True)
# test_datagen = ImageDataGenerator(rescale = 1./255)


# In[14]:




from keras.preprocessing import image
gen = image.ImageDataGenerator()


# In[15]:


batches = gen.flow(x_train, y_train, batch_size = 64)
val_batches = gen.flow(x_val, y_val, batch_size = 64)


# In[ ]:





# In[16]:


from keras.models import Sequential
from keras.layers import Convolution2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense


# In[17]:


classifier = Sequential()
classifier.add(Convolution2D(32, 3 ,3, input_shape = (150,150,3), activation  = 'relu'))
classifier.add(MaxPooling2D(pool_size = (2,2)))
classifier.add(Convolution2D(32,3,3, activation = 'relu'))
classifier.add(MaxPooling2D(pool_size = (2,2)))
classifier.add(Flatten())
classifier.add(Dense(output_dim = 75, activation = 'relu'))
classifier.add(Dense(output_dim = 5, activation = 'softmax'))

classifier.compile(optimizer = 'nadam', loss = 'sparse_categorical_crossentropy', metrics = ['accuracy'])


# In[18]:




hist = classifier.fit_generator(generator=batches, steps_per_epoch = batches.n,
                             epochs=3, validation_data=val_batches,
                             validation_steps=val_batches.n)


# In[19]:


# predictions.count_values()


# In[20]:


predictions = classifier.predict_classes(df_test_img,verbose=0)
sudmissions = pd.DataFrame({'id_code':df_test.iloc[:,0].tolist(),
                           'diagnosis': predictions})
sudmissions.to_csv("submission.csv", index = False, header = True)


# In[ ]:





# In[ ]:




