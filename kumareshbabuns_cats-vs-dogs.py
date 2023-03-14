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


import zipfile


# In[3]:


extract_files = ['train', 'test1']


# In[4]:


for file in extract_files:
    with zipfile.ZipFile("/kaggle/input/dogs-vs-cats/"+file+".zip", "r") as z:
        z.extractall(".")


# In[5]:


import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# In[6]:


train_directory = "train/"
test_directory = "test1/"
validation_directory = "validation/"


# In[7]:


import os
import shutil


# In[8]:


os.mkdir(validation_directory)


# In[9]:


for dir_name in ['cats','dogs']:
    os.mkdir("train/" + dir_name)
    os.mkdir(validation_directory + dir_name)


# In[10]:


file_list = os.listdir("train/")
for file_name in file_list:
    if(file_name.startswith("cat")):
        shutil.move("train/"+file_name, "train/cats")
    elif(file_name.startswith("dog")):
        shutil.move("train/"+file_name, "train/dogs")


# In[11]:


ls train/dogs | wc -l


# In[12]:


ls train/cats | wc -l


# In[13]:


for i in range(0,6000):
    shutil.move('train/cats/cat.' + str(i) + '.jpg', 'validation/cats')
    shutil.move('train/dogs/dog.' + str(i) + '.jpg', 'validation/dogs')


# In[14]:


ls validation/cats | wc -l


# In[15]:


import tensorflow as tf


# In[16]:


tf.__version__


# In[17]:


from tensorflow.keras.preprocessing.image import ImageDataGenerator


# In[18]:


train_generator = ImageDataGenerator(rescale=1./255)
validation_generator = ImageDataGenerator(rescale=1./255)


# In[19]:


train_generator_data = train_generator.flow_from_directory(train_directory, target_size=(150,150), batch_size=15, class_mode='categorical')
validation_generator_data = validation_generator.flow_from_directory(validation_directory, target_size=(150,150), batch_size=15, class_mode='categorical')


# In[20]:


from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dropout, Flatten, Dense, BatchNormalization


# In[21]:


model = Sequential()

model.add(Conv2D(filters=32, kernel_size=(4,4), input_shape=(150,150,3), activation='relu'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Dropout(0.25))

model.add(Conv2D(filters=64, kernel_size=(4,4), activation='relu'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Dropout(0.25))

model.add(Conv2D(filters=128, kernel_size=(4,4), activation='relu'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Dropout(0.5))

model.add(Flatten())
model.add(Dense(units=512, activation='relu'))
model.add(BatchNormalization())
model.add(Dropout(0.5))

model.add(Dense(units=2, activation='softmax'))


# In[22]:


model.summary()


# In[23]:


model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])


# In[24]:


from tensorflow.keras.callbacks import EarlyStopping


# In[25]:


earlystop = EarlyStopping(patience=5)


# In[26]:


history = model.fit_generator(train_generator_data, epochs=10, validation_data=validation_generator_data, callbacks=[earlystop])


# In[ ]:


# https://www.kaggle.com/uysimty/keras-cnn-dog-or-cat-classification
# https://www.kaggle.com/bulentsiyah/dogs-vs-cats-classification-vgg16-fine-tuning


# In[ ]:




