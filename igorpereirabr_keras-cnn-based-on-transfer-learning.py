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


from keras.applications.xception import Xception
from keras.models import Model
from keras.layers import Dense
from keras.layers import Flatten
from keras.optimizers import SGD,Adam,Adadelta
from keras.callbacks import History 
from keras.preprocessing.image import ImageDataGenerator
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
from keras.models import load_model
from keras.models import model_from_json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os


# In[3]:


os.mkdir('dataset_dogs_vs_cats/')


# In[4]:


#Copy files
cp -avr ../input/train/train/ /kaggle/working/dataset_dogs_vs_cats


# In[5]:


# organize dataset into a useful structure
from os import makedirs
from os import listdir
from shutil import copyfile
from random import seed
from random import random
# create directories
dataset_home = '/kaggle/working/dataset_dogs_vs_cats/'
subdirs = ['train/', 'test/']
for subdir in subdirs:
    # create label subdirectories
    labeldirs = ['dogs/', 'cats/']
    for labldir in labeldirs:
        newdir = dataset_home + subdir + labldir
        makedirs(newdir, exist_ok=True)


# In[6]:


os.listdir("/kaggle/working/dataset_dogs_vs_cats/")


# In[7]:


#rm -r '/kaggle/working/dataset_dogs_vs_cats/'

# seed random number generator
seed(1)
# define ratio of pictures to use for validation
val_ratio = 0.25
# copy training dataset images into subdirectories
src_directory = '/kaggle/working/dataset_dogs_vs_cats/'
for file in listdir(src_directory):
    src = src_directory + file
    if random() < val_ratio:
        dst_dir = 'test/'     
    else:
        dst_dir = 'train/'
        
    if file.startswith('cat'):
        dst = dataset_home + dst_dir + 'cats/'  + file
        copyfile(src, dst)
    elif file.startswith('dog'):
        dst = dataset_home + dst_dir + 'dogs/'  + file
        copyfile(src, dst)


# In[8]:


#Define train and test path
train_path = '/kaggle/working/dataset_dogs_vs_cats/train/'
test_path  = '/kaggle/working/dataset_dogs_vs_cats/test/'


# In[9]:


# plot dog photos from the dogs vs cats dataset
plt.figure(figsize=(18,10))
# define location of dataset
folder = train_path+'dogs/'
# plot first few images
for i in range(9):
    # define subplot
    pyplot.subplot(330 + 1 + i)
    # define filename
    filename = random.choice(os.listdir(folder))
    file = folder + filename
    # load image pixels
    image = imread(file)
    # plot raw pixel data
    pyplot.imshow(image)
# show the figure
pyplot.show()


# In[10]:


# plot dog photos from the dogs vs cats dataset
plt.figure(figsize=(18,10))
# define location of dataset
folder = train_path+'cats/'
# plot first few images
for i in range(9):
    # define subplot
    pyplot.subplot(330 + 1 + i)
    # define filename
    filename = random.choice(os.listdir(folder))
    file = folder + filename
    # load image pixels
    image = imread(file)
    # plot raw pixel data
    pyplot.imshow(image)
# show the figure
pyplot.show()


# In[11]:


# define cnn model
def Model_CNN_SGD_No_Regularization():
    # load model
    model = Xception(include_top=False,weights='imagenet', input_shape=(299, 299, 3))
    # mark loaded layers as not trainable
    for layer in model.layers:
        layer.trainable = False
    # add new classifier layers
    flat1 = Flatten()(model.layers[-1].output)
    class1 = Dense(128, activation='relu', kernel_initializer='he_uniform')(flat1)
    output = Dense(1, activation='sigmoid')(class1)
    # define new model
    model = Model(inputs=model.inputs, outputs=output)
    # compile model
    opt = SGD(lr=0.001, momentum=0.9)
    model.compile(optimizer=opt, loss='binary_crossentropy', metrics=['accuracy'])
    return model


# In[12]:


# define cnn model
def Model_CNN_ADAM_No_Regularization():
    # load model
    model = Xception(include_top=False,weights='imagenet', input_shape=(299, 299, 3))
    # mark loaded layers as not trainable
    for layer in model.layers:
        layer.trainable = False
    # add new classifier layers
    flat1 = Flatten()(model.layers[-1].output)
    class1 = Dense(128, activation='relu', kernel_initializer='he_uniform')(flat1)
    output = Dense(1, activation='sigmoid')(class1)
    # define new model
    model = Model(inputs=model.inputs, outputs=output)
    # compile model
    opt = Adam(lr=0.001, decay=0.0)
    model.compile(optimizer=opt, loss='binary_crossentropy', metrics=['accuracy'])
    return model


# In[13]:


# define cnn model
def Model_CNN_ADADELTA_No_Regularization():
    # load model
    model = Xception(include_top=False,weights='imagenet', input_shape=(299, 299, 3))
    # mark loaded layers as not trainable
    for layer in model.layers:
        layer.trainable = False
    # add new classifier layers
    flat1 = Flatten()(model.layers[-1].output)
    class1 = Dense(128, activation='relu', kernel_initializer='he_uniform')(flat1)
    output = Dense(1, activation='sigmoid')(class1)
    # define new model
    model = Model(inputs=model.inputs, outputs=output)
    # compile model
    opt = Adadelta(lr=0.001, decay=0.0)
    model.compile(optimizer=opt, loss='binary_crossentropy', metrics=['accuracy'])
    return model


# In[14]:


from keras.preprocessing.image import ImageDataGenerator, load_img


#For this case, we'll use Data Augmentation
training_datagen = ImageDataGenerator(rescale=1./255
                                      #,validation_split=0.1
                                      ,data_format='channels_last'
        ,shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        featurewise_center=True)


test_datagen = ImageDataGenerator(rescale=1./255
                                   #,validation_split=0.1
                                  ,data_format='channels_last')


training_set = training_datagen.flow_from_directory(
        directory=train_path,
        target_size=(299, 299),
        batch_size=64,
        #classes=['Dog','Cat'],
        subset = "training",
        #save_to_dir = os.path.join(dataset_path,'train'),
        class_mode='binary')

test_set = test_datagen.flow_from_directory(
       directory=test_path,
        target_size=(299, 299),
        batch_size=64,
        #classes=['Dog','Cat'],
        #subset = "validation",
        #save_to_dir = os.path.join(dataset_path,'test'),
        class_mode='binary')


# In[15]:


#Define Models
Model_CNN_SGD_No_Regularization = Model_CNN_SGD_No_Regularization()
Model_CNN_ADAM_No_Regularization = Model_CNN_ADAM_No_Regularization()
Model_CNN_ADADELTA_No_Regularization = Model_CNN_ADADELTA_No_Regularization()


# In[16]:


history_1 = History()
history_2 = History()
history_3 = History()
epochs = 10


# In[17]:


Model_CNN_SGD_No_Regularization.fit_generator(training_set,steps_per_epoch=len(training_set),epochs=epochs,validation_data=test_set,validation_steps=len(test_set),callbacks=[history_1])


# In[18]:


Model_CNN_ADAM_No_Regularization.fit_generator(training_set,steps_per_epoch=len(training_set),epochs=epochs,validation_data=test_set,validation_steps=len(test_set),callbacks=[history_2])


# In[19]:


Model_CNN_ADAM_No_Regularization.fit_generator(training_set,steps_per_epoch=len(training_set),epochs=epochs,validation_data=test_set,validation_steps=len(test_set),callbacks=[history_2])


# In[20]:





# In[20]:


df = pd.DataFrame()
df['class'] = [os.listdir(train_path)[j][:3] for j in range(len(train_path))]
df['filepath'] = [os.listdir(train_path)[j][:3] for j in range(len(train_path))]
#Rename files


# In[21]:


def rename_image_files(path):
"""This fucntion will be used to rename images from the train dataset"""
    i=0
    os.listdir(path)
    for filename in os.listdir(train_path): 
        dst = filename[4:]+str("_") + str(i) + ".jpg"
        src =path + filename 
        dst =path+ dst 

        # rename() function will 
        # rename all the files 
        os.rename(src, dst) 
        i += 1
    return None


# In[22]:


df


# In[23]:




