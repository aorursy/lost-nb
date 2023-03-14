#!/usr/bin/env python
# coding: utf-8

# In[1]:


import sys
import os
import argparse
from glob import glob

import random
from random import shuffle

import time
import datetime

from collections import Counter

# from numpy.random import seed
# seed(101)
# from tensorflow import set_random_seed
# set_random_seed(101)


import numpy as np
import pandas as pd

import shutil
from tqdm import tqdm

import itertools


# import inspect
import gc

import re

from PIL import Image
import cv2

import keras

# from keras.utils import np_utils

from keras.preprocessing.image import ImageDataGenerator

# from imgaug import augmenters as iaa
# import imgaug as ia


from keras import models
from keras.models import Model
from keras.models import Sequential


from keras.layers import Convolution1D, concatenate, SpatialDropout1D, GlobalMaxPool1D, GlobalAvgPool1D, Embedding,     Conv2D, SeparableConv1D, Add, BatchNormalization, Activation, GlobalAveragePooling2D, LeakyReLU, Flatten
from keras.layers import Dense, Input, Dropout, MaxPooling2D, Concatenate, GlobalMaxPooling2D, GlobalAveragePooling2D,     Lambda, Multiply, LSTM, Bidirectional, PReLU, MaxPooling1D
from keras.layers.pooling import _GlobalPooling1D


from keras.applications.xception import Xception
from keras.applications.resnet50 import ResNet50
from keras.applications.inception_v3 import InceptionV3
from keras.applications.inception_resnet_v2 import InceptionResNetV2
from keras.applications.densenet import DenseNet201
from keras.applications.nasnet import NASNetMobile, NASNetLarge

from keras.applications.nasnet import preprocess_input



# from keras.constraints import maxnorm


from keras import optimizers
from keras.optimizers import Adam, SGD , RMSprop

from keras.losses import mae, sparse_categorical_crossentropy, binary_crossentropy


from keras import backend as K
# K.set_image_dim_ordering('th')


from keras.callbacks import ModelCheckpoint, EarlyStopping, TensorBoard, ReduceLROnPlateau

# from keras.wrappers.scikit_learn import KerasClassifier

from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from sklearn.utils import class_weight as cw


from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix, classification_report


from mlxtend.plotting import plot_confusion_matrix

import tensorflow as tf



from IPython.display import display

import seaborn as sns

from matplotlib.pyplot import figure
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

get_ipython().run_line_magic('matplotlib', 'inline')


# In[2]:


print(os.listdir(r"../input"))


# In[3]:


input_directory = r"../input/"

training_dir = input_directory + r"train"
testing_dir = input_directory + r"test"


# In[4]:


train_files = os.listdir(training_dir)
test_files = os.listdir(testing_dir)

train_labels = []

for file in train_files:
    train_labels.append(file.split(".")[0])
    
df_train = pd.DataFrame({"id": train_files, "label": train_labels})

df_train.head()


# In[5]:


df_test = pd.DataFrame({"id": test_files})
df_test["label"] = ["cat"]*(len(test_files))
df_test.head()


# In[6]:


# print date and time for given type of representation
def date_time(x):
    if x==1:
        return 'Timestamp: {:%Y-%m-%d %H:%M:%S}'.format(datetime.datetime.now())
    if x==2:    
        return 'Timestamp: {:%Y-%b-%d %H:%M:%S}'.format(datetime.datetime.now())
    if x==3:  
        return 'Date now: %s' % datetime.datetime.now()
    if x==4:  
        return 'Date today: %s' % datetime.date.today()
    
# prints a integer for degugging
def debug(x):
    print("-"*40, x, "-"*40)


# In[7]:


def get_class_files(df, file="id", label = "label", count=5):
    label_map = {}
    
    for l in set(df[label]):
        label_map[l] = []

    for i in range(len(df)):
        label_map[df[label][i]].append(df[file][i])
    
    return label_map

def select_image_by_label(label_map, image_count_per_label):
    label_map_select = {}
    
    for l in label_map:
        label_map_select[l] = []
        
    
    for i in label_map:
        num_image = len(label_map[i])
        image_mem = {}
        for j in range(image_count_per_label):
            image_index = random.randint(0, num_image - 1)
            while image_index in image_mem:
                image_index = random.randint(0, num_image - 1)
            label_map_select[i].append(label_map[i][image_index])
            
    return label_map_select


def plot_sample_image(label_map_select, directory=training_dir, nrows=2, ncols=5, figsize=(16, 4), aspect=None):
    fig, ax = plt.subplots(nrows=nrows, ncols=ncols, figsize=(16, 4))    
    i=0
    
    for label in label_map_select:
        print(label)
        j=0
        
        for file in label_map_select[label]:
            plt.subplot(i+1, ncols, j+1)
            plot_image(file, directory, aspect=aspect)
            j=j+1
            
        plt.tight_layout()
        plt.show()
        
        i+=1
        
def plot_sample_test_image(directory=testing_dir, count=5, nrows=1, ncols=5, figsize=(16, 4), aspect=None):
    selected_files = random.sample(os.listdir(testing_dir), count)
    fig, ax = plt.subplots(nrows=nrows, ncols=ncols, figsize=(16, 4))
    
    i=0
    
    for file in selected_files:        
        plt.subplot(1, ncols, i+1)
        path = directory + file
        plot_image(file, directory, aspect=aspect)

        i=i+1

    plt.tight_layout()
    plt.show()
    
    
def plot_image(file, directory=None, sub=False, aspect=None):
    path = directory + file
    
    img = plt.imread(path)
    
    plt.imshow(img, aspect=aspect)
    
    plt.xticks([])
    plt.yticks([])
    
    plt.title(path.split("/")[-1])
    
    if sub:
        plt.show()


# In[8]:


print("Trainning")
label_map = get_class_files(df_train)
image_count_per_label = 5
aspect = 'auto'
label_map_select = select_image_by_label(label_map, image_count_per_label)
plot_sample_image(label_map_select, training_dir+"/", aspect=aspect)


# In[9]:


print("Testing")
aspect = None
plot_sample_test_image(directory=testing_dir+"/", aspect=aspect)


# In[10]:


sns.countplot(df_train["label"])


# In[11]:


# reset tensorflow graph tp free up memory and resource allocation 
def reset_graph(model=None):
    if model:
        try:
            del model
        except:
            return False
    
    tf.reset_default_graph()
    
    K.clear_session()
    
    gc.collect()
    
    return True


# reset callbacks 
def reset_callbacks(checkpoint=None, reduce_lr=None, early_stopping=None, tensorboard=None):
    checkpoint = None
    reduce_lr = None
    early_stopping = None
    tensorboard = None
    


# In[12]:


# reset_graph()
# reset_callbacks()


# In[13]:


classes = ['cat', 'dog']


# In[14]:


def get_data(batch_size=32, target_size=(96,96), class_mode="categorical", training_dir=training_dir, testing_dir=testing_dir, classes=classes, df_train=df_train, df_test=df_test):
    print("Generating data following preprocessing...\n")
    
    rescale = 1.0/255

    train_batch_size = batch_size
    test_batch_size = batch_size

    train_shuffle = True
    val_shuffle = True
    test_shuffle = False
    
    train_datagen = ImageDataGenerator(
        horizontal_flip=True,
#         vertical_flip=True,
#         rotation_range=45,
        shear_range=0.2,
        zoom_range=0.2,
        rescale=rescale,
        validation_split=0.25
    )

    train_generator = train_datagen.flow_from_dataframe(
        df_train, 
        training_dir, 
        x_col='id',
        y_col='label', 
        has_ext=True, 
        target_size=target_size, 
        classes = classes,
        class_mode=class_mode, 
        batch_size=batch_size, 
        shuffle=True, 
        seed=42,
        subset='training')
    
    validation_generator = train_datagen.flow_from_dataframe(
        df_train, 
        training_dir, 
        x_col='id',
        y_col='label', 
        has_ext=True, 
        target_size=target_size, 
        classes = classes,
        class_mode=class_mode, 
        batch_size=batch_size, 
        shuffle=True, 
        seed=42,
        subset='validation')


    test_datagen = ImageDataGenerator(rescale=rescale)
    test_generator = test_datagen.flow_from_dataframe(
        df_test, 
        testing_dir, 
        x_col='id',
        y_col='label', 
        has_ext=True, 
        target_size=target_size, 
        classes = classes,
        class_mode=class_mode, 
        batch_size=batch_size, 
        shuffle=False
#         seed=42
    )
    
    class_weights = get_weight(train_generator.classes)
    
    steps_per_epoch = len(train_generator)
    validation_steps = len(validation_generator)
    
    print("\nData batches generated.")
    
    return train_generator, validation_generator, test_generator, class_weights, steps_per_epoch, validation_steps


def get_weight(y):
    class_weight_current =  cw.compute_class_weight('balanced', np.unique(y), y)
    return class_weight_current


# In[15]:


def get_conv_model():
    model = Sequential()
    model.add(Conv2D(16, (3, 3), activation='relu', padding="same", input_shape=(3,96,96)))
    model.add(Conv2D(16, (3, 3), padding="same", activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(32, (3, 3), activation='relu', padding="same"))
    model.add(Conv2D(32, (3, 3), padding="same", activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(64, (3, 3), activation='relu', padding="same"))
    model.add(Conv2D(64, (3, 3), padding="same", activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(96, (3, 3), dilation_rate=(2, 2), activation='relu', padding="same"))
    model.add(Conv2D(96, (3, 3), padding="valid", activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(128, (3, 3), dilation_rate=(2, 2), activation='relu', padding="same"))
    model.add(Conv2D(128, (3, 3), padding="valid", activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Flatten())

    model.add(Dense(64, activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(2 , activation='softmax'))


    print(model.summary())
    
    return model


def get_model(model_name, input_shape=(96, 96, 3)):
    inputs = Input(input_shape)
    
    if model_name == "Xception":
        base_model = Xception(include_top=False, input_shape=input_shape)
    elif model_name == "ResNet50":
        base_model = ResNet50(include_top=False, input_shape=input_shape)
    elif model_name == "InceptionV3":
        # base_model = InceptionV3(include_top=False, input_shape=input_shape)
        # included weights
        base_model = InceptionV3(weights='imagenet', include_top=False, input_shape=input_shape) 
    elif model_name == "InceptionResNetV2":
        base_model = InceptionResNetV2(include_top=False, input_shape=input_shape)
    if model_name == "DenseNet201":
        base_model = DenseNet201(include_top=False, input_shape=input_shape)
    if model_name == "NASNetMobile":
        base_model = NASNetMobile(include_top=False, input_shape=input_shape)
    if model_name == "NASNetLarge":
        base_model = NASNetLarge(include_top=False, input_shape=input_shape)

    for layer in base_model.layers:
        layer.trainable = False

    # this is the model we will train
    model = Model(inputs=base_model.input, outputs=predictions)

    model.summary()
    
    return model


# In[16]:


def auroc(y_true, y_pred):
    return tf.py_func(roc_auc_score, (y_true, y_pred), tf.double)


# In[17]:


def plot_performance(history=None, figure_directory=None):
    xlabel = 'Epoch'
    legends = ['Training', 'Validation']

    ylim_pad = [0.01, 0.1]


    plt.figure(figsize=(15, 5))

    # Plot training & validation Accuracy values

    y1 = history.history['acc']
    y2 = history.history['val_acc']

    min_y = min(min(y1), min(y2))-ylim_pad[0]
    max_y = max(max(y1), max(y2))+ylim_pad[0]


    plt.subplot(121)

    plt.plot(y1)
    plt.plot(y2)

    plt.title('Model Accuracy\n'+date_time(1), fontsize=17)
    plt.xlabel(xlabel, fontsize=15)
    plt.ylabel('Accuracy', fontsize=15)
    plt.ylim(min_y, max_y)
    plt.legend(legends, loc='upper left')
    plt.grid()


    # Plot training & validation loss values

    y1 = history.history['loss']
    y2 = history.history['val_loss']

    min_y = min(min(y1), min(y2))-ylim_pad[1]
    max_y = max(max(y1), max(y2))+ylim_pad[1]


    plt.subplot(122)

    plt.plot(y1)
    plt.plot(y2)

    plt.title('Model Loss\n'+date_time(1), fontsize=17)
    plt.xlabel(xlabel, fontsize=15)
    plt.ylabel('Loss', fontsize=15)
    plt.ylim(min_y, max_y)
    plt.legend(legends, loc='upper left')
    plt.grid()
#     plt.savefig(figure_directory+"/history")

    plt.show()


# In[18]:


main_model_dir = r"models/"
main_log_dir = r"logs/"

try:
    shutil.rmtree(main_model_dir)
except:
    pass
try:
    shutil.rmtree(main_log_dir)
except:
    pass

os.mkdir(main_model_dir)
os.mkdir(main_log_dir)

os.listdir()


# In[19]:


model_dir = main_model_dir + time.strftime('%Y-%m-%d %H-%M-%S') + "/"
log_dir = main_log_dir + time.strftime('%Y-%m-%d %H-%M-%S')

os.mkdir(model_dir)
os.mkdir(log_dir)

model_file = model_dir + "{epoch:02d}-val_acc-{val_acc:.2f}-val_loss-{val_loss:.2f}.hdf5"

model_dir, os.listdir(model_dir), log_dir, os.listdir(log_dir)


# In[20]:


# reset_graph()
# reset_callbacks()


# In[21]:


print("Settting Callbacks")

checkpoint = ModelCheckpoint(
    model_file, 
    monitor='val_acc', 
    save_best_only=True)


tensorboard = TensorBoard(
    log_dir=log_dir,
    update_freq='batch')


early_stopping = EarlyStopping(
    monitor='val_loss',
    patience=2,
    verbose=1,
    restore_best_weights=True)


reduce_lr = ReduceLROnPlateau(
    monitor='val_loss',
    factor=0.5,
    patience=1,
    verbose=1)

callbacks = [reduce_lr, early_stopping, checkpoint]

print("Completed")
model_dir, os.listdir(model_dir), log_dir, os.listdir(log_dir)


# In[22]:


# print("Getting Base Model...")
# model = get_model("Xception")
# model = get_model("ResNet50")
# model = get_model("InceptionV3")
# model = get_model("InceptionResNetV2")
# model = get_model("DenseNet201")
# model = get_model("NASNetMobile")
# model = get_model("NASNetLarge")
# print("complete")


# In[23]:


print("Starting...\n")
start_time = time.time()
# print(date_time(1))

batch_size = 32
target_size = (299, 299)

train_generator, validation_generator, test_generator, class_weights, steps_per_epoch, validation_steps = get_data(batch_size=batch_size, target_size=target_size, classes=classes, df_test=df_test)
print("\n\nCompleted ...\n")


# In[24]:


loss = 'categorical_crossentropy'
metrics = ['accuracy']


# In[25]:


print("Starting...\n")
start_time = time.time()
# print(date_time(1))


# create the base pre-trained model
# input_shape = (96, 96, 3)
# base_model = InceptionV3(weights='imagenet', include_top=False, input_shape=input_shape)
base_model = InceptionV3(weights='imagenet', include_top=False)

# add a global spatial average pooling layer
x = base_model.output
x = GlobalAveragePooling2D()(x)
# let's add a fully-connected layer
x = Dense(1024, activation='relu')(x)
# and a logistic layer -- let's say we have 200 classes
predictions = Dense(2, activation='softmax')(x)

# this is the model we will train
model = Model(inputs=base_model.input, outputs=predictions)

# first: train only the top layers (which were randomly initialized)
# i.e. freeze all convolutional InceptionV3 layers
for layer in base_model.layers:
    layer.trainable = False

print("\n\nCompliling Model ...\n")
learning_rate = 0.0001
optimizer = Adam(learning_rate)
model.compile(optimizer=optimizer, loss=loss, metrics=metrics)

# train the model on the new data for a few epochs
verbose = 1
epochs = 30

print("Trainning Model ...\n")
history = model.fit_generator(
    train_generator,
    steps_per_epoch=steps_per_epoch,
    epochs=epochs,
    verbose=verbose,
    callbacks=callbacks,
    validation_data=validation_generator,
    validation_steps=validation_steps, 
    class_weight=class_weights)


elapsed_time = time.time() - start_time
elapsed_time = time.strftime("%H:%M:%S", time.gmtime(elapsed_time))

print("\nElapsed Time: " + elapsed_time)
# print("Completed\n", date_time(1))
print("\n")


# In[26]:


# print("Starting...\n")
# start_time = time.time()
# # print(date_time(1))


# # for i, layer in enumerate(base_model.layers):
# #    print(i, layer.name)

# layer_to_Freeze=172 

# for layer in model.layers[:layer_to_Freeze]:
#     layer.trainable = False
# for layer in model.layers[layer_to_Freeze:]:
#     layer.trainable = True


# print("\n\nCompliling Model ...\n")
# learning_rate = 0.00001
# optimizer = Adam(learning_rate)
# model.compile(optimizer=optimizer, loss=loss, metrics=metrics)

# verbose = 1
# epochs = 30

# print("Trainning Model ...\n")
# history = model.fit_generator(
#     train_generator,
#     steps_per_epoch=steps_per_epoch,
#     epochs=epochs,
#     verbose=verbose,
#     callbacks=callbacks,
#     validation_data=validation_generator,
#     validation_steps=validation_steps, 
#     class_weight=class_weights)


# elapsed_time = time.time() - start_time
# elapsed_time = time.strftime("%H:%M:%S", time.gmtime(elapsed_time))

# print("\nElapsed Time: " + elapsed_time)
# # print("Completed\n", date_time(1))
# print("\n")


# In[27]:


plot_performance(history=history)


# In[28]:


def generate_result(model, test_generator, nsteps=len(test_generator)):
    y_preds = model.predict_generator(test_generator, steps=nsteps, verbose=1) 
    return y_preds, y_preds[:,1]


# In[29]:


y_preds_all, y_preds = generate_result(model, test_generator)


# In[30]:


df_test = pd.DataFrame({"id": test_generator.filenames, "label": y_preds})
df_test['id'] = df_test['id'].map(lambda x: x.split('.')[0])
df_test['id'] = df_test['id'].astype(int)
df_test = df_test.sort_values('id')
df_test.to_csv('submission.csv', index=False)
# df_test.to_csv('submission.csv')
df_test.head()


# In[31]:


df_test2 = pd.read_csv("submission.csv")
df_test2.head()


# In[32]:


import random

l = {0: "Cat", 1: "Dog"} 

n = len(test_files)
f = list(np.arange(1,n))

c = 20
r =random.sample(f, c)
nrows = 4
ncols = 5
fig, ax = plt.subplots(nrows=nrows, ncols=ncols, figsize=(nrows*5, ncols*5))    
for i in range(c):
    file = str(df_test2['id'][r[i]])+".jpg"
    path = testing_dir+"/"+file
    img = plt.imread(path)
    plt.subplot(4, 5, i+1)
    plt.imshow(img, aspect='auto')
    plt.xticks([])
    plt.yticks([])
    plt.title(str(df_test2['id'][i])+".jpg" +"\n"+ str(round(df_test2['label'][i], 2)))
plt.show()
    


# In[33]:




