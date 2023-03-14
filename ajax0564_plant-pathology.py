#!/usr/bin/env python
# coding: utf-8



import os
import gc
import re

import cv2
import math
import numpy as np
import scipy as sp
import pandas as pd

import tensorflow as tf
from keras.utils import plot_model
import tensorflow.keras.layers as L
from keras.utils import model_to_dot
import tensorflow.keras.backend as K
from tensorflow.keras.models import Model
from kaggle_datasets import KaggleDatasets
from tensorflow.keras.applications import DenseNet121


import seaborn as sns
from tqdm import tqdm
import matplotlib.cm as cm
from sklearn import metrics
import matplotlib.pyplot as plt
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split




IMAGE_PATH = "../input/plant-pathology-2020-fgvc7/images/"
TEST_PATH = "../input/plant-pathology-2020-fgvc7/test.csv"
TRAIN_PATH = "../input/plant-pathology-2020-fgvc7/train.csv"
SUB_PATH = "../input/plant-pathology-2020-fgvc7/sample_submission.csv"

sub = pd.read_csv(SUB_PATH)
df_test= pd.read_csv(TEST_PATH)
df_train = pd.read_csv(TRAIN_PATH)





# Configuration
EPOCHS = 12
BATCH_SIZE = 32




def get_class(row):
    
    if row['multiple_diseases'] == 1:
        return 'multiple_diseases'
    
    elif row['rust'] == 1:
        return 'rust'
    
    elif row['scab'] == 1:
        return 'scab'
    
    else:
        return 'healthy'
    
df_train['target'] = df_train.apply(get_class, axis=1)

df_train.head()




# Filter out each class
df_healthy = df_train[df_train['target'] == 'healthy']
df_multiple_diseases = df_train[df_train['target'] == 'multiple_diseases']
df_rust = df_train[df_train['target'] == 'rust']
df_scab = df_train[df_train['target'] == 'scab']




def plotclass(cate):
    # Filter out each class
    df = df_train[df_train['target'] == cate]
    image_list = list(df['image_id'])
    plt.figure(figsize=(25,10))

# Our subplot will contain 2 rows and 4 columns
# plt.subplot(nrows, ncols, plot_number)
    plt.subplot(2,4,1)

# plt.imread reads an image from a path and converts it into an array

# starting from 1 makes the code easier to write
    for i in range(1,9):
    
         plt.subplot(2,4,i)
    
    # get an image
         image = image_list[i]
         plt.imshow(plt.imread(IMAGE_PATH + image + '.jpg'))
    
         plt.xlabel('{}'.format(cate), fontsize=20)




plotclass('healthy')




plotclass('scab')




plotclass('rust')




plotclass('multiple_diseases')




df_train['target'].value_counts()




y = df_train['target']
# shuffle
df_train = shuffle(df_train)
print(df_train.shape)




df_1 = df_train[df_train['target'] != 'multiple_diseases']

df_2 = df_train[df_train['target'] == 'multiple_diseases']

df_train_up = pd.concat([df_1, df_2,  df_2,  df_2,  df_2,  df_2], axis=0).reset_index(drop=True)

df_train = shuffle(df_train_up)

print(df_train.shape)

df_train.head()




df_train['target'].value_counts()




def edge_and_cut(path):
    img = cv2.imread(path)
    
    emb_img = img.copy()
    edges = cv2.Canny(img, 100, 200)
    edge_coors = []
    for i in range(edges.shape[0]):
        for j in range(edges.shape[1]):
            if edges[i][j] != 0:
                edge_coors.append((i, j))
    
    row_min = edge_coors[np.argsort([coor[0] for coor in edge_coors])[0]][0]
    row_max = edge_coors[np.argsort([coor[0] for coor in edge_coors])[-1]][0]
    col_min = edge_coors[np.argsort([coor[1] for coor in edge_coors])[0]][1]
    col_max = edge_coors[np.argsort([coor[1] for coor in edge_coors])[-1]][1]
    new_img = img[row_min:row_max, col_min:col_max]
    
    emb_img[row_min-10:row_min+10, col_min:col_max] = [255, 0, 0]
    emb_img[row_max-10:row_max+10, col_min:col_max] = [255, 0, 0]
    emb_img[row_min:row_max, col_min-10:col_min+10] = [255, 0, 0]
    emb_img[row_min:row_max, col_max-10:col_max+10] = [255, 0, 0]
    
    fig, ax = plt.subplots(nrows=1, ncols=3, figsize=(30, 20))
    ax[0].imshow(img, cmap='gray')
    ax[0].set_title('Original Image', fontsize=24)
    ax[1].imshow(edges, cmap='gray')
    ax[1].set_title('Canny Edges', fontsize=24)
    ax[2].imshow(emb_img, cmap='gray')
    ax[2].set_title('Bounding Box', fontsize=24)
    plt.show()




edge_and_cut('/kaggle/input/plant-pathology-2020-fgvc7/images/Train_1811.jpg')




edge_and_cut('/kaggle/input/plant-pathology-2020-fgvc7/images/Train_665.jpg')




GCS_DS_PATH = KaggleDatasets().get_gcs_path()

def format_path(st):
    return GCS_DS_PATH + '/images/' + st + '.jpg'

test_paths = df_test.image_id.apply(format_path).values
train_paths = df_train.image_id.apply(format_path).values

train_labels = np.float32(df_train.loc[:, 'healthy':'scab'].values)
train_paths, valid_paths, train_labels, valid_labels = train_test_split(train_paths, train_labels, test_size=0.15, random_state=2020)




def decode_image(filename, label=None, image_size=(224, 224)):
    bits = tf.io.read_file(filename)
    image = tf.image.decode_jpeg(bits, channels=3)
    image = tf.cast(image, tf.float32) / 255.0
    image = tf.image.resize(image, image_size)
    
    if label is None:
        return image
    else:
        return image, label
    
    
def data_augment(image, label=None):
    image = tf.image.random_flip_left_right(image)
    image = tf.image.random_flip_up_down(image)
    
    if label is None:
        return image
    else:
        return image, label




BATCH_SIZE = 32
AUTO = tf.data.experimental.AUTOTUNE

train_dataset = (
    tf.data.Dataset
    .from_tensor_slices((train_paths, train_labels))
    .map(decode_image, num_parallel_calls=AUTO)
    .map(data_augment, num_parallel_calls=AUTO)
    .repeat()
    .shuffle(224)
    .batch(BATCH_SIZE)
    .prefetch(AUTO)
)

valid_dataset = (
    tf.data.Dataset
    .from_tensor_slices((valid_paths, valid_labels))
    .map(decode_image, num_parallel_calls=AUTO)
    .batch(BATCH_SIZE)
    .cache()
    .prefetch(AUTO)
)

test_dataset = (
    tf.data.Dataset
    .from_tensor_slices(test_paths)
    .map(decode_image, num_parallel_calls=AUTO)
    .batch(BATCH_SIZE)
)




def build_lrfn(lr_start=0.00001, lr_max=0.00005, 
               lr_min=0.00001, lr_rampup_epochs=5, 
               lr_sustain_epochs=0, lr_exp_decay=.8):
    lr_max = lr_max * 0.5+lr_max

    def lrfn(epoch):
        if epoch < lr_rampup_epochs:
            lr = (lr_max - lr_start) / lr_rampup_epochs * epoch + lr_start
        elif epoch < lr_rampup_epochs + lr_sustain_epochs:
            lr = lr_max
        else:
            lr = (lr_max - lr_min) *                 lr_exp_decay**(epoch - lr_rampup_epochs                                - lr_sustain_epochs) + lr_min
        return lr
    return lrfn




from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam

from tensorflow.keras.metrics import categorical_accuracy

from tensorflow.keras.callbacks import (EarlyStopping, ReduceLROnPlateau, 
                                        ModelCheckpoint, CSVLogger, LearningRateScheduler)




lrfn = build_lrfn()
STEPS_PER_EPOCH = train_labels.shape[0] // BATCH_SIZE
lr_schedule = tf.keras.callbacks.LearningRateScheduler(lrfn, verbose=1)





model = tf.keras.Sequential([DenseNet121(input_shape=(224, 224, 3),
                                             weights='imagenet',
                                             include_top=False),
                                 L.GlobalAveragePooling2D(),
                                 L.Dense(4,
                                         activation='softmax')])
        
model.compile(optimizer='adam',
                  loss = 'categorical_crossentropy',
                  metrics=['categorical_accuracy'])
model.summary()




filepath = 'model_dnn.h5'
checkpoint = ModelCheckpoint(filepath, monitor='val_accuracy', verbose=1, 
                             save_best_only=True, mode='max')

history = model.fit(train_dataset,
                    epochs=12,
                    callbacks=[lr_schedule,checkpoint],
                    steps_per_epoch=STEPS_PER_EPOCH,
                    validation_data=valid_dataset)




acc = history.history['categorical_accuracy']
val_acc = history.history['val_categorical_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']
epochs = range(1, len(acc) + 1)

plt.plot(epochs, loss, 'bo', label='Training loss')
plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.legend()
plt.figure()


plt.plot(epochs, acc, 'bo', label='Training acc')
plt.plot(epochs, val_acc, 'b', label='Validation acc')
plt.title('Training and validation accuracy')
plt.legend()
plt.figure()

plt.show()




probs_dnn = model.predict(test_dataset, verbose=1)
sub.loc[:, 'healthy':] = probs_dnn
sub.to_csv('submission_dnn.csv', index=False)
sub.head()

