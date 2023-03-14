#!/usr/bin/env python
# coding: utf-8



import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import os
import matplotlib.pyplot as plt
import pandas as pd
import cv2
import pylab

from sklearn.model_selection import train_test_split
from tensorflow.keras import models
import tensorflow as tf
import tensorflow.keras.layers as L
from tensorflow.keras.applications import DenseNet121




## Global variables

SAMPLE_LEN = 100
BASE_DIR_PATH = "/kaggle/input/plant-pathology-2020-fgvc7"
IMAGE_PATH = "/kaggle/input/plant-pathology-2020-fgvc7/images/"
TEST_PATH = "/kaggle/input/plant-pathology-2020-fgvc7/test.csv"
TRAIN_PATH = "/kaggle/input/plant-pathology-2020-fgvc7/train.csv"
SUB_PATH = "/kaggle/input/plant-pathology-2020-fgvc7/sample_submission.csv"

IMAGE_SIZE = 150
BATCH_SIZE = 64




sub = pd.read_csv(SUB_PATH)
test_data = pd.read_csv(TEST_PATH)
train_data = pd.read_csv(TRAIN_PATH)




train_data.head()




def format_path(st):
    return BASE_DIR_PATH + '/images/' + st + '.jpg'

def load_image(image_id):
    file_path = image_id + ".jpg"
    image = cv2.imread(IMAGE_PATH + file_path)
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
def decode_image(filename, label=None, image_size=(IMAGE_SIZE, IMAGE_SIZE)):
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




test_paths = test_data.image_id.apply(format_path).values
train_paths = train_data.image_id.apply(format_path).values
train_labels = np.float32(train_data.loc[:, 'healthy':'scab'].values)
train_paths, valid_paths, train_labels, valid_labels = train_test_split(train_paths, train_labels, test_size=0.15, random_state=2020)




train_images = train_data["image_id"][:SAMPLE_LEN].apply(load_image)




train_images.shape




fig, ax = plt.subplots(nrows=1, ncols=4, figsize=(30, 10))

ax[0].imshow(cv2.resize(train_images[15][:,:,0], (IMAGE_SIZE, IMAGE_SIZE)), cmap='Reds')
ax[1].imshow(cv2.resize(train_images[15][:,:,1], (IMAGE_SIZE, IMAGE_SIZE)), cmap='Greens')
ax[2].imshow(cv2.resize(train_images[15][:,:,2], (IMAGE_SIZE, IMAGE_SIZE)), cmap='Blues')
ax[3].imshow(cv2.resize(train_images[15], (IMAGE_SIZE, IMAGE_SIZE)))




red_values   = [np.mean(train_images[idx][:, :, 0]) for idx in range(len(train_images))]
green_values = [np.mean(train_images[idx][:, :, 1]) for idx in range(len(train_images))]
blue_values  = [np.mean(train_images[idx][:, :, 2]) for idx in range(len(train_images))]




red_channel   =    [np.mean(train_images[idx][:, :, 0]) for idx in range(len(train_images))] 
green_channel =    [np.mean(train_images[idx][:, :, 1]) for idx in range(len(train_images))] 
blue_channel  =    [np.mean(train_images[idx][:, :, 2]) for idx in range(len(train_images))]

BoxName = ['red','green', 'blue']
data = [red_channel, green_channel, blue_channel ]

plt.boxplot(data)
plt.ylim(0,200)
pylab.xticks([1,2,3], BoxName)
plt.show()




healthy_leaves = train_data["image_id"][train_data['healthy'] == 1][:50].apply(load_image)
fig, ax = plt.subplots(nrows=2, ncols=4, figsize=(30, 10))
image_index = 0
for i in range(0,2):
    for j in range(0,4):
        ax[i][j].imshow(cv2.resize(healthy_leaves[healthy_leaves.index[image_index]], (IMAGE_SIZE, IMAGE_SIZE)))
        image_index = image_index + 1




sick_leafs = train_data["image_id"][train_data['healthy'] == 0][:50].apply(load_image)
fig, ax = plt.subplots(nrows=2, ncols=4, figsize=(30, 10))
image_index = 0
for i in range(0,2):
    for j in range(0,4):
        ax[i][j].imshow(cv2.resize(sick_leafs[sick_leafs.index[image_index]], (IMAGE_SIZE, IMAGE_SIZE)))
        image_index = image_index + 1




rusty_leaves = train_data["image_id"][train_data['rust'] == 1][:50].apply(load_image)
fig, ax = plt.subplots(nrows=2, ncols=4, figsize=(30, 10))
image_index = 0
for i in range(0,2):
    for j in range(0,4):
        ax[i][j].imshow(cv2.resize(rusty_leaves[rusty_leaves.index[image_index]], (IMAGE_SIZE, IMAGE_SIZE)))
        image_index = image_index + 1




scaby_leaves = train_data["image_id"][train_data['scab'] == 1][:50].apply(load_image)
fig, ax = plt.subplots(nrows=2, ncols=4, figsize=(30, 10))
image_index = 0
for i in range(0,2):
    for j in range(0,4):
        ax[i][j].imshow(cv2.resize(scaby_leaves[scaby_leaves.index[image_index]], (IMAGE_SIZE, IMAGE_SIZE)))
        image_index = image_index + 1





fig, ax = plt.subplots(nrows=1, ncols=4, figsize=(25, 10))

red_channel   =    [np.mean(healthy_leaves[idx][:, :, 0]) for idx in healthy_leaves.index] 
green_channel =    [np.mean(healthy_leaves[idx][:, :, 1]) for idx in healthy_leaves.index] 
blue_channel  =    [np.mean(healthy_leaves[idx][:, :, 2]) for idx in healthy_leaves.index]

BoxName = ['red','green', 'blue']
data = [red_channel, green_channel, blue_channel]
medianprops = dict(linestyle='-', linewidth=2.5, color='green')
ax[0].boxplot(data,labels = BoxName, medianprops=medianprops)


red_channel   =    [np.mean(sick_leafs[idx][:, :, 0]) for idx in sick_leafs.index] 
green_channel =    [np.mean(sick_leafs[idx][:, :, 1]) for idx in sick_leafs.index] 
blue_channel  =    [np.mean(sick_leafs[idx][:, :, 2]) for idx in sick_leafs.index]

BoxName = ['red','green', 'blue']
data = [red_channel, green_channel, blue_channel]
medianprops = dict(linestyle='-', linewidth=2.5, color='red')
ax[1].boxplot(data, labels = BoxName, medianprops=medianprops)
ax[0].set_title('healthy leaves')

red_channel   =    [np.mean(rusty_leaves[idx][:, :, 0]) for idx in rusty_leaves.index] 
green_channel =    [np.mean(rusty_leaves[idx][:, :, 1]) for idx in rusty_leaves.index] 
blue_channel  =    [np.mean(rusty_leaves[idx][:, :, 2]) for idx in rusty_leaves.index]

BoxName = ['red','green', 'blue']
data = [red_channel, green_channel, blue_channel]
medianprops = dict(linestyle='-', linewidth=2.5, color='orange')
ax[2].boxplot(data, labels = BoxName, medianprops=medianprops)
ax[2].set_title('rusty leaves')

red_channel   =    [np.mean(scaby_leaves[idx][:, :, 0]) for idx in scaby_leaves.index] 
green_channel =    [np.mean(scaby_leaves[idx][:, :, 1]) for idx in scaby_leaves.index] 
blue_channel  =    [np.mean(scaby_leaves[idx][:, :, 2]) for idx in scaby_leaves.index]

BoxName = ['red','green', 'blue']
data = [red_channel, green_channel, blue_channel]
medianprops = dict(linestyle='-', linewidth=2.5, color='grey')
ax[3].boxplot(data, labels = BoxName, medianprops=medianprops)
ax[3].set_title('scaby leaves')

plt.ylim(0,200)
plt.show()




train_data.healthy.value_counts()




fig = plt.figure()

# Divide the figure into a 1x4 grid, and give me the first section
ax1 = fig.add_subplot(141)

# Divide the figure into a 2x1 grid, and give me the second section
ax2 = fig.add_subplot(142)

# Divide the figure into a 1x4 grid, and give me the first section
ax3 = fig.add_subplot(143)

# Divide the figure into a 2x1 grid, and give me the second section
ax4 = fig.add_subplot(144)

size_figure = 30

series = train_data.healthy.value_counts()
series.index = ['sick', 'healthy']
series.plot.pie(figsize=(size_figure, size_figure), ax=ax1, colors= ['red', 'green'])

series = train_data.scab.value_counts()
series.index = ['not_scab', 'scab']
series.plot.pie(figsize=(size_figure, size_figure), ax=ax2, colors= ['blue', 'red'], startangle = 100)

series = train_data.rust.value_counts()
series.index = ['not_rust', 'rust']
series.plot.pie(figsize=(size_figure, size_figure), ax=ax3, colors= ['blue', 'red'], startangle = 250)

series = train_data.multiple_diseases.value_counts()
series.index = ['no_multiple_d', 'multiple_d']
series.plot.pie(figsize=(size_figure, size_figure), ax=ax4, colors= ['blue', 'red'], startangle = 280)




STEPS_PER_EPOCH = train_labels.shape[0] // BATCH_SIZE 
STEPS_PER_EPOCH




train_dataset = (
    tf.data.Dataset
    .from_tensor_slices((train_paths, train_labels))
    .map(decode_image)
    .map(data_augment)
    .repeat()
    .shuffle(1)
    .batch(BATCH_SIZE)
    .prefetch(1)
)

valid_dataset = (
    tf.data.Dataset
    .from_tensor_slices((valid_paths, valid_labels))
    .map(decode_image)
    .batch(BATCH_SIZE)
    .prefetch(1)
)

test_dataset = (
    tf.data.Dataset
    .from_tensor_slices(test_paths)
    .map(decode_image)
    .batch(BATCH_SIZE)
)




base_model = tf.keras.applications.DenseNet121(input_shape=(IMAGE_SIZE, IMAGE_SIZE, 3), include_top=False, weights='imagenet', input_tensor=None, pooling=None, classes=4)




base_model.compile(optimizer='adam',
              loss = 'categorical_crossentropy',
              metrics=['categorical_accuracy'])
base_model.summary()




base_model.trainable = False




base_model.summary()





global_average_layer = tf.keras.layers.GlobalAveragePooling2D()
prediction_layer = tf.keras.layers.Dense(4)


model = tf.keras.Sequential([
  base_model,
  global_average_layer,
  prediction_layer
])


model.compile(optimizer='adam',
              loss = 'categorical_crossentropy',
              metrics=['categorical_accuracy'])
model.summary()




EPOCHS = 20




history = model.fit(train_dataset,
                    epochs=EPOCHS,
                    steps_per_epoch=STEPS_PER_EPOCH,
                    validation_data=valid_dataset)




fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(20, 5))

ax[0].plot(np.linspace(1,EPOCHS,EPOCHS), history.history['val_loss'], color='red', label='val')
ax[0].plot(np.linspace(1,EPOCHS,EPOCHS), history.history['loss'], color='blue',label='train')

ax[0].set_xlabel('epoch')
ax[0].set_ylabel('loss')
ax[0].set_ylim(3, 12)
ax[0].legend()

ax[1].plot(np.linspace(1,EPOCHS,EPOCHS), history.history['val_categorical_accuracy'], color='red', label='val')
ax[1].plot(np.linspace(1,EPOCHS,EPOCHS), history.history['categorical_accuracy'], color='blue',label='train')

ax[1].set_xlabel('epoch')
ax[1].set_ylabel('val_categorical_accuracy')
ax[1].set_ylim(0.0, 1)
ax[1].legend()




get_ipython().system('mkdir -p saved_model')




model.save('saved_model/my_model') 




new_model = tf.keras.models.load_model('saved_model/my_model')




from IPython.display import FileLink
    FileLink(r'df_name.csv')

