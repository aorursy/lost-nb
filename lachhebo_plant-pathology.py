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

SAMPLE_LEN = 20
BASE_DIR_PATH = "/kaggle/input/plant-pathology-2020-fgvc7"
IMAGE_PATH = "/kaggle/input/plant-pathology-2020-fgvc7/images/"
TEST_PATH = "/kaggle/input/plant-pathology-2020-fgvc7/test.csv"
TRAIN_PATH = "/kaggle/input/plant-pathology-2020-fgvc7/train.csv"
SUB_PATH = "/kaggle/input/plant-pathology-2020-fgvc7/sample_submission.csv"

IMAGE_SIZE = 124
BATCH_SIZE = 64




sub = pd.read_csv(SUB_PATH)
test_data = pd.read_csv(TEST_PATH)
train_data = pd.read_csv(TRAIN_PATH)




train_data.shape




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




mean_x, mean_y = 0, 1
for image in train_images:
    mean_x = mean_x + image.shape[0]
    mean_y = mean_y + image.shape[1]
    
print(mean_x/len(train_images), mean_y/len(train_images), mean_x/len(train_images) / 5, mean_y/len(train_images) / 5)




fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(20, 20))
ax[0].imshow(cv2.resize(train_images[5], (409, 273)))
ax[0].set_title('Original Image', fontsize=20)
ax[1].imshow(cv2.resize(train_images[5], (IMAGE_SIZE, IMAGE_SIZE)))
ax[1].set_title('Resized Image', fontsize=20)




fig, ax = plt.subplots(nrows=1, ncols=4, figsize=(30, 10))

ax[0].imshow(cv2.resize(train_images[15][:,:,0], (IMAGE_SIZE, IMAGE_SIZE)), cmap='Reds')
ax[1].imshow(cv2.resize(train_images[15][:,:,1], (IMAGE_SIZE, IMAGE_SIZE)), cmap='Greens')
ax[2].imshow(cv2.resize(train_images[15][:,:,2], (IMAGE_SIZE, IMAGE_SIZE)), cmap='Blues')
ax[3].imshow(cv2.resize(train_images[15], (IMAGE_SIZE, IMAGE_SIZE)))




red_values   = [np.mean(train_images[idx][:, :, 0]) for idx in range(len(train_images))]
green_values = [np.mean(train_images[idx][:, :, 1]) for idx in range(len(train_images))]
blue_values  = [np.mean(train_images[idx][:, :, 2]) for idx in range(len(train_images))]




red_channel   =    [np.mean(train_images[idx][:, :, 0]) for idx in range(len(train_images))] # train_df['lenght_prop'][train_df['sentiment'] == 'positive'].to_numpy()
green_channel =    [np.mean(train_images[idx][:, :, 1]) for idx in range(len(train_images))] # train_df['lenght_prop'][train_df['sentiment'] == 'negative'].to_numpy()
blue_channel  =    [np.mean(train_images[idx][:, :, 2]) for idx in range(len(train_images))]  # train_df['lenght_prop'][train_df['sentiment'] == 'negative'].to_numpy()

BoxName = ['red','green', 'blue']
data = [red_channel, green_channel, blue_channel ]

plt.boxplot(data)
plt.ylim(0,200)
pylab.xticks([1,2,3], BoxName)
plt.show()




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




model = tf.keras.Sequential([DenseNet121(input_shape=(IMAGE_SIZE, IMAGE_SIZE, 3),
                                         weights='imagenet',
                                         include_top=False),
                             L.GlobalAveragePooling2D(),
                             L.Dense(train_labels.shape[1],
                                     activation='softmax')])

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

ax[0].plot(np.linspace(1,EPOCHS,EPOCHS), history.history['val_loss'], color='red')
ax[0].plot(np.linspace(1,EPOCHS,EPOCHS), history.history['loss'], color='blue')

ax[0].set_xlabel('epoch')
ax[0].set_ylabel('loss')
ax[0].set_ylim(0, 1.3)


ax[1].plot(np.linspace(1,EPOCHS,EPOCHS), history.history['val_categorical_accuracy'], color='red')
ax[1].plot(np.linspace(1,EPOCHS,EPOCHS), history.history['categorical_accuracy'], color='blue')

ax[1].set_xlabel('epoch')
ax[1].set_ylabel('val_categorical_accuracy')
ax[1].set_ylim(0.8, 1)




for i in range(1,len(history.history['val_loss'])):
    decreasing_loss     = history.history['loss'][i] < history.history['loss'][i-1]
    decreasing_accuracy = history.history['val_categorical_accuracy'][i] < history.history['val_categorical_accuracy'][i-1]
    
    if decreasing_loss and decreasing_accuracy:
        print(' we overfit at epoch ', i)




model = tf.keras.Sequential([DenseNet121(input_shape=(IMAGE_SIZE, IMAGE_SIZE, 3),
                                         weights='imagenet',
                                         include_top=False),
                             L.GlobalAveragePooling2D(),
                             L.Dense(train_labels.shape[1],
                                     activation='softmax')])

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

ax[0].plot(np.linspace(1,EPOCHS,EPOCHS), history.history['val_loss'], color='red')
ax[0].plot(np.linspace(1,EPOCHS,EPOCHS), history.history['loss'], color='blue')

ax[0].set_xlabel('epoch')
ax[0].set_ylabel('loss')
ax[0].set_ylim(0, 1.3)


ax[1].plot(np.linspace(1,EPOCHS,EPOCHS), history.history['val_categorical_accuracy'], color='red')
ax[1].plot(np.linspace(1,EPOCHS,EPOCHS), history.history['categorical_accuracy'], color='blue')

ax[1].set_xlabel('epoch')
ax[1].set_ylabel('val_categorical_accuracy')
ax[1].set_ylim(0.8, 1)




img_val = cv2.cvtColor(cv2.imread(valid_paths[7]), cv2.COLOR_BGR2RGB)




plt.imshow(cv2.resize(img_val, (409, 273)))




model.layers[0].input




model.layers




layer_outputs = [layer.output for layer in model.layers[0].layers] 




layer_outputs




activation_model = models.Model(inputs=model.layers[0].input, outputs=layer_outputs)




activations = activation_model.predict(np.array([cv2.resize(img_val, (IMAGE_SIZE, IMAGE_SIZE))]))




fig, ax = plt.subplots(nrows=7, ncols=4, figsize=(30, 25))
m = 0
for i in range(0,7):
    for j in range(0,4):
        ax[i][j].matshow(activations[79][0, :, :, m], cmap='viridis')
        m = m+1




Dispaying more advanced steps in the convnet is difficult as the more we go in depth in the convenet, the more the more the features extracted by the layers become abstract. Futhermore, 




val_images = train_data["image_id"][:SAMPLE_LEN].apply(load_image)




probs_dnn = model.predict(test_dataset, verbose=1)
sub.loc[:, 'healthy':] = probs_dnn
sub.to_csv('submission_enn.csv', index=False)
sub.head()

