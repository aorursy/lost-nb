#!/usr/bin/env python
# coding: utf-8



###Importing Important Libraries

import numpy as np
import pandas as pd
import keras
import matplotlib.pyplot as plt
import os
import random

from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential,Model
from keras.layers import Conv2D, MaxPooling2D, Dropout, Flatten, Dense, Activation, BatchNormalization
from keras.callbacks import EarlyStopping, ReduceLROnPlateau
from sklearn.model_selection import train_test_split




###Unzipping the Data
import zipfile

##Train
local_zip_train = '/kaggle/input/aerial-cactus-identification/train.zip'
zip_ref = zipfile.ZipFile(local_zip_train, 'r')
zip_ref.extractall('/kaggle/working')
zip_ref.close()

##Test
local_zip_test = '/kaggle/input/aerial-cactus-identification/test.zip'
zip_ref = zipfile.ZipFile(local_zip_test, 'r')
zip_ref.extractall('/kaggle/working')
zip_ref.close()




###Image Hyperparameters

IMAGE_SIZE = 150
IMAGE_WIDTH, IMAGE_HEIGHT = IMAGE_SIZE,IMAGE_SIZE
IMAGE_SIZE=(IMAGE_WIDTH, IMAGE_HEIGHT)
IMAGE_CHANNELS=3




cd '/kaggle/input/aerial-cactus-identification'




df = pd.read_csv('train.csv')
df.head()




df["has_cactus"] = df["has_cactus"].replace({0: 'No', 1: 'Yes'}) 




df.head()




###Making the Model
from keras.applications import VGG16

input_shape = (IMAGE_HEIGHT,IMAGE_WIDTH,IMAGE_CHANNELS)

model = Sequential

pre_trained_model = VGG16(input_shape=input_shape, include_top=False, weights="imagenet")

for layer in pre_trained_model.layers[:15]:
    layer.trainable = False
for layer in pre_trained_model.layers[15:]:
    layer.trainable = True

last_layer = pre_trained_model.get_layer('block5_pool')
last_output = last_layer.output

# Flatten the output layer to 1 dimension
x = Flatten()(last_output)
# Add a fully connected layer with 512 hidden units and ReLU activation
x = Dense(512, activation='relu')(x)
# Add a dropout rate of 0.5
x = Dropout(0.5)(x)
# Add a final sigmoid layer for classification
x = Dense(1, activation='sigmoid')(x)

model = Model(pre_trained_model.input, x)




###Compiling the Model

model.compile(optimizer = 'adam',loss='binary_crossentropy',metrics=['accuracy'])




###Model Summary

model.summary()




###Callbacks

learning_rate_reduction = ReduceLROnPlateau(monitor='val_accuracy', 
                                            patience=2, 
                                            verbose=1, 
                                            factor=0.5, 
                                            min_lr=0.00001)

earlystop = EarlyStopping(patience=10)

callbacks = [earlystop, learning_rate_reduction]




###Preparing the Data

train_df, validate_df = train_test_split(df, test_size=0.10, random_state=42)
train_df = train_df.reset_index(drop=True)
validate_df = validate_df.reset_index(drop=True)




###Hyperparameters


total_train = train_df.shape[0]
total_validate = validate_df.shape[0]
batch_size=64
epochs = 15




###Generator

##Train
train_datagen = ImageDataGenerator(
    rotation_range=15,
    rescale=1./255,
    shear_range=0.1,
    zoom_range=0.2,
    horizontal_flip=True,
    width_shift_range=0.1,
    height_shift_range=0.1
)

train_generator = train_datagen.flow_from_dataframe(
    train_df, 
    "/kaggle/working/train", 
    x_col='id',
    y_col='has_cactus',
    target_size=IMAGE_SIZE,
    class_mode='binary',
    batch_size=batch_size
)





##Val 
validation_datagen = ImageDataGenerator(rescale=1./255)
validation_generator = validation_datagen.flow_from_dataframe(
    validate_df, 
    "/kaggle/working/train", 
    x_col='id',
    y_col='has_cactus',
    target_size=IMAGE_SIZE,
    class_mode='binary',
    batch_size=batch_size
)




###Training the Model

history = model.fit_generator(
    train_generator, 
    epochs=5,
    validation_data=validation_generator,
    validation_steps=total_validate//batch_size,
    steps_per_epoch=total_train//batch_size,
    callbacks=callbacks
)




###Testing

##Creating a test DataFrame
filenames_test = os.listdir("/kaggle/working/test")
print(len(filenames_test))

df_test = pd.DataFrame({
    'id': filenames_test,
})




###Creating the Test Image Generator

test_datagen = ImageDataGenerator(rescale=1./255)

test_generator=test_datagen.flow_from_dataframe(
dataframe=df_test,
directory="/kaggle/working/test",
x_col="id",
y_col=None,
batch_size=32,
shuffle=False,
class_mode=None,
target_size=(IMAGE_HEIGHT,IMAGE_WIDTH))




###Predicting
STEP_SIZE_TEST=test_generator.n//test_generator.batch_size

pred=model.predict_generator(test_generator,
steps=STEP_SIZE_TEST,
verbose=1)




pred = np.where(pred > 0.5, 1, 0)




cd '/kaggle/working'




prediction = df_test
prediction['label'] = pred
prediction.to_csv('Prediction.csv') 

