#!/usr/bin/env python
# coding: utf-8



from shutil import unpack_archive as ua
ua("/kaggle/input/dogs-vs-cats/test1.zip", "/kaggle/output/working")
ua("/kaggle/input/dogs-vs-cats/train.zip", "/kaggle/output/working")




mkdir train




cd /kaggle/working/train




mkdir cat




mkdir dog




from shutil import copy as cp
import os
dest = "/kaggle/working/train"
for name in os.listdir("/kaggle/output/working/train"):
    d = os.path.join(dest, name.split(".")[0])
    src = os.path.join("/kaggle/output/working/train",name)
    cp(src, d)
    #print(src, d)




# Dimesion of images to be use. Can choose anything as a different dimenion will be resized to this.
ht = 180
wth = 180

# Important hyperparameters
batch_size = 32
num_classes = 2
no_of_epochs = 3
validation_split_coeff = 0.2

# The directory for training dataset to be specified in data pipeline
data_dir = "/kaggle/working/train"




import tensorflow as tf

train_ds = tf.keras.preprocessing.image_dataset_from_directory(
  data_dir,
  validation_split=validation_split_coeff,
  subset="training",
  seed=123,
  image_size=(ht, wth),
  batch_size=batch_size)




val_ds = tf.keras.preprocessing.image_dataset_from_directory(
  data_dir,
  validation_split=validation_split_coeff,
  subset="validation",
  seed=123,
  image_size=(ht, wth),
  batch_size=batch_size)




from tensorflow.keras import layers

model = tf.keras.Sequential([
  layers.experimental.preprocessing.Rescaling(1./255),
  layers.Conv2D(32, 3, activation='relu'),
  layers.MaxPooling2D(),
  layers.Conv2D(32, 3, activation='relu'),
  layers.MaxPooling2D(),
  layers.Conv2D(32, 3, activation='relu'),
  layers.MaxPooling2D(),
  layers.Flatten(),
  layers.Dense(128, activation='relu'),
  layers.Dense(num_classes)
])

model.compile(
  optimizer='adam',
  loss=tf.losses.SparseCategoricalCrossentropy(from_logits=True),
  metrics=['accuracy'])

model.fit(
  train_ds,
  validation_data=val_ds,
  epochs=no_of_epochs
)

