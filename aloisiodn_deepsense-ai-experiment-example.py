#!/usr/bin/env python
# coding: utf-8



from keras.preprocessing.image import Iterator
from keras.preprocessing.image import ImageDataGenerator
from keras import backend as K
from keras.applications.inception_v3 import InceptionV3
from keras.applications.inception_v3 import preprocess_input
from keras.models import Model
from keras.layers import Dense, GlobalAveragePooling2D, Input
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
 #import bson                       # this is installed with the pymongo package
from PIL import Image
import time
import gc
from keras.optimizers import SGD
from keras.optimizers import RMSprop
import keras
import tensorflow as tf
import io
from sklearn import preprocessing
import struct
import threading
from keras.preprocessing import image
from keras.preprocessing.image import load_img, img_to_array
from deepsense import neptune
from keras.callbacks import Callback, TensorBoard
from subprocess import check_output
print(check_output(["ls", "/public/Cdiscount"]).decode("utf8"))

print(check_output(["ls", "/input"]).decode("utf8"))

print(keras.__version__, tf.__version__)




class NeptuneCallback(Callback):
    def __init__(self, images_per_epoch=-1, phase=1):
        self.epoch_id = 0
        self.batch_id = 0
        self.phase = phase
        self.images_per_epoch = images_per_epoch

    def on_epoch_end(self, epoch, logs={}):
        self.epoch_id += 1

        # logging numeric channels
        #ctx.job.channel_send('Log-loss train ph'+str(self.phase), self.epoch_id, logs['loss'])
        ctx.job.channel_send('Log-loss val ph'+str(self.phase), self.epoch_id, logs['val_loss'])
        #ctx.job.channel_send('Accuracy training'+str(self.phase), self.epoch_id, logs['acc'])
        ctx.job.channel_send('Accuracy val ph'+str(self.phase), self.epoch_id, logs['val_acc'])

        self.batch_id += 1
        ctx.job.channel_send('Log-loss mon ph'+str(self.phase), self.batch_id, 0)
        ctx.job.channel_send('Accuracy mon ph'+str(self.phase), self.batch_id, 0)
        self.batch_id += 1
        ctx.job.channel_send('Log-loss mon ph'+str(self.phase), self.batch_id, 0)
        ctx.job.channel_send('Accuracy mon ph'+str(self.phase), self.batch_id, 0)

    def on_batch_end(self, epoch, logs={}):
        self.batch_id += 1

        # logging numeric channels
        ctx.job.channel_send('Log-loss mon ph'+str(self.phase), self.batch_id, logs['loss'])
        ctx.job.channel_send('Accuracy mon ph'+str(self.phase), self.batch_id, logs['acc'])




ctx = neptune.Context()


##################################
BATCH_SIZE = 128
SHAPE = 180 #80
CLASSES  = 5270 #5270
###################################

################################################
print("create the base pre-trained model")
################################################
input = Input(shape=(SHAPE, SHAPE, 3), name='NEW_image_input_180x180X3') #New input layer, good to the competition shape
base_model = InceptionV3(input_tensor=input, weights='imagenet', include_top=False)


x = base_model.output
#Some aditional layers
x = GlobalAveragePooling2D(name = 'NEW_GlobalAveragePooling2D')(x)
# let's add a fully-connected layer
#x = Dense(1024, activation='relu', name='NEW_Dense_1024')(x)
#x = Dense(2048, activation='relu', name='NEW_Dense_2048')(x)
# and a logistic layer -- let's say we have 200 classes

#modelo novo
predictions = Dense(CLASSES, activation='softmax', name='NEW_Predictions_5270')(x)
model = Model(inputs=base_model.input, outputs=predictions)




################################################
print("\nEncode categories...")
################################################

#Uses LabelEncoder for class_id encoding
categories_path = "/input/category_names.csv"
le = preprocessing.LabelEncoder()
le.fit(pd.read_csv(categories_path).category_id)


################################################
print("\nDefine Generator...")
################################################
from keras.applications.inception_v3 import preprocess_input

#The generator. The flow method does the generator job!
class BinFileIterator(Iterator):
    def __init__(self, bin_file_name, img_generator, samples,
                 target_size=(180,180),
                 batch_size=32, num_class=5270):
        self.file = open(bin_file_name,'rb')
        self.img_gen=img_generator
        self.target_size = tuple(target_size)
        self.image_shape = self.target_size + (3,)
        self.num_class = num_class
        self.lock = threading.Lock() #Since we have 2 files, each generator has its own lock
        super(BinFileIterator, self).__init__(samples, batch_size,
                                              shuffle=False,
                                              seed=None)

    def flow(self, index_array):
        X = np.zeros((len(index_array),) + self.image_shape, dtype=K.floatx())
        Y = np.zeros((len(index_array), self.num_class), dtype=K.floatx())

        for i, j in enumerate(index_array):
            with self.lock:
                buffer=self.file.read(8)
                if len(buffer) < 8:
                    self.file.seek(0)
                    buffer=self.file.read(8)
                encoded_class, length = struct.unpack("<ii", buffer)
                bson_img = self.file.read(length)
            img = load_img(io.BytesIO(bson_img), target_size=self.target_size)
            x = image.img_to_array(img)
            #x = self.img_gen.random_transform(x)
            #x = self.img_gen.standardize(x)
            X[i] = x
            Y[i, encoded_class] = 1

        X = preprocess_input(np.array(X))
        return X, Y

    def next(self):
        with self.lock:
            index_array = next(self.index_generator)
        return self.flow(index_array[0])

n_train_images= 10428
n_val_images= 2362

train_img_gen = ImageDataGenerator()
train_gen = BinFileIterator('/input/train_sample.bin', img_generator=train_img_gen,  samples=n_train_images,
                 target_size=(180,180),
                 batch_size=BATCH_SIZE)

val_img_gen = ImageDataGenerator()
val_gen = BinFileIterator('/input/val_sample.bin', img_generator=val_img_gen,  samples=n_val_images,
                 target_size=(180,180),
                 batch_size=BATCH_SIZE)




################################################
print("fit the new classifier")
################################################

# first: train only the top layers (which were randomly initialized)
# i.e. freeze all convolutional InceptionV3 layers

for layer in base_model.layers[1:]:
    layer.trainable = False

from keras.optimizers import RMSprop
model.compile(optimizer=RMSprop(lr=0.008, decay=0.01), loss='categorical_crossentropy', metrics=['accuracy'])

model.fit_generator(train_gen,
                    steps_per_epoch=n_train_images//BATCH_SIZE,
                    validation_data=val_gen,
                    validation_steps=n_val_images//BATCH_SIZE,
                    #shuffle=True,
                    epochs=5, callbacks=[NeptuneCallback(images_per_epoch=n_train_images//BATCH_SIZE, phase=1)])




Fine Tune some inception modules:




################################################
print("fine tune the model")
################################################


for layer in model.layers[:249]:
   layer.trainable = False
for layer in model.layers[249:]:
   layer.trainable = True

model.compile(optimizer=SGD(lr=0.008, momentum=0.1, decay=0.01), loss='categorical_crossentropy', metrics=['accuracy'])

model.fit_generator(train_gen,
                    steps_per_epoch=n_train_images//BATCH_SIZE,
                    validation_data=val_gen,
                    validation_steps=n_val_images//BATCH_SIZE,
                    #shuffle=True,
                    epochs=5, callbacks=[NeptuneCallback(images_per_epoch=n_train_images//BATCH_SIZE, phase=2)])

