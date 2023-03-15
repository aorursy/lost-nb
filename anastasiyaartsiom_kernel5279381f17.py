#!/usr/bin/env python
# coding: utf-8



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




import tensorflow
import tensorflow.keras as keras




test_df = pd.read_csv('/kaggle/input/Kannada-MNIST/test.csv')
train_df = pd.read_csv('/kaggle/input/Kannada-MNIST/train.csv')




# train_df




y = train_df['label'].values
# y.shape




X = train_df.loc[:, train_df.columns[1:]].values
# X.shape




# from sklearn.preprocessing import MinMaxScaler




# scaler = MinMaxScaler(feature_range=(-1, 1))
# X_scaled = scaler.fit_transform(X)




from skimage.io import imshow, imshow_collection




imshow_collection(X[:12].reshape(12, 28, 28))




from sklearn.model_selection import train_test_split




# X_train, X_valid, y_train, y_valid = X, test_df.loc[:, test_df.columns[1:]].values, y, test_df.loc[:, test_df.columns[:1]].values




X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.1, random_state=100500)




X_train = X_train.reshape(X_train.shape[0],28,28,1)
X_valid = X_valid.reshape(X_valid.shape[0],28,28,1)




# train_datagen = ImageDataGenerator(rescale=1./255.,
#                                    rotation_range=10,
#                                    width_shift_range=0.2,
#                                    height_shift_range=0.2,
#                                    shear_range=0.1,
#                                    zoom_range=0.2,
#                                    horizontal_flip=False)

# valid_datagen = ImageDataGenerator(rescale=1./255.)




from keras.layers import *
from keras.models import Model
from keras.callbacks import CSVLogger, ModelCheckpoint
from keras.models import Sequential
from keras.layers import Conv2D, Dropout, Dense, Flatten, BatchNormalization, MaxPooling2D






model = Sequential()

model.add(Conv2D(64, kernel_size=3, activation='relu', input_shape=(28, 28, 1),padding='same'))
model.add(BatchNormalization())
model.add(Conv2D(64, kernel_size=3, activation='relu',padding='same'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.3))

model.add(Conv2D(128, kernel_size=3, activation='relu',padding='same'))
model.add(BatchNormalization())
model.add(Conv2D(128, kernel_size=3, activation='relu',padding='same'))
model.add(BatchNormalization())
model.add(Conv2D(128, kernel_size=5, activation='relu', padding='same'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.3))

model.add(Conv2D(256, kernel_size=3, activation='relu',padding='same'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.3))

model.add(Flatten())
model.add(Dense(256))
model.add(BatchNormalization())
model.add(Dense(128))
model.add(BatchNormalization())
model.add(Dropout(0.3))

model.add(Dense(10, activation='softmax'))

model.compile(optimizer ='sgd',
              loss = 'sparse_categorical_crossentropy',
              metrics=['accuracy'])

model.summary()




batch_size = 256






model.fit(X_train, y_train,
          epochs=25,
          verbose=1,
          batch_size=batch_size,
          validation_data=(X_valid, y_valid),
          callbacks=[
              ModelCheckpoint('/kaggle/working/best_kannada_model.h5', save_best_only=True),
              CSVLogger('/kaggle/working/learning_log.csv'),
          ])




model = keras.models.load_model('model.h5')


X_test = X_test.reshape(X_test.shape[0],28,28,1)









pred_probas = model.predict(X_test, batch_size=batch_size)




result = pd.read_csv('/kaggle/input/Kannada-MNIST/sample_submission.csv')
result['label'] = pred_probas.argmax(axis=1)

result




result.to_csv('submission.csv', index=False)

