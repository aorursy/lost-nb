#!/usr/bin/env python
# coding: utf-8



# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from subprocess import check_output
print(check_output(["ls", "../input/train"]).decode("utf8"))
import os
import matplotlib.pyplot as plt
import glob
import seaborn as sns
import cv2
import datetime
import pandas as pd
import time
import warnings
warnings.filterwarnings("ignore")
from sklearn.model_selection import train_test_split
from sklearn.metrics import log_loss

# Any results you write to the current directory are saved as output.
from sklearn.cross_validation import KFold
from keras.models import Sequential
from keras.layers import Dropout, Flatten, Convolution2D, MaxPooling2D, ZeroPadding2D, Dense, Activation
from keras.optimizers import RMSprop, Adam
from keras.callbacks import EarlyStopping
from keras.utils import np_utils
from keras import backend as K




folders = check_output(["ls", "../input/train/"]).decode("utf8").strip().split('\n')
#print (folders)
count = {}
for folder in folders:
    images = len(check_output(["ls", "../input/train/" + folder]).decode("utf8").strip().split('\n'))
    print("Number of files for the species", folder, ":", images)
    count[folder] = images
    
plt.figure(figsize=(12,4))
sns.barplot(list(count.keys()), list(count.values()), alpha=0.8)
plt.xlabel('Fish Species', fontsize=12)
plt.ylabel('Number of Images', fontsize=12)
plt.show()    




def get_image(img):
    i = cv2.imread(img)
    new_image = cv2.resize(i, (32, 32), cv2.INTER_LINEAR)
    return new_image

def get_train(folders):
    X_train = []
    X_train_id = []
    y_train = []

    for folder in folders:
        classes = folders.index(folder)  # 0 to 7
        i = os.path.join('..', 'input', 'train', folder, '*.jpg')
        images = glob.glob(i)
        for image in images:
            fld = os.path.basename(image)
            new = get_image(image)
            X_train.append(new)
            X_train_id.append(fld)
            y_train.append(classes)

    return X_train, y_train, X_train_id


def get_test():
    
    X_test = []
    X_test_id = []
    
    i = os.path.join('..', 'input', 'test_stg1', '*.jpg')
    images = sorted(glob.glob(i))

    for image in images:
        fld = os.path.basename(image)
        new = get_image(image)
        X_test.append(new)
        X_test_id.append(fld)

    return X_test, X_test_id




X_train, y_train, X_train_id = get_train(folders)
X_test, X_test_id = get_test()




def normalize_features(X):
    min_value = 0
    max_value = 255
    
    X = np.array(X, dtype=np.uint8)
    X = X.transpose((0, 3, 1, 2))
    X = X.astype('float32')
    X = ((X - min_value)/(max_value - min_value))
    return X


def normalize_targets(y):

    y = np.array(y, dtype=np.uint8)
    y = np_utils.to_categorical(y, 8)
    return y




X_train = normalize_features(X_train)
X_test = normalize_features(X_test)
y_train = normalize_targets(y_train)




print (X_test.shape)
X_train, X_valid, y_train, y_valid = train_test_split(X_train, y_train, 
                                                    test_size=0.2, random_state=23, 
                                                    stratify=y_train)




model = Sequential()
model.add(ZeroPadding2D((1, 1), input_shape=(3, 32, 32), dim_ordering='th'))
model.add(Convolution2D(4, 3, 3, activation='relu', dim_ordering='th'))
model.add(ZeroPadding2D((1, 1), dim_ordering='th'))
model.add(Convolution2D(4, 3, 3, activation='relu', dim_ordering='th'))
model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2), dim_ordering='th'))

model.add(ZeroPadding2D((1, 1), dim_ordering='th'))
model.add(Convolution2D(8, 3, 3, activation='relu', dim_ordering='th'))
model.add(ZeroPadding2D((1, 1), dim_ordering='th'))
model.add(Convolution2D(8, 3, 3, activation='relu', dim_ordering='th'))
model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2), dim_ordering='th'))

model.add(Flatten())
model.add(Dense(32, activation='relu'))
model.add(Dropout(0.6))
model.add(Dense(32, activation='relu'))
model.add(Dropout(0.6))
model.add(Dense(8, activation='softmax'))


adam = Adam(lr=0.003, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
model.compile(loss = 'categorical_crossentropy', optimizer = adam)
    
model.fit(X_train, y_train, batch_size=128, nb_epoch=60,
          validation_split=0.2, verbose=1, shuffle=True)




preds = model.predict(X_valid, verbose=1)
print("Validation Log Loss: {}".format(log_loss(y_valid, preds)))




test_preds = model.predict(X_test, verbose=1)




#create_submission(test_preds, folders)
submission = pd.DataFrame(test_preds, columns = folders)
submission.insert(0, 'image', X_test_id)
submission.head()




final = 'final_submissions.csv'
submission.to_csv(final, index=False)






