#!/usr/bin/env python
# coding: utf-8



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




import numpy as np
import pandas as pd
from keras.regularizers import l2
from sklearn.model_selection import train_test_split
from keras.applications.vgg16 import VGG16
from keras.models import Sequential
from keras.layers import Activation, Dropout, Flatten, Dense
from keras.optimizers import Adam
import cv2
import matplotlib.pyplot as plt  
from tqdm import tqdm_notebook
import os




base_path = "../input/"




data_df = pd.read_csv(base_path+'train.csv')
data_df.head()




sample_image = cv2.imread("../input/train/train/008bd3d84a1145e154409c124de7cee9.jpg")




plt.imshow(sample_image)




X=[]
y=[]
for index, ex in tqdm_notebook(data_df.iterrows()):
    name =  cv2.imread(base_path+'train/train/'+ex['id'])
    name = name/255 # Normalize b/w [0,1]
    label =  ex['has_cactus']
    X.append(name)
    y.append(label)
X = np.array(X)
y = np.array(y).reshape(-1,1)
X.shape, y.shape




epochs = 1000
batch_size = 64




model_vgg16 = VGG16(include_top = False,
                   #weights = 'imagenet',
                   input_shape = (32,32,3,))




#model_vgg16.trainable = False
model_vgg16.summary()




model = Sequential()
model.add(model_vgg16)
model.add(Flatten())
model.add(Dense(256, activation='relu', activity_regularizer=l2(0.001)))
model.add(Dense(128, activation='relu', activity_regularizer=l2(0.001)))
model.add(Dropout(0.5))
model.add(Dense(1))
model.add(Activation('sigmoid'))




model.compile(loss='binary_crossentropy',
              optimizer=Adam(lr=0.00001),
              metrics=['accuracy'])




history = model.fit(X, y,
                   validation_split=0.2,
                   epochs=epochs,
                   shuffle= True,
                   batch_size=batch_size,
                   verbose=2)




model.save('model_cactus.hdf5')




plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title("Model Loss")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.legend(["train", "validation"])




plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title("Model Accuracy")
plt.xlabel("Epochs")
plt.ylabel("Accuracy")
plt.legend(["train", "validation"])




result_df = pd.DataFrame(columns=['id', 'has_cactus'])




result = {}
for filepath in tqdm_notebook(os.listdir(base_path+"test/test/")):
    image = cv2.imread(base_path+"test/test/"+filepath)
    image = image/255
    result[filepath] = 1 if model.predict(image.reshape(-1, 32, 32, 3)) > 0.7 else 0




for i, val in enumerate(result.items()):
    result_df.loc[i] = val




result_df.to_csv("submission.csv", index = False)

