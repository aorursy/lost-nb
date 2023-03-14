#!/usr/bin/env python
# coding: utf-8



# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        #print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session




import pandas as pd

train_df = pd.read_csv('/kaggle/input/mlp-en-cifar-100/cifar100_train.csv')




train_df.head()




base_dir = '/kaggle/input/mlp-en-cifar-100/'




import matplotlib.pyplot as plt
from skimage.io import imread
get_ipython().run_line_magic('matplotlib', 'inline')
img = imread(base_dir+train_df['Image'][0])
plt.imshow(img)
print(img.shape)




x_train = np.zeros((50000,32,32,3),dtype='uint8')
for i,imgname in enumerate(train_df['Image'].values):
    x_train[i,] = np.array(imread(base_dir+imgname))
    
    




y_train = train_df['Label'].values




plt.imshow(x_train[10,])
x_train[10,].shape





def plot_images(x_train,y_train,N,n_rows=3,n_cols=12):
    fig = plt.figure(figsize=(20,5))
    for i in range(N):
        ax = fig.add_subplot(n_rows, n_cols, i + 1, xticks=[], yticks=[])
        ax.set_xlabel(y_train[i])
        plt.imshow(x_train[i,])
plot_images(x_train,y_train,36)




from sklearn.preprocessing import OneHotEncoder
enc = OneHotEncoder(handle_unknown='ignore')
y_categorical = enc.fit_transform(y_train.reshape(-1, 1))




print(y_categorical[0,])




from keras.models import Sequential
from keras.layers import Dense, Flatten

# define the model
model = Sequential()
model.add(Flatten(input_shape = x_train.shape[1:])) #input: 32x32x3 #output: 3072x1
model.add(Dense(1000, activation='relu')) #input:3072x1    #ouput: 1000x1
model.add(Dense(512, activation='relu')) #input:1000x1 #output: 512x1
model.add(Dense(100, activation='softmax'))#input: 512 #output:100 

model.summary()




32*32*3





model.compile(loss='categorical_crossentropy', optimizer='rmsprop', 
                  metrics=['accuracy'])




hist = model.fit(x_train, y_categorical, batch_size=32, epochs=20,
           
          verbose=2, shuffle=True)




test_df = pd.read_csv(base_dir+'cifar100_test.csv')
# Treat the test data in the same way as training data. In this case, pull same columns.
x_test = np.zeros((10000,32,32,3),dtype='uint8')
for i,imgname in enumerate(test_df['Image'].values):
    img = imread(base_dir+imgname)
    x_test[i,:,:,:] = img
    




# Use the model to make predictions
predicted_classes = np.argmax(model.predict(x_test),axis=1)
print(predicted_classes.shape)




predicted_classes




predicted_classes[0]





y_labels = [y.split('x0_')[-1] for y in enc.get_feature_names()]
y_labels




y_labels[predicted_classes[100]]




my_submission = pd.DataFrame({'Image': test_df['Image'], 'Label': y_predictions})
# you could use any filename. We choose submission here
my_submission.to_csv('submission.csv', index=False)











