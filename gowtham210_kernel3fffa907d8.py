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




import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from numpy import argmax
import cv2
import glob
import pydicom as dicom
import random as ran
from scipy.ndimage.interpolation import rotate

from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.metrics import classification_report
from sklearn.utils import shuffle
from skmultilearn.problem_transform import BinaryRelevance

import keras
from keras.models import Sequential
from keras.layers import Dense, Flatten, Dropout
from keras.regularizers import l2, l1
from keras.optimizers import SGD




import pandas as pd
train_csv = pd.read_csv('../input/rsna-intracranial-hemorrhage-detection/rsna-intracranial-hemorrhage-detection/stage_2_train.csv')

print('Length of train_csv:',str(len(train_csv)))
print('Expected length of post-transform train_labels:', str(int(len(train_csv)/6)))

train_csv = train_csv[~train_csv.ID.str.contains('any')]
train_csv['hem'] = train_csv['ID'].str.split('_').str[2]

train_labels_pos = train_csv.loc[train_csv['Label']==1].groupby([train_csv['ID'].str.split('_').str[1]
                                                                ])['hem'].apply(lambda x: "%s" % '_'.join(x))
train_labels_neg = train_csv.groupby([train_csv["ID"].str.split("_").str[1]]).sum()
train_labels_neg[train_labels_neg == 0] = 'none'
train_labels_neg = train_labels_neg[train_labels_neg == 'none']
train_labels_neg.dropna(inplace=True)

train_labels = train_labels_pos.append(train_labels_neg['Label'])
print(train_labels.head())

train_labels.index = 'ID_'+train_labels.index
train_labels = train_labels.str.split('_')

print(train_labels.head())
print('Training labels created.\nLength of train_labels: '+str(len(train_labels)))






def import_images(total_images, hem_rate):
    image_arrays = []
    labels = []
    image_counter = 0
    total_images = total_images
    hem_img_count = 0
    total_hem_img_count = int(total_images*hem_rate)

    files = glob.glob("../input/rsna-intracranial-hemorrhage-detection/rsna-intracranial-hemorrhage-detection/stage_2_train/*.dcm")
    #files = glob.glob("./data/stage_1_train_images/*.dcm")
    ran.shuffle(files)
    ran.shuffle(files)
    ran.shuffle(files)

    for im_file in files:
        if image_counter < total_images:
            try:
                image = dicom.dcmread(im_file)
                image_array = image.pixel_array
                image_array_resized = cv2.resize(image_array,(50,50))
                image_label = train_labels.loc[train_labels.index==str(image.SOPInstanceUID)][0]

                if 'none' not in image_label and hem_img_count < total_hem_img_count:
                    if image_counter % 1000 == 0 and image_counter != 0:
                        print(str(image_counter), 'images imported.')
                    image_arrays.append(image_array_resized)
                    labels.append(image_label)
                    hem_img_count += 1
                    image_counter += 1
                elif hem_img_count >= total_hem_img_count:
                    if image_counter % 1000 == 0 and image_counter != 0:
                        print(str(image_counter), 'images imported.')
                    image_arrays.append(image_array_resized)
                    labels.append(image_label)
                    image_counter += 1    
            except:
                pass
        else:
            break

    print('Import complete.'+'\n\n'+'images: ' + str(image_counter)+'\n')

    image_arrays, labels = shuffle(np.asarray(image_arrays), np.asarray(labels), random_state=0)
    data = {'images': image_arrays, 'labels': labels}
    
    print(data['images'][0],'\n\n',data['labels'][0])
    
    return data
data = import_images(10000, .75)




training_data_array = data['images']
training_labels_array = data['labels']
np.save('training_data_array.npy', training_data_array)
np.save('training_labels_array.npy', training_labels_array)

test_data = import_images(25000, .75)

test_data_array = test_data['images']
test_labels_array = test_data['labels']
np.save('test_data_array.npy', test_data_array)
np.save('test_labels_array.npy', test_labels_array)




training_data_array = np.load('training_data_array.npy',allow_pickle=True)
training_labels_array = np.load('training_labels_array.npy',allow_pickle=True)

tdata = {'images':training_data_array,'labels':training_labels_array}




#-------------------------------------
#------------
#-------------------------------------
#------------
#-------------------------------------
#------------
#-------------------------------------
#------------




training_data_array1 = np.load('../input/sam-data/training_data_array.npy',allow_pickle=True)
training_labels_array1 = np.load('../input/sam-data/training_labels_array.npy',allow_pickle=True)
training_data_array2 = np.load('../input/sam-data/training_data_array_2.npy',allow_pickle=True)
training_labels_array2 = np.load('../input/sam-data/training_labels_array_2.npy',allow_pickle=True)

test_data_array = np.load('../input/sam-data/test_data_array.npy',allow_pickle=True)
test_labels_array = np.load('../input/sam-data/test_labels_array.npy',allow_pickle=True)
print(training_data_array1.shape)




training_data_array= np.append(training_data_array1,training_data_array2)
training_labels_array = np.append(training_labels_array1,training_labels_array2)
training_data_array = np.reshape(training_data_array,(20000,50,50))

data={'images':training_data_array,'labels':training_labels_array}
tdata = {'images':training_data_array,'labels':training_labels_array}
print(training_data_array.shape)

test_data = {'images':test_data_array,'labels':test_labels_array}

#print(training_data_array[0])
#print(test_data_array.shape)
#print(training_labels_array.shape)




hemorrhages = {'epidural':0,'intraparenchymal':0,'intraventricular':0,'subarachnoid':0,'subdural':0                ,'any':0,'none':0}
unique_labels, label_counts = np.unique(data['labels'], return_counts=True)

index = 0
for i in unique_labels:
    print(i,'-- Occurences:',label_counts[index])
    for k,v in hemorrhages.items():
        if k in i:
            hemorrhages[k] += int(label_counts[index])
    index += 1

ax = sns.barplot(x=list(hemorrhages.keys()), y=list(hemorrhages.values()))
ax.title.set_text('Label Occurences')
ax.set_xticklabels(labels = list(hemorrhages.keys()),rotation=30)
plt.show()




labels = {'subdural':[],'epidural':[],'intraparenchymal':[],'intraventricular':[],'subarachnoid':[],'none':[]}

for i in data['labels']:
    for k,v in labels.items():
        if k in i:
            v.append(True)
        else:
            v.append(False)

label_ovlp_df = pd.DataFrame(labels)
print(label_ovlp_df.head())

label_ovlp_rt_df = pd.DataFrame(index=list(labels.keys()), columns=list(labels.keys()))
label_ovlp_rt_df = label_ovlp_rt_df.astype(float)

for i in list(labels.keys()):
    for j in list(labels.keys()):
        label_ovlp_rt_df.loc[i,j] = len(label_ovlp_df[label_ovlp_df[[i,j]].eq(True).all(axis=1)])/             len(label_ovlp_df[label_ovlp_df[i].eq(True)])

ax = sns.heatmap(label_ovlp_rt_df, vmin=0, vmax=1, annot=True, fmt='.2f', linewidth=.5, cmap='Blues')
ax.title.set_text('Rate of Y = True when X = True')
plt.show()




epid_indexes = []
sudu_indexes = []
inpa_indexes = []
inve_indexes = []
suar_indexes = []
none_indexes = []

it = np.nditer(data['labels'], flags=['f_index', 'refs_ok'])
while not it.finished:
    if 'epidural' in it[0].tolist():
        epid_indexes.append(it.index)
    if 'subdural' in it[0].tolist():
        sudu_indexes.append(it.index)
    if 'intraparenchymal' in it[0].tolist():
        inpa_indexes.append(it.index)
    if 'intraventricular' in it[0].tolist():
        inve_indexes.append(it.index)
    if 'subarachnoid' in it[0].tolist():
        suar_indexes.append(it.index)
    if 'none' in it[0].tolist():
        none_indexes.append(it.index)
    it.iternext()




X_temp = tdata['images']/255

X = np.empty(shape=[X_temp.shape[0]] + [2500], dtype='float32')
print(X_temp.shape)

for im in range(X_temp.shape[0]):
    X[im,:] = X_temp[im,:,:].flatten()

mlb = MultiLabelBinarizer()
y = mlb.fit_transform(tdata['labels'])

print(y[:10,:])
print(X_temp.shape)
print(X.shape)
print(y.shape)
X[0,:]




X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=50)

print (X_train.shape)
print (y_train.shape)
stdscaler = preprocessing.StandardScaler().fit(X_train)

X_train_scaled = stdscaler.transform(X_train)
X_test_scaled  = stdscaler.transform(X_test)




label_count = len(mlb.classes_)

model = Sequential()
model.add(Dense(label_count*24, input_shape=[2500], activation='relu', W_regularizer=l2(0.1)))
model.add(Dense(label_count*18, activation='relu', W_regularizer=l2(0.01)))
model.add(Dropout(0.3))
model.add(Dense(label_count*12, activation='relu', W_regularizer=l2(0.01)))
model.add(Dropout(0.2))
model.add(Dense(label_count*8, activation='relu', W_regularizer=l2(0.001)))
model.add(Dense(label_count, activation='sigmoid', W_regularizer=l1(0.001)))


sgd = SGD(lr=0.5)
model.compile(loss='binary_crossentropy', optimizer=sgd, metrics=['accuracy'])




model.summary()




print(X_train_scaled[0])
print(y_train[0])
print(X_test_scaled[0])
print(y_test[0])




history = model.fit(X_train_scaled, y_train, batch_size = 256, 
                    epochs = 50, verbose=2, validation_data=(X_test_scaled, y_test))
fig = plt.figure(figsize=(6,4))

# Summary of loss history
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'], 'g--')
plt.title('Model Loss')
plt.ylabel('Binary Crossentropy')
plt.xlabel('Epoch')
plt.legend(['Training Loss', 'Testing Loss'], loc='upper right')
plt.title('Loss After Final Iteration: '+str(history.history['val_loss'][-1]))
print ("BC after final iteration: ", history.history['val_loss'][-1])
plt.show()




fig = plt.figure(figsize=(6,4))

# Summary of accuracy history
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'], 'g--')
plt.title('Model Accuracy')
plt.ylabel('Model Accuracy')
plt.xlabel('Epoch')
plt.legend(['Training Accuracy', 'Testing Accuracy'], loc='lower left')
plt.title('Accuracy After Final Iteration: '+str(history.history['val_accuracy'][-1]))
print ("Accuracy after final iteration: ", history.history['val_accuracy'][-1])
plt.show()




test_X_temp = test_data['images']/255

test_X = np.empty(shape=[test_X_temp.shape[0]] + [2500], dtype='float32')
print(test_X_temp.shape)

for im in range(test_X_temp.shape[0]):
    test_X[im,:] = test_X_temp[im,:,:].flatten()

test_mlb = MultiLabelBinarizer()
test_y = test_mlb.fit_transform(test_data['labels'])

print(test_y[:10])
print(test_X_temp.shape)
print(test_X.shape)
print(test_y.shape)
test_X[0,:]

test_stdscaler = preprocessing.StandardScaler().fit(test_X)

test_X_scaled = test_stdscaler.transform(test_X)
#test_X_test_scaled  = test_stdscaler.transform(X_test)




test_y_pred = model.predict(X_train_scaled, batch_size=256, verbose=2)
test_y_actual = np.argmax(test_y_pred, axis=1)





print(test_y_pred[4])
print(y_train[4])
test_y_pred_rounded = np.around(test_y_pred)
test_y_pred_rounded[4]




from skmultilearn.problem_transform import ClassifierChain
from sklearn.linear_model import LogisticRegression
# initialize classifier chains multi-label classifier
classifier = ClassifierChain(LogisticRegression())
# Training logistic regression model on train data
classifier.fit(X_train_scaled, y_train)
# predict




predictions = classifier.predict(X_train_scaled)
# accuracy
print("Accuracy = ",accuracy_score(y_test,predictions))
print("\n")






