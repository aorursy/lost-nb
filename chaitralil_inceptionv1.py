#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np 
import pandas as pd 
import cv2
from matplotlib import pyplot as plt
import os
from subprocess import check_output
import cv2
from PIL import Image
import glob


# In[2]:


trainLabels = pd.read_csv("../input/labels/trainLabels.csv")
print(trainLabels.head())

filelist = glob.glob('../input/bloodvessel/bloodvesselextraction/BloodVesselExtraction/*.jpeg')
##filelist = glob.glob('../input/diabetic-retinopathy-detection/*.jpeg')
np.size(filelist)


# In[3]:


img_data = []
img_label = []
img_r = 512
img_c = 512
for file in filelist:
    tmp = cv2.imread(file)
    tmp = cv2.resize(tmp,(img_r, img_c), interpolation = cv2.INTER_CUBIC)
    tmp = cv2.cvtColor(tmp, cv2.COLOR_BGR2GRAY)
    img_data.append(np.array(tmp).flatten())
    tmpfn = file
    tmpfn = tmpfn.replace("../input/bloodvessel/bloodvesselextraction/BloodVesselExtraction/","")
    ##tmpfn = tmpfn.replace("../input/diabetic-retinopathy-detection/","")
    tmpfn = tmpfn.replace(".jpeg","")
    img_label.append(trainLabels.loc[trainLabels.image==tmpfn, 'level'].values[0])


# In[4]:


data = pd.DataFrame({'img_data':img_data,'label':img_label})
data.sample(3)


# In[5]:


data[['label']].hist(figsize = (10, 5))


# In[6]:


from sklearn.model_selection import train_test_split
X = data['img_data']
y = data['label']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


from sklearn.utils import shuffle

data,label = shuffle(X_train,y_train, random_state=2)
train_data = pd.DataFrame({'data': data, 'label':label})
train_df = train_data.groupby(['label']).apply(lambda x: x.sample(160, replace = True)
                                                      ).reset_index(drop = True)
print('New Data Size:', train_df.shape[0], 'Old Size:', train_data.shape[0])
train_df[['label']].hist(figsize = (10, 5))


# In[7]:


X_train = train_df['data']
y_train = train_df['label']

X_train = np.asarray(X_train)
y_train = np.asarray(y_train)
X_test = np.asarray(X_test)
y_test = np.asarray(y_test)


# In[8]:


X_train_resh = np.zeros([X_train.shape[0],img_r, img_c, 1])
for i in range (X_train.shape[0]-1):
    X_train_resh[i] = np.reshape(X_train[i], (img_r, img_c, 1))
    
X_test_resh = np.zeros([X_test.shape[0],img_r, img_c, 1])
for i in range (X_test.shape[0]-1):
    X_test_resh[i] = np.reshape(X_test[i], (img_r, img_c, 1))
print(X_test_resh.shape)


# In[9]:


from keras.utils import np_utils
nb_classes = 5
Y_train = np_utils.to_categorical(y_train, nb_classes)
Y_test = np_utils.to_categorical(y_test, nb_classes)


# In[10]:


import matplotlib.pyplot as plt
import matplotlib

img=X_train_resh[100].reshape(img_r,img_c)
plt.imshow(img)
plt.imshow(img,cmap='gray')


# In[11]:


import keras
from keras.layers.core import Layer
import keras.backend as K
import tensorflow as tf

from keras.models import Model
from keras.layers import Conv2D, MaxPool2D,      Dropout, Dense, Input, concatenate,          GlobalAveragePooling2D, AveragePooling2D,    Flatten

import cv2 
import numpy as np 
from keras.datasets import cifar10 
from keras import backend as K 
from keras.utils import np_utils

import math 
from keras.optimizers import SGD 
from keras.callbacks import LearningRateScheduler

def inception_module(x,
                     filters_1x1,
                     filters_3x3_reduce,
                     filters_3x3,
                     filters_5x5_reduce,
                     filters_5x5,
                     filters_pool_proj,
                     name=None):
    
    conv_1x1 = Conv2D(filters_1x1, (1, 1), padding='same', activation='relu', kernel_initializer=kernel_init, bias_initializer=bias_init)(x)
    
    conv_3x3 = Conv2D(filters_3x3_reduce, (1, 1), padding='same', activation='relu', kernel_initializer=kernel_init, bias_initializer=bias_init)(x)
    conv_3x3 = Conv2D(filters_3x3, (3, 3), padding='same', activation='relu', kernel_initializer=kernel_init, bias_initializer=bias_init)(conv_3x3)

    conv_5x5 = Conv2D(filters_5x5_reduce, (1, 1), padding='same', activation='relu', kernel_initializer=kernel_init, bias_initializer=bias_init)(x)
    conv_5x5 = Conv2D(filters_5x5, (5, 5), padding='same', activation='relu', kernel_initializer=kernel_init, bias_initializer=bias_init)(conv_5x5)

    pool_proj = MaxPool2D((3, 3), strides=(1, 1), padding='same')(x)
    pool_proj = Conv2D(filters_pool_proj, (1, 1), padding='same', activation='relu', kernel_initializer=kernel_init, bias_initializer=bias_init)(pool_proj)

    output = concatenate([conv_1x1, conv_3x3, conv_5x5, pool_proj], axis=3, name=name)
    
    return output

kernel_init = keras.initializers.glorot_uniform()
bias_init = keras.initializers.Constant(value=0.2)

input_layer = Input(shape=(512, 512, 1))

x = Conv2D(64, (7, 7), padding='same', strides=(2, 2), activation='relu', name='conv_1_7x7/2', kernel_initializer=kernel_init, bias_initializer=bias_init)(input_layer)
x = MaxPool2D((3, 3), padding='same', strides=(2, 2), name='max_pool_1_3x3/2')(x)
x = Conv2D(64, (1, 1), padding='same', strides=(1, 1), activation='relu', name='conv_2a_3x3/1')(x)
x = Conv2D(192, (3, 3), padding='same', strides=(1, 1), activation='relu', name='conv_2b_3x3/1')(x)
x = MaxPool2D((3, 3), padding='same', strides=(2, 2), name='max_pool_2_3x3/2')(x)

x = inception_module(x,
                     filters_1x1=64,
                     filters_3x3_reduce=96,
                     filters_3x3=128,
                     filters_5x5_reduce=16,
                     filters_5x5=32,
                     filters_pool_proj=32,
                     name='inception_3a')

x = inception_module(x,
                     filters_1x1=128,
                     filters_3x3_reduce=128,
                     filters_3x3=192,
                     filters_5x5_reduce=32,
                     filters_5x5=96,
                     filters_pool_proj=64,
                     name='inception_3b')

x = MaxPool2D((3, 3), padding='same', strides=(2, 2), name='max_pool_3_3x3/2')(x)

x = inception_module(x,
                     filters_1x1=192,
                     filters_3x3_reduce=96,
                     filters_3x3=208,
                     filters_5x5_reduce=16,
                     filters_5x5=48,
                     filters_pool_proj=64,
                     name='inception_4a')


x1 = AveragePooling2D((5, 5), strides=3)(x)
x1 = Conv2D(128, (1, 1), padding='same', activation='relu')(x1)
x1 = Flatten()(x1)
x1 = Dense(6400, activation='relu')(x1)
x1 = Dense(3200, activation='relu')(x1)
x1 = Dense(1600, activation='relu')(x1)
x1 = Dense(1024, activation='relu')(x1)
x1 = Dense(800, activation='relu')(x1)
x1 = Dense(400, activation='relu')(x1)
x1 = Dense(200, activation='relu')(x1)
x1 = Dropout(0.7)(x1)
x1 = Dense(5, activation='softmax', name='auxilliary_output_1')(x1)

x = inception_module(x,
                     filters_1x1=160,
                     filters_3x3_reduce=112,
                     filters_3x3=224,
                     filters_5x5_reduce=24,
                     filters_5x5=64,
                     filters_pool_proj=64,
                     name='inception_4b')

x = inception_module(x,
                     filters_1x1=128,
                     filters_3x3_reduce=128,
                     filters_3x3=256,
                     filters_5x5_reduce=24,
                     filters_5x5=64,
                     filters_pool_proj=64,
                     name='inception_4c')

x = inception_module(x,
                     filters_1x1=112,
                     filters_3x3_reduce=144,
                     filters_3x3=288,
                     filters_5x5_reduce=32,
                     filters_5x5=64,
                     filters_pool_proj=64,
                     name='inception_4d')


x2 = AveragePooling2D((5, 5), strides=3)(x)
x2 = Conv2D(128, (1, 1), padding='same', activation='relu')(x2)
x2 = Flatten()(x2)
x2 = Dense(6400, activation='relu')(x2)
x2 = Dense(3200, activation='relu')(x2)
x2 = Dense(1600, activation='relu')(x2)
x2 = Dense(1024, activation='relu')(x2)
x2 = Dense(800, activation='relu')(x2)
x2 = Dense(400, activation='relu')(x2)
x2 = Dense(200, activation='relu')(x2)
x2 = Dropout(0.7)(x2)
x2 = Dense(5, activation='softmax', name='auxilliary_output_2')(x2)

x = inception_module(x,
                     filters_1x1=256,
                     filters_3x3_reduce=160,
                     filters_3x3=320,
                     filters_5x5_reduce=32,
                     filters_5x5=128,
                     filters_pool_proj=128,
                     name='inception_4e')

x = MaxPool2D((3, 3), padding='same', strides=(2, 2), name='max_pool_4_3x3/2')(x)

x = inception_module(x,
                     filters_1x1=256,
                     filters_3x3_reduce=160,
                     filters_3x3=320,
                     filters_5x5_reduce=32,
                     filters_5x5=128,
                     filters_pool_proj=128,
                     name='inception_5a')

x = inception_module(x,
                     filters_1x1=384,
                     filters_3x3_reduce=192,
                     filters_3x3=384,
                     filters_5x5_reduce=48,
                     filters_5x5=128,
                     filters_pool_proj=128,
                     name='inception_5b')

x = GlobalAveragePooling2D(name='avg_pool_5_3x3/1')(x)
x = Dense(6400, activation='relu')(x)
x = Dense(3200, activation='relu')(x)
x = Dense(1600, activation='relu')(x)
x = Dense(1024, activation='relu')(x)
x = Dense(800, activation='relu')(x)
x = Dense(400, activation='relu')(x)
x = Dense(200, activation='relu')(x)
x = Dropout(0.4)(x)

x = Dense(5, activation='softmax', name='output')(x)

model = Model(input_layer, [x, x1, x2], name='inception_v1')
print(model.summary())


# In[12]:


sgd = SGD(lr = 0.01, momentum=0.9, nesterov=False)
model.compile(loss=['categorical_crossentropy', 'categorical_crossentropy', 'categorical_crossentropy'], loss_weights=[1, 0.3, 0.3], optimizer=sgd, metrics=['accuracy'])
from keras.callbacks import ReduceLROnPlateau
reduceLROnPlat = ReduceLROnPlateau(monitor='val_loss', factor=0.8, patience=3, verbose=1, mode='auto', epsilon=0.0001, cooldown=5, min_lr=0.0001)


# In[13]:


history = model.fit(X_train_resh, [Y_train, Y_train, Y_train], validation_data=(X_test_resh, [Y_test, Y_test, Y_test]), epochs=200, batch_size=32, callbacks=[reduceLROnPlat])


# In[14]:


import matplotlib.pyplot as pyplot
pyplot.subplot(211)
pyplot.title('Cross-Entropy Loss', pad=-40)
pyplot.plot(history.history['loss'], label='train')
pyplot.plot(history.history['val_loss'], label='test')
pyplot.legend()
# plot accuracy learning curves
pyplot.subplot(212)
pyplot.title('Accuracy', pad=-40)
pyplot.plot(history.history['output_acc'], label='train')
pyplot.plot(history.history['val_output_acc'], label='test')
pyplot.legend()
pyplot.show()


# In[15]:


pyplot.savefig("trainmetrics.png")


# In[16]:


from sklearn.metrics import accuracy_score, classification_report
pred_Y = model.predict(X_test_resh, batch_size = 32, verbose = True)

pred_Y_cat = np.argmax(pred_Y, -1)
#print(pred_Y_cat[0])

test_Y_cat = np.argmax(Y_test, -1)
print('Accuracy on Test Data: %2.2f%%' % (accuracy_score(test_Y_cat, pred_Y_cat[0])))
print(classification_report(test_Y_cat, pred_Y_cat[0]))


# In[17]:


import seaborn as sns
from sklearn.metrics import confusion_matrix
sns.heatmap(confusion_matrix(test_Y_cat, pred_Y_cat[0]), 
            annot=True, fmt="d", cbar = False, cmap = plt.cm.Blues, vmax = X_test_resh.shape[0]//16)


# In[18]:


pre = pred_Y[0]

from sklearn.metrics import roc_curve, roc_auc_score
sick_vec = test_Y_cat>0
sick_score = np.sum(pre[:,1:],1)
fpr, tpr, _ = roc_curve(sick_vec, sick_score)
fig, ax1 = plt.subplots(1,1, figsize = (6, 6), dpi = 150)
ax1.plot(fpr, tpr, 'b.-', label = 'Model Prediction (AUC: %2.2f)' % roc_auc_score(sick_vec, sick_score))
ax1.plot(fpr, fpr, 'g-', label = 'Random Guessing')
ax1.legend()
ax1.set_xlabel('False Positive Rate')
ax1.set_ylabel('True Positive Rate');


# In[19]:


fig.savefig("roc.png")


# In[20]:


from keras.models import model_from_json
# Model to JSON
model_json = model.to_json()
with open("model_project_work.json", "w") as json_file:
    json_file.write(model_json)
# Weights to HDF5
model.save("model_project_work.h5")


# In[21]:


os.listdir("../working")


# In[22]:


from keras.models import load_model
new_model = load_model('model_project_work.h5')


# In[23]:


pred_Y = new_model.predict(X_test_resh, batch_size = 32, verbose = True)

pred_Y_cat = np.argmax(pred_Y, -1)
#print(pred_Y_cat[0])

test_Y_cat = np.argmax(Y_test, -1)
print('Accuracy on Test Data: %2.2f%%' % (accuracy_score(test_Y_cat, pred_Y_cat[0])))
print(classification_report(test_Y_cat, pred_Y_cat[0]))

