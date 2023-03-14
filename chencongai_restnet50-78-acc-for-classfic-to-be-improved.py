#!/usr/bin/env python
# coding: utf-8



import os
import pandas as pd
import shutil


fileList = os.listdir("./") # get file list in the path directory

# list files
for f in fileList: 
    print(f)
fileList2 = os.listdir("./model") # get file list in the path directory
# list files
for f2 in fileList2: 
    print(f2)

#shutil.os.mkdir("./graph")
#shutil.os.mkdir("./model")
#shutil.os.mkdir("./model/model.h5")
#shutil.os.mkdir("./model/model.json")




# So until here we have input : (m,1024,1024,1), output : (m,1)
from keras import layers
from keras.layers import Input, Add, Dense, Activation, ZeroPadding2D, BatchNormalization, Flatten, Conv2D,     AveragePooling2D, MaxPooling2D, GlobalMaxPooling2D
from keras.models import Model, load_model
from keras.preprocessing import image
from keras.utils import layer_utils
from keras.utils.data_utils import get_file
from keras.applications.imagenet_utils import preprocess_input
from keras.utils.vis_utils import model_to_dot
from keras.utils import plot_model
from keras.initializers import glorot_uniform 
import scipy.misc
import keras.backend as K

K.set_image_data_format('channels_last')
K.set_learning_phase(1)


def identity_block(X, f, filters, stage, block):
    """
    Implementation of the identity block as defined in Figure 3

    Arguments:
    X -- input tensor of shape (m, n_H_prev, n_W_prev, n_C_prev)
    f -- integer, specifying the shape of the middle CONV's window for the main path
    filters -- python list of integers, defining the number of filters in the CONV layers of the main path
    stage -- integer, used to name the layers, depending on their position in the network
    block -- string/character, used to name the layers, depending on their position in the network

    Returns:
    X -- output of the identity block, tensor of shape (n_H, n_W, n_C)
    """

    # defining name basis
    conv_name_base = 'res' + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branch'

    # Retrieve Filters: number of filters
    F1, F2, F3 = filters

    # Save the input value. You'll need this later to add back to the main path.
    X_shortcut = X

    # First component of main path
    X = Conv2D(filters=F1, kernel_size=(1, 1), strides=(1, 1), padding='valid', name=conv_name_base + '2a',
               kernel_initializer=glorot_uniform(seed=0))(X)
    X = BatchNormalization(axis=3, name=bn_name_base + '2a')(X)
    X = Activation('relu')(X)

    # Second component of main path (≈3 lines)
    X = Conv2D(filters=F2, kernel_size=(f, f), strides=(1, 1), padding='same', name=conv_name_base + '2b',
               kernel_initializer=glorot_uniform(seed=0))(X)
    X = BatchNormalization(axis=3, name=bn_name_base + '2b')(X)
    X = Activation('relu')(X)

    # Third component of main path (≈2 lines)
    X = Conv2D(filters=F3, kernel_size=(1, 1), strides=(1, 1), padding='valid', name=conv_name_base + '2c',
               kernel_initializer=glorot_uniform(seed=0))(X)
    X = BatchNormalization(axis=3, name=bn_name_base + '2c')(X)

    # Final step: Add shortcut value to main path, and pass it through a RELU activation (≈2 lines)
    X = Add()([X, X_shortcut])
    X = Activation('relu')(X)

    return X


def convolutional_block(X, f, filters, stage, block, s=2):
    """
    Implementation of the convolutional block as defined in Figure 4

    Arguments:
    X -- input tensor of shape (m, n_H_prev, n_W_prev, n_C_prev)
    f -- integer, specifying the shape of the middle CONV's window for the main path
    filters -- python list of integers, defining the number of filters in the CONV layers of the main path
    stage -- integer, used to name the layers, depending on their position in the network
    block -- string/character, used to name the layers, depending on their position in the network
    s -- Integer, specifying the stride to be used

    Returns:
    X -- output of the convolutional block, tensor of shape (n_H, n_W, n_C)
    """

    # defining name basis
    conv_name_base = 'res' + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branch'

    # Retrieve Filters
    F1, F2, F3 = filters

    # Save the input value
    X_shortcut = X

    ##### MAIN PATH #####
    # First component of main path
    X = Conv2D(F1, (1, 1), strides=(s, s), padding='valid', name=conv_name_base + '2a',
               kernel_initializer=glorot_uniform(seed=0))(X)
    X = BatchNormalization(axis=3, name=bn_name_base + '2a')(X)
    X = Activation('relu')(X)

    # Second component of main path (≈3 lines)
    X = Conv2D(F2, (f, f), strides=(1, 1), padding='same', name=conv_name_base + '2b',
               kernel_initializer=glorot_uniform(seed=0))(X)
    X = BatchNormalization(axis=3, name=bn_name_base + '2b')(X)
    X = Activation('relu')(X)

    # Third component of main path (≈2 lines)
    X = Conv2D(F3, (1, 1), strides=(1, 1), padding='valid', name=conv_name_base + '2c',
               kernel_initializer=glorot_uniform(seed=0))(X)
    X = BatchNormalization(axis=3, name=bn_name_base + '2c')(X)

    ##### SHORTCUT PATH #### (≈2 lines)
    X_shortcut = Conv2D(F3, (1, 1), strides=(s, s), padding='valid', name=conv_name_base + '1',
                        kernel_initializer=glorot_uniform(seed=0))(X_shortcut)
    X_shortcut = BatchNormalization(axis=3, name=bn_name_base + '1')(X_shortcut)

    # Final step: Add shortcut value to main path, and pass it through a RELU activation (≈2 lines)
    X = Add()([X, X_shortcut])
    X = Activation('relu')(X)

    return X


#ResNet50

def ResNet50(input_shape=(1024, 1024, 1), classes=1):
    """
    Implementation of the popular ResNet50 the following architecture:
    CONV2D -> BATCHNORM -> RELU -> MAXPOOL -> CONVBLOCK -> IDBLOCK*2 -> CONVBLOCK -> IDBLOCK*3
    -> CONVBLOCK -> IDBLOCK*5 -> CONVBLOCK -> IDBLOCK*2 -> AVGPOOL -> TOPLAYER

    Arguments:
    input_shape -- shape of the images of the dataset
    classes -- integer, number of classes

    Returns:
    model -- a Model() instance in Keras
    """

    # Define the input as a tensor with shape input_shape
    X_input = Input(input_shape)

    # Zero-Padding
    X = ZeroPadding2D((3, 3))(X_input)

    # Stage 1
    X = Conv2D(64, (7, 7), strides=(2, 2), name='conv1', kernel_initializer=glorot_uniform(seed=0))(X)
    X = BatchNormalization(axis=3, name='bn_conv1')(X)
    X = Activation('relu')(X)
    X = MaxPooling2D((3, 3), strides=(2, 2))(X)

    # Stage 2
    X = convolutional_block(X, f=3, filters=[64, 64, 256], stage=2, block='a', s=1)
    X = identity_block(X, 3, [64, 64, 256], stage=2, block='b')
    X = identity_block(X, 3, [64, 64, 256], stage=2, block='c')

    # Stage 3 (≈4 lines)
    X = convolutional_block(X, f=3, filters=[128, 128, 512], stage=3, block='a', s=2)
    X = identity_block(X, 3, [128, 128, 512], stage=3, block='b')
    X = identity_block(X, 3, [128, 128, 512], stage=3, block='c')
    X = identity_block(X, 3, [128, 128, 512], stage=3, block='d')

    # Stage 4 (≈6 lines)
    X = convolutional_block(X, f=3, filters=[256, 256, 1024], stage=4, block='a', s=2)
    X = identity_block(X, 3, [256, 256, 1024], stage=4, block='b')
    X = identity_block(X, 3, [256, 256, 1024], stage=4, block='c')
    X = identity_block(X, 3, [256, 256, 1024], stage=4, block='d')
    X = identity_block(X, 3, [256, 256, 1024], stage=4, block='e')
    X = identity_block(X, 3, [256, 256, 1024], stage=4, block='f')

    # Stage 5 (≈3 lines)
    X = convolutional_block(X, f=3, filters=[512, 512, 2048], stage=5, block='a', s=2)
    X = identity_block(X, 3, [512, 512, 2048], stage=5, block='b')
    X = identity_block(X, 3, [512, 512, 2048], stage=5, block='c')

    print('X.shape : ', X.shape)
    # AVGPOOL (≈1 line). Use "X = AveragePooling2D(...)(X)"
    X = AveragePooling2D((2, 2), name='avg_pool')(X)

    print('X.shape : ', X.shape)
    # output layer
    X = Flatten()(X)
    #X_FOR_BOUNDINGBOX = X
    print('X.shape : ', X.shape)
    X = Dense(classes, activation='sigmoid', name='fc' + str(classes), kernel_initializer=glorot_uniform(seed=0))(X)
    #X_FOR_BOUNDINGBOX = Dense(classes, activation='softmax', name='fc' + str(classes), kernel_initializer=glorot_uniform(seed=0))(X_FOR_BOUNDINGBOX)
    print('X.shape after Dense : ', X.shape)

    # Create model
    model = Model(inputs=X_input, outputs=X, name='ResNet50')

    return model




import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
import math
import pydicom
from skimage.transform import rescale, resize, downscale_local_mean
from keras.optimizers import Adam
from keras.callbacks import TensorBoard
import os
import time

#get all image file
train_images_file_list = os.listdir("../input/stage_1_train_images")
test_images_file_list = os.listdir("../input/stage_1_test_images")
#25684 images

#-----stage_1_train_labels.csv-----
data_stage_1_train_labels = pd.read_csv(("../input/stage_1_train_labels.csv"))
'''
['patientId', 'x', 'y', 'width', 'height', 'Target']
'''

#-----stage_1_detailed_class_info.csv-----
data_stage_1_detailed_class_info = pd.read_csv(("../input/stage_1_detailed_class_info.csv"))
'''
Provides detailed information about the type of positive or negative class for each image:
['patientId', 'class']
'''


#-----stage_1_sample_submission.csv-----
data_stage_1_sample_submission = pd.read_csv(("../input/stage_1_sample_submission.csv"))
'''
It is just one sample datasets :
['patientId', 'PredictionString']
patientId PredictionString
0  000924cf-0f8d-42bd-9158-1af53881a557  0.5 0 0 100 100
'''

'''
m_stage_1_train_images :  16226
m_stage_1_test_images :  1000
train image shape : (1024, 1024,1)

Need to create the data_train_input_image : (16626,1024,1024,1)
'''
data_train_input_patientid = []
data_train_input_image=[]
data_test_input_image_file = []
data_test_input_image=[]
#use small train data
small_m=5000#16225
small_m_test=10
imagesize = 64
for i in range(small_m):
    dcmFile = "../input/stage_1_train_images/"+train_images_file_list[i]
    readDcmFile = pydicom.dcmread(dcmFile)
    data_train_input_patientid.append(readDcmFile.PatientID)
    dcmPixelArray = readDcmFile.pixel_array
    #change the image from (1024,1024) to (64,64)
    image_resized = resize(dcmPixelArray, (dcmPixelArray.shape[0] / 16, dcmPixelArray.shape[1] / 16),anti_aliasing=True)
    data_train_input_image.append(image_resized)

data_train_input_image_arr = np.asarray(data_train_input_image)
data_train_input_image_arr = data_train_input_image_arr.reshape(small_m,imagesize,imagesize)

data_train_input_image_arr = data_train_input_image_arr/255

#transform to (m,1024,1024,1)
data_train_input_image_arr = data_train_input_image_arr.reshape(small_m,imagesize,imagesize,1)
data_train_input_image_arr[:,[2]] = 1

'''
I will use classfication + regression model to train the 'target' and bounding box
I begin with classfication :
classfication :input : (m,1024,1024) ['patientId','Target']
               output : (m , 1)
'''
small_data_stage_1_train_labels_patientId = []
small_data_stage_1_train_labels_target = []
#{patientId:target}
patientid_target = {}
data_stage_1_train_labels_patientId = data_stage_1_train_labels['patientId']
data_stage_1_train_labels_target = data_stage_1_train_labels['Target']
#transform into dictionary
for i in range(len(data_stage_1_train_labels_patientId)):
    small_data_stage_1_train_labels_patientId.append(data_stage_1_train_labels_patientId[i])
    patientid_target[data_stage_1_train_labels_patientId[i]] = data_stage_1_train_labels_target[i]
for i in range(len(data_train_input_patientid)):
    patientId = data_train_input_patientid[i]
    try:
        small_data_stage_1_train_labels_target.append(patientid_target[patientId])
    except:
        small_data_stage_1_train_labels_target.append(np.nan)
small_data_stage_1_train_labels_target_arr = np.asarray(small_data_stage_1_train_labels_target).reshape(small_m,1)


#delete NaN from small_data_stage_1_train_labels_target_arr and not use image corresponded
idx_delete=[]
for i in range(len(small_data_stage_1_train_labels_target_arr)):
    if (math.isnan(small_data_stage_1_train_labels_target_arr[i][0])):
        idx_delete.append(i)

small_data_stage_1_train_labels_target_arr = np.delete(small_data_stage_1_train_labels_target_arr, idx_delete, 0)
data_train_input_image_arr = np.delete(data_train_input_image_arr, idx_delete, 0)


model = ResNet50(input_shape = (imagesize, imagesize,1), classes = 1)

tbCallBack = TensorBoard(log_dir='./graph', histogram_freq=0, write_graph=True, write_images=True)

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
model.fit(data_train_input_image_arr, small_data_stage_1_train_labels_target_arr, epochs = 9, batch_size = 64,
          callbacks=[tbCallBack])

#save model
'''
print("Save model ongoing.....")

if not os.path.exists("./model/model.json"):
    with open('./model/model.json', 'w'):
        pass
    
if not os.path.exists("./model/model.h5"):
    with open('./model/model.h5', 'w'):
        pass
model_json = model.to_json()
with open("./model/model.json","w") as json_file:
    json_file.write(model_json)
model.save_weights("./model/model.h5")
print("Saved model to disk")
'''






5000/5000 [==============================] - 573s 115ms/step - loss: 0.4589 - acc: 0.7794
Epoch 9/9
64/5000 [..............................] - ETA: 9:23 - loss: 0.3616 - acc: 0.8594

I have  acc : 0.85 here(first 64 batch of the 9th epoch)  
                
4928/5000 [============================>.] - ETA: 8s - loss: 0.4620 - acc: 0.7845

4992/5000 [============================>.] - ETA: 0s - loss: 0.4623 - acc: 0.7851

5000/5000 [==============================] - 560s 112ms/step - loss: 0.4624 - acc: 0.7850

