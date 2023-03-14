%matplotlib inline
import numpy as np
import pandas as pd
import cv2
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import os
import seaborn as sns
import gc
!pip install segmentation-models
!pip install git+https://github.com/qubvel/segmentation_models 

pip install --upgrade pip

trainImgPath = "/kaggle/input/severstal-steel-defect-detection/train_images/"
trainCsv = "/kaggle/input/severstal-steel-defect-detection/train.csv"
data=pd.read_csv(trainCsv)
data.ClassId=data.ClassId.astype(int)

train_Img_Id = []
train_class_Id = []
for i in os.listdir(trainImgPath):
    for j in range(1,5):
        train_Img_Id.append(i)
        train_class_Id.append(j)
train_Imgs = pd.DataFrame(train_Img_Id,columns=['ImageId'])
train_Imgs['ClassId'] = train_class_Id
train_Imgs.head(10)
        

train_data = pd.merge(train_Imgs,data ,how='outer', on=['ImageId','ClassId']) 
train_data = train_data.fillna('') 
train_data.head(10)

train_data = pd.pivot_table(train_data, values='EncodedPixels', index='ImageId',columns='ClassId', aggfunc=np.sum).astype(str)
train_data = train_data.reset_index() # add Index column to one level with classID   
train_data.columns = ['ImageId','Defect_1','Defect_2','Defect_3','Defect_4']
train_data.columns

train_data.head(15) 

has_defect = []
stratify = []
for index,row in train_data.iterrows():
    if row.Defect_1 or row.Defect_2 or row.Defect_3 or row.Defect_4: 
        has_defect.append(1)
    else:
        has_defect.append(0)
        
train_data["has_defect"] = has_defect 
 
 
for index , row in train_data.iterrows():
    if row.Defect_1 != '':
        stratify.append(1)
    elif row.Defect_2 != '':
        stratify.append(2)
    elif row.Defect_3 != '':
        stratify.append(3)
    elif row.Defect_4 != '':
        stratify.append(4)
    else:
        stratify.append(0)
        

        
train_data["stratify"] = stratify   
train_data.head(15)


x_train, x_test = train_test_split(train_data, test_size = 0.1, stratify=train_data['stratify'], random_state=42)
x_train, x_val = train_test_split(x_train, test_size = 0.2, stratify = x_train['stratify'], random_state=42)
print(x_train.shape, x_val.shape, x_test.shape)

# Some Data Analysis & Data Visualization for the Data Set 
nums_def_1 = sum(x_train.Defect_1 != '')
nums_def_2 = sum(x_train.Defect_2 != '')
nums_def_3 = sum(x_train.Defect_3 != '')
nums_def_4 = sum(x_train.Defect_4 != '')

print ( "Number of Images In Train Dataset is : ", len(x_train), '\n' ) 
print ( "Number of Images In Defect (1) : ", nums_def_1, '\n' ) 
print ( "Number of Images In Defect (2) : ", nums_def_2, '\n' ) 
print ( "Number of Images In Defect (3) : ", nums_def_3, '\n' ) 
print ( "Number of Images In Defect (4) : ", nums_def_4, '\n' ) 

sum_of_defects = [nums_def_1 ,nums_def_2,nums_def_3,nums_def_4]
x_axis = ['1' , '2' , '3' , '4']
fig, ax = plt.subplots()
sns.barplot(x=x_axis,y=sum_of_defects) 
ax.set_title("Number of images for each Defect")
ax.set_xlabel("Label")
plt.show()


zero_defects = 0
one_defect = 0
multi_defect = 0

for index,row in x_train.iterrows():
    cnt = 0
    if row.Defect_1:
        cnt+=1
    if row.Defect_2:
        cnt+=1
    if row.Defect_3:
        cnt+=1
    if row.Defect_4:
        cnt+=1
        
    if cnt > 1:
        multi_defect += 1
    elif cnt == 0:
        zero_defects += 1
    else:
        one_defect += 1 
        

print( zero_defects )
print(one_defect )
print(multi_defect)     
    

num_of_defects = [zero_defects ,one_defect,multi_defect]
x_axis = [ 'No Defects' , '1 label' ,'multi label']
fig, ax = plt.subplots()
sns.barplot(x=x_axis,y=num_of_defects) 
ax.set_title("Number of defects in images..")
ax.set_xlabel("Label")
plt.show()

def convert_to_mask(encoded_pixels):
    counts=[]
    mask=np.zeros((256*1600), dtype=np.uint8) #don't change this
    pre_mask=np.asarray([int(point) for point in encoded_pixels.split()])
    for index,count in enumerate(pre_mask):
        if(index%2!=0):
            counts.append(count)
    i=0
    for index,pixel in enumerate(pre_mask):
        if(index%2==0):
            if(i==len(counts)):
                break
            mask[pixel:pixel+counts[i]]=1
            i+=1
    mask=np.reshape(mask,(1600,256)) #don't change this
    mask=cv2.resize(mask,(256,1600)).T
    return mask

print("Samples of Images that have Defect 1: ")
Defect1 = x_train[x_train.Defect_1 != ''] 
cnt = 0
for index ,row in Defect1[::-1].iterrows():
    if cnt == 5:
        break
    fig, (ax1,ax2) = plt.subplots(nrows = 1,ncols = 2,figsize=(15, 7))
    Img = cv2.imread( trainImgPath + row.ImageId )
    mask = convert_to_mask(row.Defect_1)
    ax1.imshow(Img)
    ax1.set_title(i[0])
    ax2.imshow(mask)
    cnt+=1
     

print("Samples of Images that have Defect 2: ")
Defect2 = x_train[x_train.Defect_2 != '']
#Defect2 = Defect2[::-1]
cnt = 0 
for index ,row in Defect2.iterrows():
    if cnt == 5:
        break
    fig, (ax1,ax2) = plt.subplots(nrows = 1,ncols = 2,figsize=(15, 7))
    Img = cv2.imread( trainImgPath + row.ImageId )
    mask = convert_to_mask(row.Defect_2)
    ax1.imshow(Img)
    ax1.set_title(i[0])
    ax2.imshow(mask)
    cnt+=1
    

print("Samples of Images that have Defect 3: ")
Defect3 = x_train[x_train.Defect_3 != ''] 
cnt = 0
for index ,row in Defect3[::-1].iterrows():
    if cnt == 5:
        break
    fig, (ax1,ax2) = plt.subplots(nrows = 1,ncols = 2,figsize=(15, 7))
    Img = cv2.imread( trainImgPath + row.ImageId )
    mask = convert_to_mask(row.Defect_3)
    ax1.imshow(Img)
    ax1.set_title(i[0])
    ax2.imshow(mask)
    cnt+=1

print("Samples of Images that have Defect 4: ")
Defect4 = x_train[x_train.Defect_4 != ''] 
cnt = 0
for index ,row in Defect4[::-1].iterrows():
    if cnt == 5:
        break
    fig, (ax1,ax2) = plt.subplots(nrows = 1,ncols = 2,figsize=(15, 7))
    Img = cv2.imread( trainImgPath + row.ImageId )
    mask = convert_to_mask(row.Defect_4)
    ax1.imshow(Img)
    ax1.set_title(i[0])
    ax2.imshow(mask)
    cnt+=1

Defect2.describe()

WIDTH=512
HEIGHT=256
TRAINING_SIZE=7095

print("sample of dataset that have defect_1 : ")
Img = cv2.imread( trainImgPath + x_train['ImageId'][0] ) 
plt.imshow(Img)
plt.show() 

mask = convert_to_mask(x_train['Defect_1'][0]) 
plt.imshow(mask)
plt.show()

epochs = 50 
from tensorflow.keras.utils import plot_model
import keras 
from keras import backend as K
from keras.layers import GlobalAveragePooling2D, Dense, Conv2D, BatchNormalization, Dropout
from keras.models import Model, load_model


def dice_coef(y_true, y_pred, smooth=1):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)

x_train_defected_non_defected = x_train[['ImageId','has_defect']]
x_val_defected_non_defected = x_val[['ImageId','has_defect']]
x_test_defected_non_defected = x_test[['ImageId','has_defect']] 
print(x_train_defected_non_defected.shape , x_val_defected_non_defected.shape,x_test_defected_non_defected.shape)


from keras.preprocessing.image import ImageDataGenerator 

train_datagen = ImageDataGenerator(rescale=1./255., shear_range=0.2, zoom_range=0.05, rotation_range=5,
                           width_shift_range=0.2, height_shift_range=0.2, horizontal_flip=True, vertical_flip=True)

test_datagen = ImageDataGenerator(rescale=1./255)

train_data_generator = train_datagen.flow_from_dataframe(
        dataframe=x_train_defected_non_defected.astype(str),
        directory=trainImgPath,
        x_col="ImageId",
        y_col="has_defect",
        target_size=(HEIGHT,WIDTH),
        batch_size=16,
        class_mode='binary') 

valid_data_generator = test_datagen.flow_from_dataframe(
        dataframe=x_val_defected_non_defected.astype(str),
        directory=trainImgPath,
        x_col="ImageId",
        y_col="has_defect",
        target_size=(HEIGHT,WIDTH),
        batch_size=16,
        class_mode='binary')





Classification_Model = keras.applications.xception.Xception(include_top = False, input_shape = (HEIGHT,WIDTH,3))

layer = Classification_Model.output
layer = GlobalAveragePooling2D()(layer)

layer = Dense(1024, activation='relu')(layer)
layer = BatchNormalization()(layer)
layer = Dropout(0.3)(layer)

layer = Dense(512, activation='relu')(layer)
layer = BatchNormalization()(layer)
layer = Dropout(0.3)(layer)

layer = Dense(64, activation='relu')(layer)
predictions = Dense(1, activation='sigmoid')(layer)
model = Model(inputs=Classification_Model.input, outputs=predictions)
model.summary()

import tensorflow as tf
from datetime import datetime
from tensorflow.python.keras.callbacks import TensorBoard
from keras.callbacks import ModelCheckpoint

model.compile(optimizer='adam', loss='binary_crossentropy',metrics=['accuracy'])
Training_Model = model.fit_generator(train_data_generator, validation_data = valid_data_generator, epochs = 30, verbose=1, callbacks = [mc,tensorboard_callback])

Training_Model.save('classification_model')


X_train_multi = x_train[['ImageId','has_defect_1','has_defect_2','has_defect_3','has_defect_4']][x_train['has_defect']==1]
X_val_multi = x_val[['ImageId','has_defect_1','has_defect_2','has_defect_3','has_defect_4']][x_val['has_defect']==1]
X_test_multi = x_test[['ImageId','has_defect_1','has_defect_2','has_defect_3','has_defect_4']][x_test['has_defect']==1]


train_DataGenerator_2 = ImageDataGenerator(rescale=1./255., shear_range=0.2, zoom_range=0.05, rotation_range=5,
                           width_shift_range=0.2, height_shift_range=0.2, horizontal_flip=True, vertical_flip=True)


train_generator = train_DataGenerator_2.flow_from_dataframe(
        dataframe=X_train_multi.astype(str),
        directory= trainImgPath,
        x_col="ImageId",
        y_col=["has_defect_1","has_defect_2","has_defect_3","has_defect_4"],
        target_size=(256,512),
        batch_size=16,
        class_mode='other')


test_DataGenerator_2 = ImageDataGenerator(rescale=1./255)
validation_generator = test_DataGenerator_2.flow_from_dataframe(
        dataframe=X_val_multi.astype(str),
        directory=trainImgPath,
        x_col="ImageId",
        y_col=["has_defect_1","has_defect_2","has_defect_3","has_defect_4"],
        target_size=(256,512),
        batch_size=16,
        class_mode='other')

train_DataGenerator_multi_class = ImageDataGenerator(rescale=1./255., shear_range=0.2, zoom_range=0.05, rotation_range=5,
                           width_shift_range=0.2, height_shift_range=0.2, horizontal_flip=True, vertical_flip=True)

multi_class_train_gen = train_DataGenerator_multi_class.flow_from_dataframe(
        dataframe= x_multi_class_train.astype(str),
        directory=trainImgPath,
        x_col="ImageId",
        y_col=['has_defect_1','has_defect_2','has_defect_3','has_defect_4'],
        target_size=(HEIGHT,WIDTH),
        batch_size=16,
        class_mode='other')

test_DataGenerator_multi = ImageDataGenerator(rescale=1./255)
multi_class_val_gen = test_DataGenerator_multi.flow_from_dataframe(
        dataframe=x_multi_class_val.astype(str),
        directory=trainImgPath,
        x_col="ImageId",
        y_col=['has_defect_1','has_defect_2','has_defect_3','has_defect_4'],
        target_size=(HEIGHT,WIDTH),
        batch_size=16,
        class_mode='other')


multi_class_model = keras.applications.xception.Xception(include_top = False, input_shape = (256,512,3))

# add a global spatial average pooling layer
x = multi_class_model.output
x = GlobalAveragePooling2D()(x)

# let's add fully-connected layers
x = Dense(512, activation='relu')(x)
x = BatchNormalization()(x)
x = Dropout(0.3)(x)

x = Dense(256, activation='relu')(x)
x = BatchNormalization()(x)
x = Dropout(0.3)(x)

x = Dense(64, activation='relu')(x)

# and the prediction layer
predictions = Dense(4, activation='sigmoid')(x)

# this is the model we will train
model = Model(inputs=multi_class_model.input, outputs=predictions)
model.summary()

model.compile(optimizer='adam', loss='binary_crossentropy',metrics=['accuracy'])
history = model.fit_generator(train_generator, validation_data = validation_generator, epochs = 15, verbose=1)


train_data_1 = x_train[x_train['has_defect_1']==1][['ImageId','Defect_1']]
train_data_2 = x_train[x_train['has_defect_2']==1][['ImageId','Defect_2']]
train_data_3 = x_train[x_train['has_defect_3']==1][['ImageId','Defect_3']]
train_data_4 = x_train[x_train['has_defect_4']==1][['ImageId','Defect_4']]

val_data_1 = x_val[x_val['has_defect_1']==1][['ImageId','Defect_1']]
val_data_2 = x_val[x_val['has_defect_2']==1][['ImageId','Defect_2']]
val_data_3 = x_val[x_val['has_defect_3']==1][['ImageId','Defect_3']]
val_data_4 = x_val[x_val['has_defect_4']==1][['ImageId','Defect_4']]

test_data_1 = x_test[x_test['has_defect_1']==1][['ImageId','Defect_1']]
test_data_2 = x_test[x_test['has_defect_2']==1][['ImageId','Defect_2']]
test_data_3 = x_test[x_test['has_defect_3']==1][['ImageId','Defect_3']]
test_data_4 = x_test[x_test['has_defect_4']==1][['ImageId','Defect_4']]

train_data_1.columns = train_data_2.columns = train_data_3.columns = train_data_4.columns = ['ImageId','EncodedPixels']
val_data_1.columns = val_data_2.columns = val_data_3.columns = val_data_4.columns = ['ImageId','EncodedPixels']
test_data_1.columns = test_data_2.columns = test_data_3.columns = test_data_4.columns = ['ImageId','EncodedPixels']

import keras

from keras.models import Model, load_model
import tensorflow as tf
from tensorflow.python.keras.callbacks import TensorBoard
from keras.callbacks import ModelCheckpoint

from sklearn.metrics import recall_score
from random import random
from random import seed

# https://github.com/qubvel/segmentation_models
import segmentation_models 

import segmentation_models as sm
from segmentation_models import Unet
from segmentation_models import get_preprocessing

network = 'resnet34'
process_input = get_preprocessing(network)
x_train = process_input(x_train)
model = Unet(network,input_shape = (WIDTH, HEIGHT, 3),classes=4,activation='sigmoid')
model.compile('adam', loss='binary_crossentropy',metrics=[dice_coef])

!pip uninstall tf-nightly

import tensorflow as tf

train_data_1

 
from keras.preprocessing.image import ImageDataGenerator
from keras import backend as K
from keras.layers import GlobalAveragePooling2D, Dense, Conv2D, BatchNormalization, Dropout
from keras.models import Model, load_model
import tensorflow as tf
from tensorflow.python.keras.callbacks import TensorBoard
from keras.callbacks import ModelCheckpoint
 

# https://github.com/qubvel/segmentation_models
import segmentation_models
print(segmentation_models.__version__)

import segmentation_models as sm
from segmentation_models import Unet
from segmentation_models import get_preprocessing

from tensorflow.keras.utils import plot_model
network = 'resnet34'
preprocess = get_preprocessing(network) 
train_data_1 = preprocess(train_data_1)

model = Unet(network ,input_shape = (WIDTH, HEIGHT, 3), classes=1, activation='sigmoid') 
model.summary()

pip install tensorflow==2.1.0


!pip install tf-nightly

!pip uninstall tf-nightly 


from collections import OrderedDict
from lasagne.layers import (InputLayer, ConcatLayer, Pool2DLayer, ReshapeLayer, DimshuffleLayer, NonlinearityLayer,
                            DropoutLayer, Deconv2DLayer, batch_norm)
try:
    from lasagne.layers.dnn import Conv2DDNNLayer as ConvLayer
except ImportError:
    from lasagne.layers import Conv2DLayer as ConvLayer
import lasagne
from lasagne.init import HeNormal 

def build_UNet(n_input_channels=1, BATCH_SIZE=None, num_output_classes=2, pad='same', nonlinearity=lasagne.nonlinearities.elu, input_dim=(None, None), base_n_filters=64, do_dropout=False):
    net = OrderedDict()
    net['input'] = InputLayer((BATCH_SIZE, n_input_channels, input_dim[0], input_dim[1]))

    net['contr_1_1'] = batch_norm(ConvLayer(net['input'], base_n_filters, 3, nonlinearity=nonlinearity, pad=pad, W=HeNormal(gain="relu")))
    net['contr_1_2'] = batch_norm(ConvLayer(net['contr_1_1'], base_n_filters, 3, nonlinearity=nonlinearity, pad=pad, W=HeNormal(gain="relu")))
    net['pool1'] = Pool2DLayer(net['contr_1_2'], 2)

    net['contr_2_1'] = batch_norm(ConvLayer(net['pool1'], base_n_filters*2, 3, nonlinearity=nonlinearity, pad=pad, W=HeNormal(gain="relu")))
    net['contr_2_2'] = batch_norm(ConvLayer(net['contr_2_1'], base_n_filters*2, 3, nonlinearity=nonlinearity, pad=pad, W=HeNormal(gain="relu")))
    net['pool2'] = Pool2DLayer(net['contr_2_2'], 2)

    net['contr_3_1'] = batch_norm(ConvLayer(net['pool2'], base_n_filters*4, 3, nonlinearity=nonlinearity, pad=pad, W=HeNormal(gain="relu")))
    net['contr_3_2'] = batch_norm(ConvLayer(net['contr_3_1'], base_n_filters*4, 3, nonlinearity=nonlinearity, pad=pad, W=HeNormal(gain="relu")))
    net['pool3'] = Pool2DLayer(net['contr_3_2'], 2)

    net['contr_4_1'] = batch_norm(ConvLayer(net['pool3'], base_n_filters*8, 3, nonlinearity=nonlinearity, pad=pad, W=HeNormal(gain="relu")))
    net['contr_4_2'] = batch_norm(ConvLayer(net['contr_4_1'], base_n_filters*8, 3, nonlinearity=nonlinearity, pad=pad, W=HeNormal(gain="relu")))
    l = net['pool4'] = Pool2DLayer(net['contr_4_2'], 2)
    
    # the paper does not really describe where and how dropout is added. Feel free to try more options
    if do_dropout:
        l = DropoutLayer(l, p=0.4)

    net['encode_1'] = batch_norm(ConvLayer(l, base_n_filters*16, 3, nonlinearity=nonlinearity, pad=pad, W=HeNormal(gain="relu")))
    net['encode_2'] = batch_norm(ConvLayer(net['encode_1'], base_n_filters*16, 3, nonlinearity=nonlinearity, pad=pad, W=HeNormal(gain="relu")))
    net['upscale1'] = batch_norm(Deconv2DLayer(net['encode_2'], base_n_filters*16, 2, 2, crop="valid", nonlinearity=nonlinearity, W=HeNormal(gain="relu")))
    net['concat1'] = ConcatLayer([net['upscale1'], net['contr_4_2']], cropping=(None, None, "center", "center"))
    net['expand_1_1'] = batch_norm(ConvLayer(net['concat1'], base_n_filters*8, 3, nonlinearity=nonlinearity, pad=pad, W=HeNormal(gain="relu")))
    net['expand_1_2'] = batch_norm(ConvLayer(net['expand_1_1'], base_n_filters*8, 3, nonlinearity=nonlinearity, pad=pad, W=HeNormal(gain="relu")))

    net['upscale2'] = batch_norm(Deconv2DLayer(net['expand_1_2'], base_n_filters*8, 2, 2, crop="valid", nonlinearity=nonlinearity, W=HeNormal(gain="relu")))
    net['concat2'] = ConcatLayer([net['upscale2'], net['contr_3_2']], cropping=(None, None, "center", "center"))
    net['expand_2_1'] = batch_norm(ConvLayer(net['concat2'], base_n_filters*4, 3, nonlinearity=nonlinearity, pad=pad, W=HeNormal(gain="relu")))
    net['expand_2_2'] = batch_norm(ConvLayer(net['expand_2_1'], base_n_filters*4, 3, nonlinearity=nonlinearity, pad=pad, W=HeNormal(gain="relu")))

    net['upscale3'] = batch_norm(Deconv2DLayer(net['expand_2_2'], base_n_filters*4, 2, 2, crop="valid", nonlinearity=nonlinearity, W=HeNormal(gain="relu")))
    net['concat3'] = ConcatLayer([net['upscale3'], net['contr_2_2']], cropping=(None, None, "center", "center"))
    net['expand_3_1'] = batch_norm(ConvLayer(net['concat3'], base_n_filters*2, 3, nonlinearity=nonlinearity, pad=pad, W=HeNormal(gain="relu")))
    net['expand_3_2'] = batch_norm(ConvLayer(net['expand_3_1'], base_n_filters*2, 3, nonlinearity=nonlinearity, pad=pad, W=HeNormal(gain="relu")))

    net['upscale4'] = batch_norm(Deconv2DLayer(net['expand_3_2'], base_n_filters*2, 2, 2, crop="valid", nonlinearity=nonlinearity, W=HeNormal(gain="relu")))
    net['concat4'] = ConcatLayer([net['upscale4'], net['contr_1_2']], cropping=(None, None, "center", "center"))
    net['expand_4_1'] = batch_norm(ConvLayer(net['concat4'], base_n_filters, 3, nonlinearity=nonlinearity, pad=pad, W=HeNormal(gain="relu")))
    net['expand_4_2'] = batch_norm(ConvLayer(net['expand_4_1'], base_n_filters, 3, nonlinearity=nonlinearity, pad=pad, W=HeNormal(gain="relu")))
    net['output_segmentation'] = ConvLayer(net['expand_4_2'], num_output_classes, 1, nonlinearity=None)
    net['dimshuffle'] = DimshuffleLayer(net['output_segmentation'], (1, 0, 2, 3))
    net['reshapeSeg'] = ReshapeLayer(net['dimshuffle'], (num_output_classes, -1))
    net['dimshuffle2'] = DimshuffleLayer(net['reshapeSeg'], (1, 0))
    net['output_flattened'] = NonlinearityLayer(net['dimshuffle2'], nonlinearity=lasagne.nonlinearities.softmax)

    return net

model = build_UNet(n_input_channels=3,input_dim=(WIDTH, HEIGHT)) 

