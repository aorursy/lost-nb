#!/usr/bin/env python
# coding: utf-8



# System
import sys
import os
import argparse

# Time
import time
import datetime

# Numerical Data
import random
import numpy as np 
import pandas as pd

# Tools
import shutil
from glob import glob
from tqdm import tqdm
import gc

# NLP
import re

# Preprocessing
from sklearn import preprocessing
from sklearn.utils import class_weight as cw
from sklearn.utils import shuffle

# Model Selection
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import cross_val_score

# Machine Learning Models
from sklearn import svm
from sklearn.svm import LinearSVC, SVC

# Evaluation Metrics
from sklearn import metrics 
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score, confusion_matrix, classification_report, roc_auc_score


# Deep Learning - Keras -  Preprocessing
from keras.preprocessing.image import ImageDataGenerator

# Deep Learning - Keras - Model
import keras
from keras import models
from keras.models import Model
from keras.models import load_model
from keras.models import Sequential

# Deep Learning - Keras - Layers
from keras.layers import Convolution1D, concatenate, SpatialDropout1D, GlobalMaxPool1D, GlobalAvgPool1D, Embedding,     Conv2D, SeparableConv1D, Add, BatchNormalization, Activation, GlobalAveragePooling2D, LeakyReLU, Flatten
from keras.layers import Dense, Input, Dropout, MaxPool2D, MaxPooling2D, Concatenate, GlobalMaxPooling2D, GlobalAveragePooling2D,     Lambda, Multiply, LSTM, Bidirectional, PReLU, MaxPooling1D
from keras.layers.pooling import _GlobalPooling1D

from keras.regularizers import l2

# Deep Learning - Keras - Pretrained Models
from keras.applications.xception import Xception
from keras.applications.resnet50 import ResNet50
from keras.applications.inception_v3 import InceptionV3
from keras.applications.inception_resnet_v2 import InceptionResNetV2
from keras.applications.densenet import DenseNet201
from keras.applications.nasnet import NASNetMobile, NASNetLarge

from keras.applications.nasnet import preprocess_input

# Deep Learning - Keras - Model Parameters and Evaluation Metrics
from keras import optimizers
from keras.optimizers import Adam, SGD , RMSprop
from keras.losses import mae, sparse_categorical_crossentropy, binary_crossentropy

# Deep Learning - Keras - Visualisation
from keras.callbacks import ModelCheckpoint, EarlyStopping, TensorBoard, ReduceLROnPlateau, LearningRateScheduler
# from keras.wrappers.scikit_learn import KerasClassifier
from keras import backend as K

# Deep Learning - TensorFlow
import tensorflow as tf

# Graph/ Visualization
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure
import matplotlib.image as mpimg
import seaborn as sns
from mlxtend.plotting import plot_confusion_matrix

# Image
import cv2
from PIL import Image
from IPython.display import display

# np.random.seed(42)

get_ipython().run_line_magic('matplotlib', 'inline')

# Input data
print(os.listdir("../input/"))




train = pd.read_csv("../input/ashrae-energy-prediction/train.csv")
building_metadata = pd.read_csv("../input/ashrae-energy-prediction/building_metadata.csv")
weather_train = pd.read_csv("../input/ashrae-energy-prediction/weather_train.csv")

sample_submission = pd.read_csv("../input/ashrae-energy-prediction/sample_submission.csv")
test = pd.read_csv("../input/ashrae-energy-prediction/test.csv")
weather_test = pd.read_csv("../input/ashrae-energy-prediction/weather_test.csv")




train.head()




weather_train.head()




building_metadata.head()




train_building_metadata = pd.merge(train, building_metadata, on="building_id")
train_building_metadata.head()




train_building_metadata_weather = pd.merge(train_building_metadata, weather_train, on=["site_id", "timestamp"])
train_building_metadata_weather.head()




cols = train_building_metadata_weather.columns.tolist()
cols = [cols[4]] + [cols[0]] + [cols[2]] + [cols[1]] + cols[5:] + [cols[3]] 

train_building_metadata_weather = train_building_metadata_weather[cols]




train_building_metadata_weather.head()




col = "meter"

figure(num=None, figsize=(18, 6), dpi=80, facecolor='w', edgecolor='k')
sns.countplot(x=col, data=train_building_metadata_weather)

plt.title(re.sub("_", " ", col).title())




col = "site_id"

figure(num=None, figsize=(18, 6), dpi=80, facecolor='w', edgecolor='k')
sns.countplot(x=col, data=train_building_metadata_weather)

plt.title(re.sub("_", " ", col).title())




col = "primary_use"

figure(num=None, figsize=(18, 6), dpi=80, facecolor='w', edgecolor='k')
sns.countplot(x=col, data=train_building_metadata_weather)

plt.title(re.sub("_", " ", col).title())
plt.xticks(rotation=90)




col = "year_built"

figure(num=None, figsize=(18, 25), dpi=80, facecolor='w', edgecolor='k')
sns.countplot(y=col, data=train_building_metadata_weather)
plt.title(re.sub("_", " ", col).title())




col = "floor_count"

figure(num=None, figsize=(18, 6), dpi=80, facecolor='w', edgecolor='k')
sns.countplot(x=col, data=train_building_metadata_weather)
plt.title(re.sub("_", " ", col).title())
plt.xticks(rotation=90)




col = "air_temperature"

data = train_building_metadata_weather[col]

data = data[~np.isnan(data)] 

figure(num=None, figsize=(18, 6), dpi=80, facecolor='w', edgecolor='k')
sns.distplot(data)
plt.title(re.sub("_", " ", col).title())




figure(num=None, figsize=(18, 6), dpi=80, facecolor='w', edgecolor='k')
sns.countplot(x="cloud_coverage", data=train_building_metadata_weather)
plt.title(re.sub("_", " ", col).title())




col = "dew_temperature"

data = train_building_metadata_weather[col]

data = data[~np.isnan(data)] 

figure(num=None, figsize=(18, 6), dpi=80, facecolor='w', edgecolor='k')
sns.distplot(data)
plt.title(re.sub("_", " ", col).title())




col = "precip_depth_1_hr"

data = train_building_metadata_weather[col]

data = data[~np.isnan(data)] 

figure(num=None, figsize=(18, 6), dpi=80, facecolor='w', edgecolor='k')
sns.distplot(data)
plt.title(re.sub("_", " ", col).title())




col = "sea_level_pressure"

data = train_building_metadata_weather[col]

data = data[~np.isnan(data)] 

figure(num=None, figsize=(18, 6), dpi=80, facecolor='w', edgecolor='k')
sns.distplot(data)




col = "wind_direction"

data = train_building_metadata_weather[col]

data = data[~np.isnan(data)] 

figure(num=None, figsize=(18, 6), dpi=80, facecolor='w', edgecolor='k')
sns.distplot(data)
plt.title(re.sub("_", " ", col).title())




col = "wind_speed"

data = train_building_metadata_weather[col]

data = data[~np.isnan(data)] 

figure(num=None, figsize=(18, 6), dpi=80, facecolor='w', edgecolor='k')
sns.distplot(data)
plt.title(re.sub("_", " ", col).title())




train_building_metadata_weather.info()






