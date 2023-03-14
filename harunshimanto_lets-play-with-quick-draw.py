#!/usr/bin/env python
# coding: utf-8



get_ipython().run_cell_magic('html', '', "<style>\n@import url('https://fonts.googleapis.com/css?family=Ewert|Roboto&effect=3d|ice|');\nbody {background-color: gainsboro;} \na {color: #37c9e1; font-family: 'Roboto';} \nh1 {color: #37c9e1; font-family: 'Orbitron'; text-shadow: 4px 4px 4px #aaa;} \nh2, h3 {color: slategray; font-family: 'Orbitron'; text-shadow: 4px 4px 4px #aaa;}\nh4 {color: #818286; font-family: 'Roboto';}\nspan {font-family:'Roboto'; color:black; text-shadow: 5px 5px 5px #aaa;}  \ndiv.output_area pre{font-family:'Roboto'; font-size:110%; color:lightblue;}      \n</style>")




import numpy as np
import pandas as pd

get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC, LinearSVC
from sklearn.linear_model import SGDClassifier
from sklearn.neural_network import MLPClassifier

from sklearn.metrics import accuracy_score, confusion_matrix, roc_curve, roc_auc_score
from sklearn.model_selection import train_test_split, GridSearchCV
#deep lerning libraries
import keras
from keras import backend as K
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from IPython.display import SVG
from keras.utils.vis_utils import model_to_dot

import pickle # Read/Write with Serialization
import requests # Makes HTTP requests
from io import BytesIO # Use When expecting bytes-like objects




# Classes we will load
categories = ['cannon','eye', 'face', 'nail', 'pear','piano','radio','spider','star','sword']

# Dictionary for URL and class labels
URL_DATA = {}
for category in categories:
    URL_DATA[category] = 'https://storage.googleapis.com/quickdraw_dataset/full/numpy_bitmap/' + category +'.npy'




classes_dict = {}
for key, value in URL_DATA.items():
    response = requests.get(value)
    classes_dict[key] = np.load(BytesIO(response.content))




for i, (key, value) in enumerate(classes_dict.items()):
    value = value.astype('float32')/255.
    if i == 0:
        classes_dict[key] = np.c_[value, np.zeros(len(value))]
    else:
        classes_dict[key] = np.c_[value,i*np.ones(len(value))]

# Create a dict with label codes
label_dict = {0:'cannon',1:'eye', 2:'face', 3:'nail', 4:'pear', 
              5:'piana',6:'radio', 7:'spider', 8:'star', 9:'sword'}




lst = []
for key, value in classes_dict.items():
    lst.append(value[:3000])
doodles = np.concatenate(lst)




# Split the data into features and class labels (X & y respectively)
y = doodles[:,-1].astype('float32')
X = doodles[:,:784]

# Split each dataset into train/test splits
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.3,random_state=1)




# Save X_train dataset as a pickle file
with open('xtrain_doodle.pickle', 'wb') as f:
    pickle.dump(X_train, f)
    
# Save X_test dataset as a pickle file
with open('xtest_doodle.pickle', 'wb') as f:
    pickle.dump(X_test, f)
    
# Save y_train dataset as a pickle file
with open('ytrain_doodle.pickle', 'wb') as f:
    pickle.dump(y_train, f)
    
# Save y_test dataset as a pickle file
with open('ytest_doodle.pickle', 'wb') as f:
    pickle.dump(y_test, f)

