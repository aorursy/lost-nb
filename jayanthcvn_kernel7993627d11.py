#!/usr/bin/env python
# coding: utf-8

# In[ ]:


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
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


# In[ ]:


from sklearn.model_selection import train_test_split
train_df = pd.read_csv('/kaggle/input/facial-keypoints-detection/training.zip')
test_df = pd.read_csv('/kaggle/input/facial-keypoints-detection/test.zip')
lookid_data = pd.read_csv('/kaggle/input/facial-keypoints-detection/IdLookupTable.csv')
lookid_data


# In[ ]:


train_columns = train_df.columns[:-1].values
train_df.head().T


# In[ ]:


print(train_df.shape)
train_df.isnull().sum()


# In[ ]:


whisker_width = 1.5
rows = train_df.shape[0]
missing_col = 0
for col in train_columns:
    count = train_df[col].count()
    q1 = train_df[col].quantile(0.25)
    q3 = train_df[col].quantile(0.75)
    iqr = q3 - q1
    outliers = train_df[(train_df[col] < q1-whisker_width*iqr)| (train_df[col] > q3+whisker_width*iqr)][col].count()
    print(f'dv:{col}, dv_rows:{count}, missings:{rows-count}%, missing_pct:{round(100*(1-count/rows),2)}%, outliers:{outliers}, outliers_pct:{round(100*(outliers/rows),2)}%')
    if (100*(1-count/rows))>65:
        missing_col += 1
print(f'{missing_col} number of columns have missing values out of {len(train_columns)}')


# In[ ]:


main_features = ['left_eye_center_x', 'left_eye_center_y',
            'right_eye_center_x','right_eye_center_y',
            'nose_tip_x', 'nose_tip_y',
            'mouth_center_bottom_lip_x',
            'mouth_center_bottom_lip_y', 'Image']

train8 = train_df[main_features].dropna().reset_index()
train30 = train_df.dropna().reset_index()
main_features = np.array(main_features)


# In[ ]:


#training = train_df.dropna()
Y30 = train30[train_columns].values
X30 = (np.vstack(train30['Image'].apply(lambda i:np.fromstring(i,sep = ' ')).values)/255.0).astype(np.float32)
X30 = X30.reshape(-1,96,96,1)
x_train30, x_val30, y_train30, y_val30 = train_test_split(X30, Y30, test_size=0.3, random_state=1)
print("30 Features: Train sample:",x_train30.shape,"Val sample:",x_val30.shape)

Y8 = train8[main_features].values
X8 = (np.vstack(train8['Image'].apply(lambda i:np.fromstring(i,sep = ' ')).values)/255.0).astype(np.float32)
X8 = X8.reshape(-1,96,96,1)
x_train8, x_val8, y_train8, y_val8 = train_test_split(X8, Y8, test_size=0.3, random_state=1)
print("8 Features: Train sample:",x_train8.shape,"Val sample:",x_val8.shape)

testing = (np.vstack(test_df['Image'].apply(lambda i:np.fromstring(i,sep = ' ')).values)/255.0).astype(np.float32)
testing = testing.reshape(-1,96,96,1)


# In[ ]:


y_train8 = np.delete(y_train8,8,1)
y_val8 = np.delete(y_val8,8,1)

from keras import backend as K
x_train8 = K.cast_to_floatx(x_train8)
y_train8 = K.cast_to_floatx(y_train8)
x_val8 = K.cast_to_floatx(x_val8)
y_val8 = K.cast_to_floatx(y_val8)

'''y_train30 = (y_train30 - 48) / 48
y_train8 = (y_train8 - 48) / 48
y_val30 = (y_val30 - 48) / 48
y_val8 = (y_val8 - 48) /48'''


# In[ ]:


import tensorflow as tf
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, MaxPool2D
from keras.layers import Conv2D, MaxPooling2D, BatchNormalization, Convolution2D
from keras import backend as K
from keras.regularizers import l2,l1
from keras.layers.advanced_activations import LeakyReLU


# In[ ]:


def get_model(out)
    model = Sequential()
    initializer = tf.keras.initializers.HeNormal()
    #layer = tf.keras.layers.Dense(3, kernel_initializer=initializer)
    model.add(Conv2D(32, kernel_size=(3, 3),kernel_initializer=initializer, padding='same', activation='relu', input_shape=(96, 96, 1)))
    model.add(BatchNormalization())
    model.add(Conv2D(64, (3, 3), padding='same', activation='relu'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(64, kernel_size=(3, 3), padding='same', activation='relu'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(96, (3, 3), padding='same', activation='relu'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(96, kernel_size=(3, 3), padding='same', activation='relu'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(128, (3, 3), padding='same', activation='relu'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(128, kernel_size=(3, 3), padding='same', activation='relu'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(256, (3, 3), padding='same', activation='relu'))
    model.add(BatchNormalization())
    #model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(256, kernel_size=(3, 3), padding='same', activation='relu'))
    model.add(BatchNormalization())
    #model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(512, (3, 3), padding='same', activation='relu'))
    model.add(BatchNormalization())
    #model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(512, kernel_size=(3, 3), padding='same', activation='relu'))
    model.add(BatchNormalization())
    #model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Flatten())
    model.add(Dense(512,activation='relu'))
    model.add(Dropout(0.1))
    model.add(Dense(out))

    return model


# In[ ]:


from keras import backend
 
def rmse(y_true, y_pred):
    return backend.sqrt(backend.mean(backend.square(y_pred - y_true), axis=-1))


# In[ ]:


model8 = get_model(8)
opt = tf.keras.optimizers.Adam(
    learning_rate=0.001,
    beta_1=0.9,
    beta_2=0.999,
    epsilon=1e-08,
)
model8.compile(loss='mean_squared_error',
              optimizer=opt,
              metrics=[rmse,'mse', 'mae'])
model8.summary()


# In[ ]:


model30 = get_model(30)
opt = tf.keras.optimizers.Adam(
    learning_rate=0.001,
    beta_1=0.9,
    beta_2=0.999,
    epsilon=1e-08,
)
model30.compile(loss='mean_squared_error',
              optimizer=opt,
              metrics=[rmse,'mse', 'mae'])
model30.summary()


# In[ ]:


LR_callback_30 = keras.callbacks.ReduceLROnPlateau(monitor='val_loss', patience=4, verbose=10, factor=.4, min_lr=.00001)
EarlyStop_callback_30 = keras.callbacks.EarlyStopping(patience=15, restore_best_weights=True)

LR_callback_8 = keras.callbacks.ReduceLROnPlateau(monitor='val_loss', patience=4, verbose=10, factor=.4, min_lr=.00001)
EarlyStop_callback_8 = keras.callbacks.EarlyStopping(patience=15, restore_best_weights=True)


# In[ ]:


hist30 = model30.fit(x_train30, y_train30,
          batch_size=64,
          epochs=100,
          validation_data=(x_val30, y_val30),
          callbacks=[LR_callback_30,EarlyStop_callback_30]  
                    )


# In[ ]:


# Plot the loss and accuracy curves for training and validation
import matplotlib.pyplot as plt
def plot_loss(history):
    fig, ax = plt.subplots(2,1)
    ax[0].plot(history.history['loss'], color='b', label="Training loss")
    ax[0].plot(history.history['val_loss'], color='r', label="validation loss",axes =ax[0])
    legend = ax[0].legend(loc='best', shadow=True)

    ax[1].plot(history.history['mae'], color='b', label="Training mae")
    ax[1].plot(history.history['val_mae'], color='r',label="Validation mae")
    legend = ax[1].legend(loc='best', shadow=True)


# In[ ]:


plot_loss(hist30)


# In[ ]:


score = model30.evaluate(x_val30, y_val30, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])
score = model30.evaluate(x_train30, y_train30, verbose=0)
print('Train loss:', score[0])
print('Train accuracy:', score[1])


# In[ ]:


y_predict30 = model30.predict(testing)


# In[ ]:


hist8 = model8.fit(x_train8, y_train8,
          batch_size=64,
          epochs=100,
          validation_data=(x_val8, y_val8),
          callbacks=[LR_callback_8,EarlyStop_callback_8]
            )


# In[ ]:


plot_loss(hist8)


# In[ ]:


score = model8.evaluate(x_val8, y_val8, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])
score = model8.evaluate(x_train8, y_train8, verbose=0)
print('Train loss:', score[0])
print('Train accuracy:', score[1])


# In[ ]:


y_predict8 = model8.predict(testing)


# In[ ]:


feature_8_ind = [0, 1, 2, 3, 20, 21, 28, 29]
#Merge 2 prediction from y_hat_30 and y_hat_8.
for i in range(8):
    print('Copy "{}" feature column from y_hat_8 --> y_hat_30'.format(main_features[i]))
    y_predict30[:,feature_8_ind[i]] = y_predict8[:,i]

'''y_predict30 = (y_predict30 * 48) + 48
y_predict8 = (y_predict8 * 48) +48'''


# In[ ]:


#All required features in order.
required_features = list(lookid_data['FeatureName'])
#All images nmber in order.
imageID = list(lookid_data['ImageId']-1)
#Generate Directory to map feature name 'Str' into int from 0 to 29.
feature_to_num = dict(zip(required_features[0:30], range(30)))
feature_ind = []
for f in required_features:
    feature_ind.append(feature_to_num[f])
required_pred = []
for x,y in zip(imageID,feature_ind):
    required_pred.append(y_predict30[x, y])


# In[ ]:


feature_names = list(lookid_data['FeatureName'])
image_ids = list(lookid_data['ImageId']-1)
row_ids = list(lookid_data['RowId'])

feature_list = []
for feature in feature_names:
    feature_list.append(feature_names.index(feature))
    
predictions = []
for x,y in zip(image_ids, feature_list):
    predictions.append(y_predict30[x][y])
    
row_ids = pd.Series(row_ids, name = 'RowId')
locations = pd.Series(predictions, name = 'Location')
locations = locations.clip(0.0,96.0)
submission_result = pd.concat([row_ids,locations],axis = 1)
submission_result.to_csv('2Model_raghu_.csv',index = False)


# In[ ]:


submission_result

