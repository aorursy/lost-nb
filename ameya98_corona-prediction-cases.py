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




try:
  # %tensorflow_version only exists in Colab.
  get_ipython().run_line_magic('tensorflow_version', '2.x')
except Exception:
  pass
import tensorflow as tf

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd




tf.__version__




train=pd.read_csv('/kaggle/input/covid19-global-forecasting-week-2/train.csv')




train.head()




train=train[['Date','ConfirmedCases']]




train.head()




# train['Date'] = pd.to_datetime(train['Date'])
train.Date.unique().shape




train=train.groupby(train['Date']).sum()




train.tail()




train.shape




train.plot(subplots=True)




train = train.values




uni_train_mean = train[:].mean()
uni_train_std = train[:].std()




train = (train-uni_train_mean)/uni_train_std




def univariate_data(dataset, start_index, end_index, history_size, target_size):
  data = []
  labels = []

  start_index = start_index + history_size
  if end_index is None:
    end_index = len(dataset) - target_size

  for i in range(start_index, end_index):
    indices = range(i-history_size, i)
    # Reshape data from (history_size,) to (history_size, 1)
    data.append(np.reshape(dataset[indices], (history_size, 1)))
    labels.append(dataset[i+target_size])
  return np.array(data), np.array(labels)




univariate_past_history = 3
univariate_future_target = 0

x_train_uni, y_train_uni = univariate_data(train, 0, None,
                                           univariate_past_history,
                                           univariate_future_target)

x_val_uni, y_val_uni = univariate_data(train, 55, None,
                                       univariate_past_history,
                                       univariate_future_target)









print ('Single window of past history')
print (x_train_uni[-1])
print ('\n Target cases to predict')
print (y_train_uni[-1])




def create_time_steps(length):
  return list(range(-length, 0))




def show_plot(plot_data, delta, title):
  labels = ['History', 'True Future', 'Model Prediction']
  marker = ['.-', 'rx', 'go']
  time_steps = create_time_steps(plot_data[0].shape[0])
  if delta:
    future = delta
  else:
    future = 0

  plt.title(title)
  for i, x in enumerate(plot_data):
    if i:
      plt.plot(future, plot_data[i], marker[i], markersize=10,
               label=labels[i])
    else:
      plt.plot(time_steps, plot_data[i].flatten(), marker[i], label=labels[i])
  plt.legend()
  plt.xlim([time_steps[0], (future+5)*2])
  plt.xlabel('Time-Step')
  return plt




show_plot([x_train_uni[0], y_train_uni[0]], 0, 'Sample Example')




def baseline(history):
  return np.mean(history)




show_plot([x_train_uni[0], y_train_uni[0], baseline(x_train_uni[0])], 0,
           'Baseline Prediction Example')




BATCH_SIZE = 4
BUFFER_SIZE = 10

train_univariate = tf.data.Dataset.from_tensor_slices((x_train_uni, y_train_uni))
train_univariate = train_univariate.cache().shuffle(BUFFER_SIZE).batch(BATCH_SIZE).repeat()

val_univariate = tf.data.Dataset.from_tensor_slices((x_val_uni, y_val_uni))
val_univariate = val_univariate.batch(BATCH_SIZE).repeat()




model_1 = tf.keras.models.Sequential([
    tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(8, input_shape=x_train_uni.shape[-2:],return_sequences=True)),
    tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(8, activation='relu')),
    tf.keras.layers.Dense(1)
])

model_1.compile(optimizer='adam', loss='mae')




get_ipython().system('mkdir cp')




from tensorflow.keras.callbacks import ModelCheckpoint
filepath="cp/model1.hdf5"
checkpoint = ModelCheckpoint(filepath, monitor='val_loss', verbose=1, save_best_only=True, mode='min')




import math
from tensorflow.keras.callbacks import LearningRateScheduler

def step_decay(epoch):
  initial_lrate = 0.001
  drop = 0.5

  epochs_drop = 500.0


  lrate = initial_lrate * math.pow(drop, math.floor((1+epoch)/epochs_drop))

  return lrate



lrate = LearningRateScheduler(step_decay)
callbacks_list = [checkpoint,lrate]




for x, y in val_univariate.take(1):
    print(model_1.predict(x).shape)




EVALUATION_INTERVAL = 10
EPOCHS = 1500

model_1.fit(train_univariate, epochs=EPOCHS,
                      steps_per_epoch=EVALUATION_INTERVAL,
                      validation_data=val_univariate, validation_steps=7,callbacks= callbacks_list)




from tensorflow.keras.models import load_model
model_1=load_model('cp/model1.hdf5')




for x, y in val_univariate.take(3):
  plot = show_plot([x[0].numpy(), y[0].numpy(),
                    model_1.predict(x)[0]], 0, 'Simple LSTM model')
  plot.show()









train=pd.read_csv('/kaggle/input/covid19-global-forecasting-week-2/train.csv')




train.head()




train=train[['Date','ConfirmedCases','Fatalities']]




train.plot(subplots=True)




train=train.groupby(train['Date']).sum()




train.tail()




train.plot(subplots=True)




dataset = train.values
data_mean = dataset[:].mean(axis=0)
data_std = dataset[:].std(axis=0)




dataset = (dataset-data_mean)/data_std




def multivariate_data(dataset, target, start_index, end_index, history_size,
                      target_size, step, single_step=False):
  data = []
  labels = []

  start_index = start_index + history_size
  if end_index is None:
    end_index = len(dataset) - target_size

  for i in range(start_index, end_index):
    indices = range(i-history_size, i, step)
    data.append(dataset[indices])

    if single_step:
      labels.append(target[i+target_size])
    else:
      labels.append(target[i:i+target_size])

  return np.array(data), np.array(labels)




past_history = 3
future_target = 3
STEP = 1

x_train_single, y_train_single = multivariate_data(dataset, dataset[:, 0], 0,
                                                   None, past_history,
                                                   future_target, STEP,
                                                   single_step=True)
x_val_single, y_val_single = multivariate_data(dataset, dataset[:, 0],
                                               55, None, past_history,
                                               future_target, STEP,
                                               single_step=True)




print ('Single window of past history')
print (x_train_single[-1])
print ('\n Target cases to predict')
print (y_train_single[-1])




train_data_single = tf.data.Dataset.from_tensor_slices((x_train_single, y_train_single))
train_data_single = train_data_single.cache().shuffle(BUFFER_SIZE).batch(BATCH_SIZE).repeat()

val_data_single = tf.data.Dataset.from_tensor_slices((x_val_single, y_val_single))
val_data_single = val_data_single.batch(BATCH_SIZE).repeat()




model_2 = tf.keras.models.Sequential([
    tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(8, input_shape=x_train_uni.shape[-2:],return_sequences=True)),
    tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(8, activation='relu')),
    tf.keras.layers.Dense(1)
])




model_2.compile(optimizer='adam', loss='mae')




from tensorflow.keras.callbacks import ModelCheckpoint
filepath="cp/model2.hdf5"
checkpoint = ModelCheckpoint(filepath, monitor='val_loss', verbose=1, save_best_only=True, mode='min')




import math
from tensorflow.keras.callbacks import LearningRateScheduler

def step_decay(epoch):
  initial_lrate = 0.001
  drop = 0.5

  epochs_drop = 500.0


  lrate = initial_lrate * math.pow(drop, math.floor((1+epoch)/epochs_drop))

  return lrate



lrate = LearningRateScheduler(step_decay)
callbacks_list = [checkpoint,lrate]




for x, y in val_data_single.take(1):
  print(model_2.predict(x).shape)




single_step_history = model_2.fit(train_data_single, epochs=EPOCHS,
                                            steps_per_epoch=EVALUATION_INTERVAL,
                                            validation_data=val_data_single,
                                            validation_steps=7,callbacks= callbacks_list)




model_2=load_model('cp/model2.hdf5')




def plot_train_history(history, title):
  loss = history.history['loss']
  val_loss = history.history['val_loss']

  epochs = range(len(loss))

  plt.figure()

  plt.plot(epochs, loss, 'b', label='Training loss')
  plt.plot(epochs, val_loss, 'r', label='Validation loss')
  plt.title(title)
  plt.legend()

  plt.show()




plot_train_history(single_step_history,
                   'Single Step Training and validation loss')




for x, y in val_data_single.take(3):
  plot = show_plot([x[0][:, 0].numpy(), y[0].numpy(),
                    model_2.predict(x)[0]], 0,
                   'Single Step Prediction')
  plot.show()




future_target = 3
x_train_multi, y_train_multi = multivariate_data(dataset, dataset[:, 0], 0,
                                                 None, past_history,
                                                 future_target, STEP)
x_val_multi, y_val_multi = multivariate_data(dataset, dataset[:, 0],
                                             55, None, past_history,
                                             future_target, STEP)




print ('Single window of past history : {}'.format(x_train_multi[0]))
print ('\n Target cases to predict : {}'.format(y_train_multi[0]))




train_data_multi = tf.data.Dataset.from_tensor_slices((x_train_multi, y_train_multi))
train_data_multi = train_data_multi.cache().shuffle(BUFFER_SIZE).batch(BATCH_SIZE).repeat()

val_data_multi = tf.data.Dataset.from_tensor_slices((x_val_multi, y_val_multi))
val_data_multi = val_data_multi.batch(BATCH_SIZE).repeat()




def multi_step_plot(history, true_future, prediction):
  plt.figure(figsize=(12, 6))
  num_in = create_time_steps(len(history))
  num_out = len(true_future)

  plt.plot(num_in, np.array(history[:, 1]), label='History')
  plt.plot(np.arange(num_out)/STEP, np.array(true_future), 'bo',
           label='True Future')
  if prediction.any():
    plt.plot(np.arange(num_out)/STEP, np.array(prediction), 'ro',
             label='Predicted Future')
  plt.legend(loc='upper left')
  plt.show()




for x, y in train_data_multi.take(1):
  multi_step_plot(x[0], y[0], np.array([0]))




model_3 = tf.keras.models.Sequential([
    tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(8, input_shape=x_train_uni.shape[-2:],return_sequences=True)),
    tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(8, activation='relu')),
    tf.keras.layers.Dense(3)
])




model_3.compile(optimizer='adam', loss='mae')




from tensorflow.keras.callbacks import ModelCheckpoint
filepath="cp/model3.hdf5"
checkpoint = ModelCheckpoint(filepath, monitor='val_loss', verbose=1, save_best_only=True, mode='min')




import math
from tensorflow.keras.callbacks import LearningRateScheduler

def step_decay(epoch):
  initial_lrate = 0.001
  drop = 0.5

  epochs_drop = 500.0


  lrate = initial_lrate * math.pow(drop, math.floor((1+epoch)/epochs_drop))

  return lrate



lrate = LearningRateScheduler(step_decay)
callbacks_list = [checkpoint,lrate]









multi_step_history = model_3.fit(train_data_multi, epochs=EPOCHS,
                                            steps_per_epoch=EVALUATION_INTERVAL,
                                            validation_data=val_data_multi,
                                            validation_steps=7,callbacks= callbacks_list)




model_3=load_model('cp/model3.hdf5')




plot_train_history(multi_step_history, 'Multi-Step Training and validation loss')




for x, y in val_data_multi.take(3):
  multi_step_plot(x[0], y[0], model_3.predict(x)[0])






