#!/usr/bin/env python
# coding: utf-8



from IPython.display import Image
Image("../input/covid19-forecast-week-2/COVID-19 forecast - week 2.png")




import pandas as pd
import numpy as np
from datetime import datetime

# Reading data
data = pd.read_csv('../input/covid19-global-forecasting-week-2/train.csv')

# Transforming ConfirmedCases and Fatalities to log scale, to make it easier for 
# metrics evaluation later (which is based on log scale)
data['ConfirmedCases'] = data['ConfirmedCases'].apply(lambda x: np.log(x + 1))
data['Fatalities'] = data['Fatalities'].apply(lambda x: np.log(x + 1))

# Combining Country and Subdivision (i.e. province/state) into one column
# as one single label (for convenience)
data["Country & Subdivision"] = [(data['Country_Region'][i],   data['Province_State'][i]) for i in range(len(data))]
#data = data.drop(columns=['Province/State', 'Country/Region', 'Lat', 'Long'])

# Encoding areas into integers, e.g. Afghanistan --> 0, Albania --> 1
# This "encoding" process is for the "embedding layer" in the model (explained later)
unique_areas = data['Country & Subdivision'].unique()
unique_areas_dict = {j : i for i, j in enumerate(unique_areas)}
NUM_DATES = 100; TRAIN_SIZE = 57;
train = data[data['Id']%NUM_DATES <= TRAIN_SIZE] # this is stupid --> fix later
encoded_train = np.array(list(map(lambda x: unique_areas_dict[x],
                                  train['Country & Subdivision'].to_numpy())))\
                .reshape((-1, 1))

# Building the model
# This model features the use of "embedding layer", which is typically used in
# Natural Language Processing to make word embeddings. In this task, I borrowed
# this idea and attempt to capture information about each geographic area using this
# embedding process
import tensorflow as tf

class MyModel(tf.keras.Model):
  def __init__(self, ):
    super(MyModel, self).__init__()

    self.embedding_size = 10
    self.LSTM_size = 30
    self.w1 = tf.keras.layers.Embedding(len(unique_areas), self.embedding_size,)
    self.w2 = tf.keras.layers.LSTM(self.LSTM_size,
                return_sequences=True,
                stateful=False, )
    self.w3 = tf.keras.layers.LSTM(forecast_length,
                return_sequences=True,
                stateful=False, activation='relu' )
   
  def call(self, inputs):
    location, curr_num = tf.split(inputs, [-1, 1], 2)
    x = self.w1(location)
    x = tf.reshape(x, [-1,inputs.shape[1],self.embedding_size])
    x = tf.concat([x, curr_num], axis=2)
    x = self.w2(x, )
    x = self.w3(x, )
    return x 

forecast_length = 13

### ----- Predicting for Confirmed Cases ----- ###
# Preparing the input, X, by joining the area encoding and the time series values
X = np.concatenate((encoded_train, 
                    train['ConfirmedCases'].to_numpy().reshape((-1, 1))),
                   axis=1)
X = X.reshape( (len(unique_areas), -1, 2 ) )

# Preparing the output, Y. At each timestep, Y has the future values of 
# forecast_length days, which the model trys to predict for
total_length = X.shape[1]
Y = X[:, :, -1]
Y = [ [y[i:i+forecast_length] for i in range(1, total_length-forecast_length+1)]
     for y in Y ]
Y = np.array(Y)

# Shortening the input, X, to exclude the last step's Y values
X = X[:, :-forecast_length, :]

model_1 = MyModel()

model_1.compile(optimizer='adam', loss='mae', )

# Training the model
# A while loop is used to ensure the model is succcessfully trained in the end
# because sometimes the model has "spikes" in loss and needs restarting
while 'history_1' not in vars() or history_1.history['loss'][-1] > 0.1:
  history_1 = model_1.fit(X, Y, epochs=10000, batch_size=len(unique_areas), )
model_1.save_weights('model_1_weights.h5')

# Making inference on the trained model
# Preparing the area encoding for feeding into the model
encoded_test = np.array(list(map(lambda x: unique_areas_dict[x],
                                 data['Country & Subdivision'].to_numpy())))\
                .reshape((-1, 1))
# Joining the encoding with values, similar to preparation of training set
test_X = np.concatenate((encoded_test, 
                         data['ConfirmedCases'].to_numpy().reshape((-1, 1))),
                        axis=1)  
test_X = test_X.reshape( (len(unique_areas), -1, 2 ) )
test_X = test_X[:, :TRAIN_SIZE, :]

subm = pd.read_csv('../input/covid19-global-forecasting-week-2/submission.csv')

# Preparing to collect the results. For loops are used to iteratively query the
# model, since the model can only forecast ~ 14 days, but we need to predict for
# roughly one month. So making 4 iterations of prediction is sufficient. Here, 
# the predicted values of 1st iteration is used as input for 2nd iteration, etc.
results = np.zeros( (X.shape[0], 0 ))
for i in range(4):
  model_1.reset_states()
  test_Y = model_1(test_X).numpy()[:, -1, :]
  results = np.concatenate( (results, np.exp( test_Y ) - 1), axis=1 )
  test_Y = test_Y.reshape( (-1, forecast_length, 1) )
  test_Y = np.concatenate( 
    ([[[i]] * forecast_length for i in range(len(test_Y))], test_Y,), axis = 2)
  test_X = np.concatenate((test_X, test_Y), axis=1)#[:, test_X.shape[1]:, :]

# Filling the results into the submission form dataframe
subm['ConfirmedCases'] = results[:, :NUM_DATES-TRAIN_SIZE].reshape(-1)

### ----- Predicting for Fatalities ----- ###
# The following code is identical to the last part, so comments are not repeated
X = np.concatenate((encoded_train, 
                    train['Fatalities'].to_numpy().reshape((-1, 1))),
                   axis=1)
X = X.reshape( (len(unique_areas), -1, 2 ) )

total_length = X.shape[1]
Y = X[:, :, -1]
Y = [ [y[i:i+forecast_length] for i in range(1, total_length-forecast_length+1)]
     for y in Y ]
Y = np.array(Y)
X = X[:, :-forecast_length, :]

model_2 = MyModel()

model_2.compile(optimizer='adam', loss='mse', )

while 'history_2' not in vars() or history_2.history['loss'][-1] > 0.01:
  history_2 = model_2.fit(X, Y, epochs=10000, batch_size=len(unique_areas), )
model_2.save_weights('model_2_weights.h5')

encoded_test = np.array(list(map(lambda x: unique_areas_dict[x],
                                 data['Country & Subdivision'].to_numpy())))\
                .reshape((-1, 1))
test_X = np.concatenate((encoded_test, 
                         data['Fatalities'].to_numpy().reshape((-1, 1))),
                        axis=1)
test_X = test_X.reshape( (len(unique_areas), -1, 2 ) )
test_X = test_X[:, :TRAIN_SIZE, :]

results = np.zeros( (X.shape[0], 0 ))
for i in range(4):
  model_2.reset_states()
  test_Y = model_2(test_X).numpy()[:, -1, :]
  print(np.exp(test_Y[2])-1)
  results = np.concatenate( (results, np.exp(test_Y) - 1), axis=1 )
  test_Y = test_Y.reshape( (-1, forecast_length, 1) )
  test_Y = np.concatenate( 
    ([[[i]] * forecast_length for i in range(len(test_Y))], test_Y,), axis = 2)
  test_X = np.concatenate((test_X, test_Y), axis=1)#[:, test_X.shape[1]:, :]

subm['Fatalities'] = results[:, :NUM_DATES-TRAIN_SIZE].reshape(-1)

subm.to_csv('submission.csv', index=False)

