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
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session




get_ipython().run_cell_magic('bash', '', 'cd ../input/pubg-finish-placement-prediction\nls')




get_ipython().run_cell_magic('bash', '', 'pip3 install apache-beam')




import apache_beam as beam

def eval_filter(data):
    eval_split = 0.1
    if(data != None and 'Id' not in data):
        if(abs(hash(data[0])) % 1000000 < (1000000 * eval_split)):
            return True
        else:
            return False
    else:
        return False
    

def train_filter(data):
    eval_split = 0.1
    if(data != None and 'Id' not in data):
        if(abs(hash(data[0])) % 1000000 >= (1000000 * eval_split)):
            return True
        else:
            return False
    else:
        return False
       

cols = ['Id', 'groupId', 'matchId', 'assists', 'boosts', 'damageDealt', 'DBNOs',
       'headshotKills', 'heals', 'killPlace', 'killPoints', 'kills',
       'killStreaks', 'longestKill', 'matchDuration', 'matchType', 'maxPlace',
       'numGroups', 'rankPoints', 'revives', 'rideDistance', 'roadKills',
       'swimDistance', 'teamKills', 'vehicleDestroys', 'walkDistance',
       'weaponsAcquired', 'winPoints', 'winPlacePerc']

class SplitWords(beam.DoFn):
    def __init__(self, delimiter=','):
        self.delimiter = delimiter

    def process(self, text):
        yield text.split(self.delimiter)
        
class ConvertToCsv(beam.DoFn):
    def process(self, text):
        yield ','.join(text)
      

    
with beam.Pipeline() as pipeline:
    out = (pipeline
           | "Read Data" >> beam.io.ReadFromText('../input/pubg-finish-placement-prediction/train_V2.csv')
           | "Split to array" >> beam.ParDo(SplitWords(','))
    )
    
    train = (out
             | "Filter train" >> beam.Filter(train_filter)
             | "convert to array train" >> beam.ParDo(ConvertToCsv())
             | "Write train" >> beam.io.WriteToText(
                     header = ','.join(cols),
                    file_path_prefix = 'train',
                    file_name_suffix = '.csv',
                    shard_name_template = ''
                )
    )
    
    eval = (out
             | "Filter eval" >> beam.Filter(eval_filter)
             | "convert to array eval" >> beam.ParDo(ConvertToCsv())
             | "Write eval" >> beam.io.WriteToText(
                     header = ','.join(cols),
                    file_path_prefix = 'eval',
                    file_name_suffix = '.csv',
                    shard_name_template = ''
                )
    )





import tensorflow as tf




for item in x_train:
    print({item : "{} : {}".format(x_train[item].unique(),len(x_train[item].unique()))})




def gennerate_feature_columns():
    return [
        tf.feature_column.numeric_column('assists'),
        tf.feature_column.numeric_column('boosts'),
        tf.feature_column.numeric_column('damageDealt'),
        #tf.feature_column.bucketized_column(tf.feature_column.numeric_column('DBNOs'),[0,1,2,3,5,10,15,20,25,30]),
        tf.feature_column.numeric_column('headshotKills'),
        tf.feature_column.numeric_column('heals'),
        tf.feature_column.bucketized_column(tf.feature_column.numeric_column('killPlace'),[0,10,20]),
        tf.feature_column.bucketized_column(tf.feature_column.numeric_column('killPoints'),[50,750,1000,1100]),
        tf.feature_column.bucketized_column(tf.feature_column.numeric_column('killStreaks'),[1,3,5,10]),
        #tf.feature_column.bucketized_column(tf.feature_column.numeric_column('longestKill'),[0,20,40,60,80,100,150,200]),
        tf.feature_column.numeric_column('matchDuration'),
        #tf.feature_column.indicator_column(tf.feature_column.categorical_column_with_vocabulary_list('matchType',['squad-fpp','duo-fpp','squad','solo-fpp','duo'])),
        #tf.feature_column.bucketized_column(tf.feature_column.numeric_column('numGroups'),[0,10,20,22,24,26,28,30,40,42,44,46,48,50,60,70,80,82,84,86,88,90,92,94,96,98,100]),
        tf.feature_column.bucketized_column(tf.feature_column.numeric_column('revives'),[0,5,10,15,20,25,30,35,40,45,50]),
        tf.feature_column.bucketized_column(tf.feature_column.numeric_column('walkDistance'),[0,1000,3000]),
        tf.feature_column.bucketized_column(tf.feature_column.numeric_column('weaponsAcquired'),[0,5,10,15,20]),
    ]  




# def model_fn(features, labels, mode):
#     model = tf.keras.Sequential([
#       tf.keras.layers.DenseFeatures(gennerate_feature_columns()),
#       tf.keras.layers.Dense(1,activation = 'relu'),
#       tf.keras.layers.Dense(1,activation = 'softmax')
#     ])
    
#     logits = model(features, training=False)
    
#     if mode == tf.estimator.ModeKeys.PREDICT:
#         predictions = {'logits': logits}
#         return tf.estimator.EstimatorSpec(labels=labels, predictions=predictions)
    
#     optimizer = tf.compat.v1.train.AdamOptimizer()
#     loss = tf.keras.losses.MAE(labels, logits)
    
#     if mode == tf.estimator.ModeKeys.EVAL:
#         return tf.estimator.EstimatorSpec(mode = mode, loss=loss)

#     return tf.estimator.EstimatorSpec(
#           mode=mode,
#           loss=loss,
#           train_op=optimizer.minimize(
#           loss, tf.compat.v1.train.get_or_create_global_step()))




train_batch_size = 100
eval_batch_size = 10

train_dataset = tf.data.experimental.make_csv_dataset(
    ['train.csv'],
    train_batch_size,
    label_name='winPlacePerc',
    num_epochs=3)

eval_dataset = tf.data.experimental.make_csv_dataset(
    ['eval.csv'],
    train_batch_size,
    label_name='winPlacePerc',
    num_epochs=3)




# loss_object = tf.keras.losses.MeanAbsoluteError()

# import time

# def loss(model, x, y, training):
#     y_ = model(x, training=training)
#     return loss_object(y_true=y, y_pred=y_)

# def grad(model, inputs, targets):
#     with tf.GradientTape() as tape:
#         loss_value = loss(model, inputs, targets, training=True)
#     return loss_value, tape.gradient(loss_value, model.trainable_variables)

# optimizer = tf.keras.optimizers.Adam(learning_rate=0.1)

# model = tf.keras.Sequential([
#       tf.keras.layers.DenseFeatures(gennerate_feature_columns()),
#       tf.keras.layers.Dense(100,activation = 'relu'),
#       tf.keras.layers.Dense(10,activation = 'relu'),
#       tf.keras.layers.Dense(1,activation = 'softmax')
#     ])

# train_loss_results = []
# train_accuracy_results = []

# num_epochs = 1
# start_time = 0
# counnt = 1
# for epoch in range(num_epochs):
#     epoch_loss_avg = tf.keras.metrics.Mean()
#     epoch_accuracy = tf.keras.metrics.SparseCategoricalAccuracy()
#     start_time = time.time()
#     print("Epoch:- ",counnt)
#     for x, y in train_dataset:
#         loss_value, grads = grad(model, x, y)
#         optimizer.apply_gradients(zip(grads, model.trainable_variables))
#         # Track progress
#         epoch_loss_avg.update_state(loss_value)  # Add current batch loss
#         # Compare predicted label to actual label
#         # training=True is needed only if there are layers with different
#         # behavior during training versus inference (e.g. Dropout).
#         epoch_accuracy.update_state(tf.reshape(y,(100,1)), model(x, training=True))
#         print(counnt)
#         counnt+=1
#      # End epoch
    
#     train_loss_results.append(epoch_loss_avg.result())
#     train_accuracy_results.append(epoch_accuracy.result())
#     print("Epoch {:03d}: Loss: {:.3f}, Accuracy: {:.3%}".format(epoch,
#                                                                 epoch_loss_avg.result(),
#                                                                 epoch_accuracy.result()))




next(iter(train_dataset))[1]




model = tf.keras.Sequential([
      tf.keras.layers.DenseFeatures(gennerate_feature_columns()),
      tf.keras.layers.Dense(2048,activation = tf.keras.activations.relu),
     tf.keras.layers.Dense(1024,activation = tf.keras.activations.relu),
    tf.keras.layers.Dropout(0.2),
     tf.keras.layers.Dense(2048,activation = tf.keras.activations.relu),
     tf.keras.layers.Dense(1024,activation = tf.keras.activations.relu),
    tf.keras.layers.Dropout(0.2),
     tf.keras.layers.Dense(2048,activation = tf.keras.activations.relu),
     tf.keras.layers.Dense(1024,activation = tf.keras.activations.relu),
    tf.keras.layers.Dropout(0.2),
     tf.keras.layers.Dense(2048,activation = tf.keras.activations.relu),
     tf.keras.layers.Dense(1024,activation = tf.keras.activations.relu),
    tf.keras.layers.Dropout(0.2),
     tf.keras.layers.Dense(2048,activation = tf.keras.activations.relu),
     tf.keras.layers.Dense(1024,activation = tf.keras.activations.relu),
    tf.keras.layers.Dense(1,activation = 'softmax')
    ])

model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate = 0.01), loss = tf.keras.losses.mean_squared_error, metrics=["acc"])
model.fit(train_dataset ,epochs=3,verbose = 1,validation_data = eval_dataset,workers=-1,batch_size = 100)

