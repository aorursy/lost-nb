#!/usr/bin/env python
# coding: utf-8



# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.a




import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns 
sns.set()
from sklearn import preprocessing
from sklearn.preprocessing import OneHotEncoder

import warnings
warnings.filterwarnings('ignore')




train = pd.read_csv('../input/train_V2.csv')
test = pd.read_csv('../input/test_V2.csv')




train.head()





# Remove the row with the missing target value
train = train[train['winPlacePerc'].isna() != True]





# Add a feature containing the number of players that joined each match.
train['playersJoined'] = train.groupby('matchId')['matchId'].transform('count')




train['killsNorm'] = train['kills']*((100-train['playersJoined'])/100 + 1)
train['headshotKillsNorm'] = train['headshotKills']*((100-train['playersJoined'])/100 + 1)
train['killPlaceNorm'] = train['killPlace']*((100-train['playersJoined'])/100 + 1)
train['killPointsNorm'] = train['killPoints']*((100-train['playersJoined'])/100 + 1)
train['killStreaksNorm'] = train['killStreaks']*((100-train['playersJoined'])/100 + 1)
train['longestKillNorm'] = train['longestKill']*((100-train['playersJoined'])/100 + 1)
train['roadKillsNorm'] = train['roadKills']*((100-train['playersJoined'])/100 + 1)
train['teamKillsNorm'] = train['teamKills']*((100-train['playersJoined'])/100 + 1)
train['damageDealtNorm'] = train['damageDealt']*((100-train['playersJoined'])/100 + 1)
train['DBNOsNorm'] = train['DBNOs']*((100-train['playersJoined'])/100 + 1)
train['revivesNorm'] = train['revives']*((100-train['playersJoined'])/100 + 1




# Features to remove
train = train.drop([ 'kills', 'headshotKills', 'killPlace', 'killPoints', 'killStreaks', 
 'longestKill', 'roadKills', 'teamKills', 'damageDealt', 'DBNOs', 'revives'],axis=1)




train.head()




# Total distance travelled
train['totalDistance'] = train['walkDistance'] + train['rideDistance'] + train['swimDistance']




def standardize_matchType(data):
    data['matchType'][data['matchType'] == 'normal-solo'] = 'Solo'
    data['matchType'][data['matchType'] == 'solo-fpp'] = 'Solo'
    data['matchType'][data['matchType'] == 'normal-solo-fpp'] = 'Solo'
    data['matchType'][data['matchType'] == 'normal-duo-fpp'] = 'Duo'
    data['matchType'][data['matchType'] == 'normal-duo'] = 'Duo'
    data['matchType'][data['matchType'] == 'duo-fpp'] = 'Duo'
    data['matchType'][data['matchType'] == 'squad-fpp'] = 'Squad'
    data['matchType'][data['matchType'] == 'normal-squad'] = 'Squad'
    data['matchType'][data['matchType'] == 'normal-squad-fpp'] = 'Squad'
    data['matchType'][data['matchType'] == 'flaretpp'] = 'Other'
    data['matchType'][data['matchType'] == 'flarefpp'] = 'Other'
    data['matchType'][data['matchType'] == 'crashtpp'] = 'Other'
    data['matchType'][data['matchType'] == 'crashfpp'] = 'Other'

    return data


data = standardize_matchType(train)
#print (set(data['matchType']))




data.head()




solo = data[data['matchType'] == 'Solo']
duo = data[data['matchType'] == 'Duo']
squad = data[data['matchType'] == 'Squad']
other = data[data['matchType'] == 'Other']




solo_features = ['boosts','heals', 'rideDistance','walkDistance','weaponsAcquired',
                 # Engineered Features
                 'damageDealtNorm','headshotKillsNorm','killPlaceNorm',
                 'killsNorm','killStreaksNorm','longestKillNorm',
                 'playersJoined','totalDistance']

solo = solo[solo_features]
solo.head()




f,ax = plt.subplots(figsize=(15, 15))
sns.heatmap(solo.corr(), annot=True, linewidths=.5, fmt= '.2f',ax=ax)
plt.show()




duo_features = ['assists','boosts', 'heals','rideDistance','walkDistance',
                'weaponsAcquired',
                # Engineered Features
                'damageDealtNorm','DBNOsNorm', 'killPlaceNorm',
                'killsNorm','killStreaksNorm','longestKillNorm',
                'revivesNorm', 'playersJoined','totalDistance']

duo = duo[duo_features]
duo.head()




f,ax = plt.subplots(figsize=(15, 15))
sns.heatmap(duo.corr(), annot=True, linewidths=.5, fmt= '.2f',ax=ax)
plt.show()




squad_features = ['assists','boosts','heals','rideDistance',
                  'walkDistance','weaponsAcquired',
                  # Engineered Features
                  'damageDealtNorm','DBNOsNorm', 'killPlaceNorm',
                  'killsNorm','killStreaksNorm','longestKillNorm',
                  'revivesNorm','playersJoined','totalDistance']

squad = squad[squad_features]
squad.head()




f,ax = plt.subplots(figsize=(15, 15))
sns.heatmap(squad.corr(), annot=True, linewidths=.5, fmt= '.2f',ax=ax)
plt.show()




other_features = ['assists','boosts','heals','rideDistance',
                  'walkDistance','weaponsAcquired',
                  # Engineered Features
                  'damageDealtNorm','DBNOsNorm','headshotKillsNorm',
                  'killPlaceNorm','killsNorm','killStreaksNorm','longestKillNorm',
                  'revivesNorm','playersJoined','totalDistance']

other = other[other_features]
other.head()




f,ax = plt.subplots(figsize=(15, 15))
sns.heatmap(other.corr(), annot=True, linewidths=.5, fmt= '.2f',ax=ax)
plt.show()




import math
from sklearn import metrics
import tensorflow as tf
from tensorflow.python.data import Dataset
import matplotlib.pyplot as plt
import seaborn as sns 
import warnings
warnings.filterwarnings("ignore")




df0 = pd.read_csv("../input/train_V2.csv")
df = df0.head(20000)
df.info()




data = df.copy()
data = data[data['walkDistance'] < df['walkDistance'].quantile(0.99)]
plt.figure(figsize=(15,10))
plt.title("Walking Distance Distribution",fontsize=15)
sns.distplot(data['walkDistance'])
plt.show()




# Define the input feature: total_rooms.
my_feature = df[["heals", "kills","killPlace"]]

# Configure a feature column for total_rooms.
feature_columns = [tf.feature_column.numeric_column("heals"),
                   tf.feature_column.numeric_column("kills"),
                   tf.feature_column.numeric_column("killPlace"),
                   tf.feature_column.bucketized_column(
                   source_column = tf.feature_column.numeric_column("heals"), 
                   boundaries = [0, 10, 100]),
                   tf.feature_column.crossed_column(
                   [tf.feature_column.bucketized_column(
                   source_column = tf.feature_column.numeric_column("kills"), 
                   boundaries = [0, 1, 5, 10]),
                   tf.feature_column.bucketized_column(
                   source_column = tf.feature_column.numeric_column("killPlace"), 
                   boundaries = [0, 2, 3, 5, 10])], 
                   hash_bucket_size = 500
                   )
                  ]

# Define the label.
targets = df["winPlacePerc"]




print(data.shape)
data.head




data.maxPlace.plot(kind='hist')




import matplotlib.pyplot as  plt
import seaborn as sns




print('There are {} unique maxPlace.'.format(len(data.maxPlace.unique())))




data_store_by_maxPlace = {}
for x in data.maxPlace.unique():
    data_store_by_maxPlace[x] = data.loc[data.maxPlace==x]




f, ax = plt. subplots(figsize=(15, 15))
sns.heatmap(my_feature.corr(), annot=True, linewidths=.5, ax=ax)
plt.show()





def adjust_pred(x, maxPlace):
    space = 1/(maxPlace-1)
    return np.round(x / space) * space




def generate_lgb_model(data, for_eval):
    # split data into X and y
    X = data.iloc[:,3:33]
    
    maxPlace = X.maxPlace.unique()
    X = X.drop('maxPlace', axis=1)
    Y = data.iloc[:,33]
    
    if for_eval == True:
        seed = 7
        test_size = 0.33
        X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=test_size, random_state=seed)
    else:
        X_train, y_train = X, Y
        
    d_train = lgb.Dataset(X_train, label=y_train)

    params = {}
    params['objective'] = 'regression'
    params['metric'] = 'mae'

    model = lgb.train(params, d_train)
    
    if for_eval == True:
        #Prediction
        y_pred=model.predict(X_test)
        
        y_pred = adjust_pred(y_pred, maxPlace)
        mae = mean_absolute_error(y_test, y_pred)
        print("MAE for maxPlace = {}: {}".format(maxPlace, mae))
    
        return [model, (list(y_test), y_pred)]
    
    else:
        return model




my_optimizer=tf.train.GradientDescentOptimizer(learning_rate=0.001)
my_optimizer = tf.contrib.estimator.clip_gradients_by_norm(my_optimizer, 5.0)

# Configure the linear regression model with our feature columns and optimizer.
# Set a learning rate of 0.0000001 for Gradient Descent.
linear_regressor = tf.estimator.LinearClassifier(
    feature_columns=feature_columns,
    optimizer=my_optimizer
)




def my_input_fn(features, targets, batch_size=1, shuffle=True, num_epochs=None):
    """Trains a linear regression model of one feature.
  
    Args:
      features: pandas DataFrame of features
      targets: pandas DataFrame of targets
      batch_size: Size of batches to be passed to the model
      shuffle: True or False. Whether to shuffle the data.
      num_epochs: Number of epochs for which data should be repeated. None = repeat indefinitely
    Returns:
      Tuple of (features, labels) for next data batch
    """
  
    # Convert pandas data into a dict of np arrays.
    features = {key:np.array(value) for key,value in dict(features).items()}                                           
 
    # Construct a dataset, and configure batching/repeating.
    ds = Dataset.from_tensor_slices((features,targets)) # warning: 2GB limit
    ds = ds.batch(batch_size).repeat(num_epochs)
    
    # Shuffle the data, if specified.
    if shuffle:
      ds = ds.shuffle(buffer_size=10000)
    
    # Return the next batch of data.
    features, labels = ds.make_one_shot_iterator().get_next()
    return features, labels




_ = linear_regressor.train(
    input_fn = lambda:my_input_fn(my_feature, targets),
    steps=5000)




# Create an input function for predictions.
# Note: Since we're making just one prediction for each example, we don't 
# need to repeat or shuffle the data here.
prediction_input_fn =lambda: my_input_fn(my_feature, targets, num_epochs=1, shuffle=False)

# Call predict() on the linear_regressor to make predictions.
predictions = linear_regressor.predict(input_fn=prediction_input_fn)

# Format predictions as a NumPy array, so we can calculate error metrics.
# predictions = np.array([item['predictions'][0] for item in predictions])

predictions_prob = np.array([item['probabilities'][1] for item in predictions])

# Print Mean Squared Error and Root Mean Squared Error.
mean_squared_error = metrics.mean_absolute_error(predictions_prob, targets)
root_mean_squared_error = math.sqrt(mean_squared_error)
print("Mean Squared Error (on training data): %0.3f" % mean_squared_error)
print("Root Mean Squared Error (on training data): %0.3f" % root_mean_squared_error)




predictions_prob[0:15]




df.head(15)




test_df0 = pd.read_csv("../input/test_V2.csv")
#test_df = test_df0.head(1000)
test_df = test_df0.copy()
test_df.info()




predictions_prob[0:15]




df.head(15)




test_df0 = pd.read_csv("../input/test_V2.csv")
#test_df = test_df0.head(1000)
test_df = test_df0.copy()
test_df.info()




pred_my_feature = test_df[["heals", "kills","killPlace"]]
pred_targets = test_df['assists']
prediction_input_fn =lambda: my_input_fn(pred_my_feature, pred_targets, num_epochs=1, shuffle=False)

# Call predict() on the linear_regressor to make predictions.
predictions = linear_regressor.predict(input_fn=prediction_input_fn)

# Format predictions as a NumPy array, so we can calculate error metrics.
# predictions = np.array([item['predictions'][0] for item in predictions])

predictions_prob = np.array([item['probabilities'][1] for item in predictions])

predictions_prob[0:15]




test_id = test_df["Id"]
result = pd.DataFrame({"Id": test_id, "winPlacePerc": predictions_prob})




result




test_df0.head()




result.to_csv("submission.csv", index = False)




print(os.listdir("."))






