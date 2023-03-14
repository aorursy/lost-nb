#!/usr/bin/env python
# coding: utf-8

# In[1]:


# Standard libraries
import os
import numpy as np
import pandas as pd
import random as rn
import tensorflow as tf

# Visualization
get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt

# For calculating the metric
from sklearn.metrics import mean_squared_log_error

# Path specifications
BASE_PATH = "../input/ashrae-energy-prediction/"
TRAIN_PATH = BASE_PATH + "train.csv"
SAMP_SUB_PATH = BASE_PATH + "sample_submission.csv"

# Seed for reproducability
seed = 1234
rn.seed(seed)
np.random.seed(seed)
os.environ['PYTHONHASHSEED'] = str(seed)


# In[2]:


# Read in data
df = pd.read_csv(TRAIN_PATH)
# Remove outliers
df = df[df['meter_reading'] < 250000]


# In[3]:


def RMSLE(y_true:np.ndarray, y_pred:np.ndarray) -> np.float64:
    """
        The Root Mean Squared Log Error (RMSLE) metric 
        
        :param y_true: The ground truth labels given in the dataset
        :param y_pred: Our predictions
        :return: The RMSLE score
    """
    return np.sqrt(mean_squared_log_error(y_true, y_pred))


# In[4]:


def NumPyRMSLE(y_true:list, y_pred:list) -> float:
    """
        The Root Mean Squared Log Error (RMSLE) metric using only NumPy
        N.B. This function is a lot slower than sklearn's implementation
        
        :param y_true: The ground truth labels given in the dataset
        :param y_pred: Our predictions
        :return: The RMSLE score
    """
    n = len(y_true)
    msle = np.mean([(np.log(y_pred[i] + 1) - np.log(y_true[i] + 1)) ** 2.0 for i in range(n)])
    return np.sqrt(msle)


# In[5]:


def RMSLETF(y_pred:tf.Tensor, y_true:tf.Tensor) -> tf.float64:
    '''
        The Root Mean Squared Log Error (RMSLE) metric for TensorFlow / Keras
        
        :param y_true: The ground truth labels given in the dataset
        :param y_pred: Predicted values
        :return: The RMSLE score
    '''
    y_pred = tf.cast(y_pred, tf.float64)
    y_true = tf.cast(y_true, tf.float64) 
    y_pred = tf.nn.relu(y_pred) 
    return tf.sqrt(tf.reduce_mean(tf.squared_difference(tf.log1p(y_pred), tf.log1p(y_true))))


# In[6]:


mean = np.mean(df['meter_reading'])

print(f"RMSLE for predicting only 0: {round(RMSLE(df['meter_reading'], np.zeros(len(df))), 5)}")
print(f"RMSLE for predicting only 1: {round(RMSLE(df['meter_reading'], np.ones(len(df))), 5)}")
print(f"RMSLE for predicting only 50: {round(RMSLE(df['meter_reading'], np.full(len(df), 50)), 5)}")
print(f"RMSLE for predicting the mean ({round(mean, 2)}): {round(RMSLE(df['meter_reading'], np.full(len(df), mean)), 5)}")


# In[7]:


const_rmsles = dict()
for i in range(75):
    const = i*2
    rmsle = round(RMSLE(df['meter_reading'], np.full(len(df), const)), 5)
#     print(f"RMSLE for predicting only {const}: {rmsle}")
    const_rmsles[const] = rmsle

xs = list(const_rmsles.keys())
ys = list(const_rmsles.values())

pd.DataFrame(ys, index=xs).plot(figsize=(15, 10), legend=None)
plt.scatter(min(const_rmsles, key=const_rmsles.get), sorted(ys)[0], color='red')
plt.title("RMSLE scores for constant predictions", fontsize=18, weight='bold')
plt.xticks(fontsize=14)
plt.xlabel("Constant", fontsize=14)
plt.ylabel("RMSLE", rotation=0, fontsize=14);


# In[8]:


# Formulate the best constant for this metric
best_const = np.expm1(np.mean(np.log1p(df['meter_reading'])))


# In[9]:


print(f"The best constant for our data is: {best_const}...")
print(f"RMSLE for predicting the best possible constant on our data: {round(RMSLE(df['meter_reading'], np.full(len(df), best_const)), 5)}\n")

print("This is the optimal RMSLE score that we can get with only a constant prediction and using all data available.\nWe therefore call it the best 'Naive baseline'\nA model should at least perform better than this RMSLE score.")


# In[10]:


# Random predictions
rand_rmsles = dict()
for i in range(15):
    magn = 10**(0.2*(i+1))
    rand_preds = np.random.randint(0, magn, len(df))
    rmsle = round(RMSLE(df['meter_reading'], rand_preds), 5)
    rand_rmsles[magn] = rmsle

xs = list(rand_rmsles.keys())
ys = list(rand_rmsles.values())  
    
pd.DataFrame(ys, index=xs).plot(figsize=(15, 10), legend=None)
plt.scatter(min(rand_rmsles, key=rand_rmsles.get),sorted(ys)[0],color='red')
plt.title("RMSLE scores for random predictions", fontsize=18, weight='bold')
plt.xticks(fontsize=14)
plt.xlabel("Maximum value", fontsize=14)
plt.ylabel("RMSLE", rotation=0, fontsize=14);


# In[11]:


# Read in sample submission and fill all predictions with the best constant
samp_sub = pd.read_csv(SAMP_SUB_PATH)
samp_sub['meter_reading'] = best_const
samp_sub.to_csv("best_constant_submission.csv", index=False)


# In[12]:


# Check Final Submission
print("Final Submission:")
samp_sub.head(2)

