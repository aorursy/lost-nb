#!/usr/bin/env python
# coding: utf-8



# Standard Dependencies
import os
import scipy as sp
import numpy as np
import random as rn
import pandas as pd
from numba import jit
from functools import partial

# The metric in question
from sklearn.metrics import cohen_kappa_score

# Machine learning
import tensorflow as tf
import keras.backend as K
from keras.callbacks import Callback

# Set seed for reproducability
seed = 1234
rn.seed(seed)
np.random.seed(seed)
os.environ['PYTHONHASHSEED'] = str(seed)

# Specify paths
PATH = "../input/data-science-bowl-2019/"
TRAIN_PATH = PATH + "train_labels.csv"
SUB_PATH = PATH + "sample_submission.csv"




# File sizes and specifications
print('\n# Files and file sizes')
for file in os.listdir(PATH):
    print('{}| {} MB'.format(file.ljust(30), 
                             str(round(os.path.getsize(PATH + file) / 1000000, 2))))




# Load in data
df = pd.read_csv(TRAIN_PATH)




df.head(3)




def sklearn_qwk(y_true, y_pred) -> np.float64:
    """
    Function for measuring Quadratic Weighted Kappa with scikit-learn
    
    :param y_true: The ground truth labels
    :param y_pred: The predicted labels
    
    :return The Quadratic Weighted Kappa Score (QWK)
    """
    return cohen_kappa_score(y_true, y_pred, weights="quadratic")




@jit
def cpmp_qwk(a1, a2, max_rat=3) -> float:
    """
    A ultra fast implementation of Quadratic Weighted Kappa (QWK)
    Source: https://www.kaggle.com/c/data-science-bowl-2019/discussion/114133
    
    :param a1: The ground truth labels
    :param a2: The predicted labels
    :param max_rat: The maximum target value
    
    return: A floating point number with the QWK score
    """
    assert(len(a1) == len(a2))
    a1 = np.asarray(a1, dtype=int)
    a2 = np.asarray(a2, dtype=int)

    hist1 = np.zeros((max_rat + 1, ))
    hist2 = np.zeros((max_rat + 1, ))

    o = 0
    for k in range(a1.shape[0]):
        i, j = a1[k], a2[k]
        hist1[i] += 1
        hist2[j] += 1
        o +=  (i - j) * (i - j)

    e = 0
    for i in range(max_rat + 1):
        for j in range(max_rat + 1):
            e += hist1[i] * hist2[j] * (i - j) * (i - j)

    e = e / a1.shape[0]

    return 1 - o / e




# Get the ground truth labels
true_labels = df['accuracy_group']




# Check which labels are present
print("Label Distribution:")
df['accuracy_group'].value_counts()




# Calculate scores for very naive baselines
dumb_score = sklearn_qwk(true_labels, np.full(len(true_labels), 3))
random_score = round(sklearn_qwk(true_labels, np.random.randint(0, 4, size=len(true_labels))), 5)
print(f"Simply predicting the most common class will yield a QWK score of:\n{dumb_score}\n")
print(f"Random predictions will yield a QWK score of:\n{random_score}")




print("Assessment types in the training data:")
list(set(df['title']))




# Group by assessments and take the mode
mode_mapping = df.groupby('title')['accuracy_group'].agg(lambda x:x.value_counts().index[0])
mode_preds = df['title'].map(mode_mapping)

# Group by assessments and take the rounded mean
mean_mapping = df.groupby('title')['accuracy_group'].mean().round()
mean_preds = df['title'].map(mean_mapping)




# Check which a score a less naive baseline would give
grouped_mode_score = round(sklearn_qwk(true_labels, mode_preds), 5)
grouped_mean_score = round(sklearn_qwk(true_labels, mean_preds), 5)
print(f"The naive grouping of the assessments and taking the mode will yield us a QWK score of:\n{grouped_mode_score}")
print(f"The naive grouping of the assessments and taking the rounded mean will yield us a QWK score of:\n{grouped_mean_score}")




# Map the mean based on the assessment title
raw_mean_mapping = df.groupby('title')['accuracy_group'].mean()
raw_mean_preds = df['title'].map(raw_mean_mapping)




class OptimizedRounder(object):
    """
    An optimizer for rounding thresholds
    to maximize Quadratic Weighted Kappa (QWK) score
    """
    def __init__(self):
        self.coef_ = 0

    def _kappa_loss(self, coef, X, y):
        """
        Get loss according to
        using current coefficients
        
        :param coef: A list of coefficients that will be used for rounding
        :param X: The raw predictions
        :param y: The ground truth labels
        """
        X_p = np.copy(X)
        for i, pred in enumerate(X_p):
            if pred < coef[0]:
                X_p[i] = 0
            elif pred >= coef[0] and pred < coef[1]:
                X_p[i] = 1
            elif pred >= coef[1] and pred < coef[2]:
                X_p[i] = 2
            elif pred >= coef[2] and pred < coef[3]:
                X_p[i] = 3
            else:
                X_p[i] = 4

        ll = cohen_kappa_score(y, X_p, weights='quadratic')
        return -ll

    def fit(self, X, y):
        """
        Optimize rounding thresholds
        
        :param X: The raw predictions
        :param y: The ground truth labels
        """
        loss_partial = partial(self._kappa_loss, X=X, y=y)
        initial_coef = [0.5, 1.5, 2.5, 3.5]
        self.coef_ = sp.optimize.minimize(loss_partial, initial_coef, method='nelder-mead')

    def predict(self, X, coef):
        """
        Make predictions with specified thresholds
        
        :param X: The raw predictions
        :param coef: A list of coefficients that will be used for rounding
        """
        X_p = np.copy(X)
        for i, pred in enumerate(X_p):
            if pred < coef[0]:
                X_p[i] = 0
            elif pred >= coef[0] and pred < coef[1]:
                X_p[i] = 1
            elif pred >= coef[1] and pred < coef[2]:
                X_p[i] = 2
            elif pred >= coef[2] and pred < coef[3]:
                X_p[i] = 3
            else:
                X_p[i] = 4
        return X_p

    def coefficients(self):
        """
        Return the optimized coefficients
        """
        return self.coef_['x']




# Optimize rounding thresholds (No effect since we have naive baselines)
optR = OptimizedRounder()
optR.fit(mode_preds, true_labels)
coefficients = optR.coefficients()
opt_preds = optR.predict(raw_mean_preds, coefficients)
new_score = sklearn_qwk(true_labels, opt_preds)




print(f"Optimized Thresholds:\n{coefficients}\n")
print(f"The Quadratic Weighted Kappa (QWK)\nwith optimized rounding thresholds is: {round(new_score, 5)}\n")
print(f"This is an improvement of {round(new_score - grouped_mean_score, 5)} over the unoptimized rounding.")




class QWK(Callback):
    """
    A custom Keras callback for saving the best model
    according to the Quadratic Weighted Kappa (QWK) metric
    """
    def __init__(self, model_name="model.h5"):
        self.model_name = model_name
    
    def on_train_begin(self, logs={}):
        """
        Initialize list of QWK scores on validation data
        """
        self.val_kappas = []

    def on_epoch_end(self, epoch, logs={}):
        """
        Gets QWK score on the validation data
        
        :param epoch: The current epoch number
        """
        # Get predictions and convert to integers
        y_pred, labels = get_preds_and_labels(model, val_generator)
        y_pred = np.rint(y_pred).astype(np.uint8).clip(0, 4)
        _val_kappa = cpmp_qwk(labels, y_pred)
        self.val_kappas.append(_val_kappa)
        print(f"val_kappa: {round(_val_kappa, 4)}")
        if _val_kappa == max(self.val_kappas):
            print("Validation Kappa has improved. Saving model.")
            self.model.save(self.model_name)
        return
    
def get_preds_and_labels(model, generator):
    """
    Get predictions and labels from the generator
    """
    preds = []
    labels = []
    for _ in range(int(np.ceil(generator.samples / BATCH_SIZE))):
        x, y = next(generator)
        preds.append(model.predict(x))
        labels.append(y)
    # Flatten list of numpy arrays
    return np.concatenate(preds).ravel(), np.concatenate(labels).ravel()




def _cohen_kappa(y_true, y_pred, num_classes, weights=None, metrics_collections=None, updates_collections=None, name=None):
    kappa, update_op = tf.contrib.metrics.cohen_kappa(y_true, y_pred, num_classes, weights, metrics_collections, updates_collections, name)
    K.get_session().run(tf.local_variables_initializer())
    with tf.control_dependencies([update_op]):
        kappa = tf.identity(kappa)
    return kappa

def cohen_kappa_loss(num_classes, weights=None, metrics_collections=None, updates_collections=None, name=None):
    """
    A loss function that measures the Quadratic Weighted Kappa (QWK) score
    and can be used in a Tensorflow / Keras model
    """
    def cohen_kappa(y_true, y_pred):
        return -_cohen_kappa(y_true, y_pred, num_classes, weights, metrics_collections, updates_collections, name)
    return cohen_kappa




# Read in Test Data
test_df = pd.read_csv(PATH + "test.csv")

# Map the mode to the test data and create the final predictions through aggregation
test_df['preds'] = test_df['title'].map(mode_mapping)
final_preds = test_df.groupby('installation_id')['preds'].agg(lambda x:x.value_counts().index[0])




# Make submission for Kaggle
sub_df = pd.read_csv(SUB_PATH)
sub_df['accuracy_group'] = list(final_preds.fillna(0).astype(np.uint8))
sub_df.to_csv("submission.csv", index=False);




print('Final predictions:')
sub_df.head(2)

