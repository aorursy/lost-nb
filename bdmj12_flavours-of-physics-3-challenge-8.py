#!/usr/bin/env python
# coding: utf-8



version = '8.0'
folder = '../input/'




### THIS CELL IS JUST THE EVALUATION PYTHON FILE 

import numpy
from sklearn.metrics import roc_curve, auc


def __rolling_window(data, window_size):
    """
    Rolling window: take window with definite size through the array

    :param data: array-like
    :param window_size: size
    :return: the sequence of windows

    Example: data = array(1, 2, 3, 4, 5, 6), window_size = 4
        Then this function return array(array(1, 2, 3, 4), array(2, 3, 4, 5), array(3, 4, 5, 6))
    """
    shape = data.shape[:-1] + (data.shape[-1] - window_size + 1, window_size)
    strides = data.strides + (data.strides[-1],)
    return numpy.lib.stride_tricks.as_strided(data, shape=shape, strides=strides)


def __cvm(subindices, total_events):
    """
    Compute Cramer-von Mises metric.
    Compared two distributions, where first is subset of second one.
    Assuming that second is ordered by ascending

    :param subindices: indices of events which will be associated with the first distribution
    :param total_events: count of events in the second distribution
    :return: cvm metric
    """
    target_distribution = numpy.arange(1, total_events + 1, dtype='float') / total_events
    subarray_distribution = numpy.cumsum(numpy.bincount(subindices, minlength=total_events), dtype='float')
    subarray_distribution /= 1.0 * subarray_distribution[-1]
    return numpy.mean((target_distribution - subarray_distribution) ** 2)


def compute_cvm(predictions, masses, n_neighbours=200, step=50):
    """
    Computing Cramer-von Mises (cvm) metric on background events: take average of cvms calculated for each mass bin.
    In each mass bin global prediction's cdf is compared to prediction's cdf in mass bin.

    :param predictions: array-like, predictions
    :param masses: array-like, in case of Kaggle tau23mu this is reconstructed mass
    :param n_neighbours: count of neighbours for event to define mass bin
    :param step: step through sorted mass-array to define next center of bin
    :return: average cvm value
    """
    predictions = numpy.array(predictions)
    masses = numpy.array(masses)
    assert len(predictions) == len(masses)

    # First, reorder by masses
    predictions = predictions[numpy.argsort(masses)]

    # Second, replace probabilities with order of probability among other events
    predictions = numpy.argsort(numpy.argsort(predictions, kind='mergesort'), kind='mergesort')

    # Now, each window forms a group, and we can compute contribution of each group to CvM
    cvms = []
    for window in __rolling_window(predictions, window_size=n_neighbours)[::step]:
        cvms.append(__cvm(subindices=window, total_events=len(predictions)))
    return numpy.mean(cvms)


def __roc_curve_splitted(data_zero, data_one, sample_weights_zero, sample_weights_one):
    """
    Compute roc curve

    :param data_zero: 0-labeled data
    :param data_one:  1-labeled data
    :param sample_weights_zero: weights for 0-labeled data
    :param sample_weights_one:  weights for 1-labeled data
    :return: roc curve
    """
    labels = [0] * len(data_zero) + [1] * len(data_one)
    weights = numpy.concatenate([sample_weights_zero, sample_weights_one])
    data_all = numpy.concatenate([data_zero, data_one])
    fpr, tpr, _ = roc_curve(labels, data_all, sample_weight=weights)
    return fpr, tpr


def compute_ks(data_prediction, mc_prediction, weights_data, weights_mc):
    """
    Compute Kolmogorov-Smirnov (ks) distance between real data predictions cdf and Monte Carlo one.

    :param data_prediction: array-like, real data predictions
    :param mc_prediction: array-like, Monte Carlo data predictions
    :param weights_data: array-like, real data weights
    :param weights_mc: array-like, Monte Carlo weights
    :return: ks value
    """
    assert len(data_prediction) == len(weights_data), 'Data length and weight one must be the same'
    assert len(mc_prediction) == len(weights_mc), 'Data length and weight one must be the same'

    data_prediction, mc_prediction = numpy.array(data_prediction), numpy.array(mc_prediction)
    weights_data, weights_mc = numpy.array(weights_data), numpy.array(weights_mc)

    assert numpy.all(data_prediction >= 0.) and numpy.all(data_prediction <= 1.), 'Data predictions are out of range [0, 1]'
    assert numpy.all(mc_prediction >= 0.) and numpy.all(mc_prediction <= 1.), 'MC predictions are out of range [0, 1]'

    weights_data /= numpy.sum(weights_data)
    weights_mc /= numpy.sum(weights_mc)

    fpr, tpr = __roc_curve_splitted(data_prediction, mc_prediction, weights_data, weights_mc)

    Dnm = numpy.max(numpy.abs(fpr - tpr))
    return Dnm


def roc_auc_truncated(labels, predictions, tpr_thresholds=(0.2, 0.4, 0.6, 0.8),
                      roc_weights=(4, 3, 2, 1, 0)):
    """
    Compute weighted area under ROC curve.

    :param labels: array-like, true labels
    :param predictions: array-like, predictions
    :param tpr_thresholds: array-like, true positive rate thresholds delimiting the ROC segments
    :param roc_weights: array-like, weights for true positive rate segments
    :return: weighted AUC
    """
    assert numpy.all(predictions >= 0.) and numpy.all(predictions <= 1.), 'Data predictions are out of range [0, 1]'
    assert len(tpr_thresholds) + 1 == len(roc_weights), 'Incompatible lengths of thresholds and weights'
    fpr, tpr, _ = roc_curve(labels, predictions)
    area = 0.
    tpr_thresholds = [0.] + list(tpr_thresholds) + [1.]
    for index in range(1, len(tpr_thresholds)):
        tpr_cut = numpy.minimum(tpr, tpr_thresholds[index])
        tpr_previous = numpy.minimum(tpr, tpr_thresholds[index - 1])
        area += roc_weights[index - 1] * (auc(fpr, tpr_cut, reorder=True) - auc(fpr, tpr_previous, reorder=True))
    tpr_thresholds = numpy.array(tpr_thresholds)
    # roc auc normalization to be 1 for an ideal classifier
    area /= numpy.sum((tpr_thresholds[1:] - tpr_thresholds[:-1]) * numpy.array(roc_weights))
    return area




def check_ag_test(model,var):
    check_agreement = pd.read_csv(folder + 'check_agreement.csv', index_col='id')
    agreement_probs = model.predict_proba(check_agreement[var])[:, 1]
    
    ks = compute_ks(
        agreement_probs[check_agreement['signal'].values == 0],
        agreement_probs[check_agreement['signal'].values == 1],
        check_agreement[check_agreement['signal'] == 0]['weight'].values,
        check_agreement[check_agreement['signal'] == 1]['weight'].values)
    print('KS metric', ks, ks < 0.09)
    return ks<0.09




def check_corr_test(model,var):
    

    check_correlation = pd.read_csv(folder + 'check_correlation.csv', index_col='id')
    correlation_probs = model.predict_proba(check_correlation[var])[:, 1]
    cvm = compute_cvm(correlation_probs, check_correlation['mass'])
    print('CvM metric', cvm, cvm < 0.002)
    return cvm<0.002




from sklearn.model_selection import cross_val_score

def comp_auc(model,var,data):
    train_eval = data[data['min_ANNmuon'] > 0.4]
    train_probs = model.predict_proba(train_eval[var])[:, 1]
    AUC = roc_auc_truncated(train_eval['signal'], train_probs)
    print('AUC', AUC)
    return AUC




def pred_file(model,var):

    test = pd.read_csv(folder + 'test.csv', index_col='id')
    
    result = pd.DataFrame({'id': test.index})
    result['prediction'] = model.predict_proba(test[var])[:, 1]
    result.to_csv('prediction %s .csv' % version, index=False, sep=',')




import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sklearn
import seaborn as sns




train = pd.read_csv(folder + 'training.csv', index_col='id')




train.head()




train.info() # to check for missing values




plt.figure(figsize=(5,20))
sns.heatmap(train.corr()["signal"].to_frame().sort_values(by="signal", ascending=False), annot=True, center=0)




# these are the variables that we want to include 
#(mainly based on common sense and trial and error)

variables = train.drop(["production", "min_ANNmuon","signal","mass", # these are not to be included
                        "SPDhits", # including this makes agreement test fail
                        "FlightDistanceError" # this seems to worsen score - perhaps not relevant (noise)
                       ],axis=1).columns
variables




candidate_models = {}   # we'll store candidate models here

#  split train dataset into 4 subsets for cross-validation
from sklearn.utils import shuffle
s_train = shuffle(train)
l = len(s_train)

l_1 = int(l/4)
l_2 = int(l/2)
l_3 = int(3*l/4)

ind_1=s_train.index[[i for i in range(l_1)]]
ind_2=s_train.index[[i for i in range(l_1,l_2)]]
ind_3 = s_train.index[[i for i in range(l_2,l_3)]]
ind_4 = s_train.index[[i for i in range(l_3,len(s_train))]]

sig_1 = s_train['signal'].drop(ind_1)
sig_2 = s_train['signal'].drop(ind_2)
sig_3 = s_train['signal'].drop(ind_3)
sig_4 = s_train['signal'].drop(ind_4)

var_1 = s_train[variables].drop(ind_1)
var_2 = s_train[variables].drop(ind_2)
var_3 = s_train[variables].drop(ind_3)
var_4 = s_train[variables].drop(ind_4)

train_1 = s_train[:l_1]
train_2 = s_train[l_1:l_2]
train_3 = s_train[l_2:l_3]
train_4 = s_train[l_3:]


def test_model(model):
    #if the model passes the tests...
    model.fit(train[variables], train['signal'])
    if(check_corr_test(model,variables) and check_ag_test(model,variables)):
       
        # evaluate the model on the 4 subsets
        model.fit(var_1,sig_1)
        val_1 = comp_auc(model,variables,train_1) 
        model.fit(var_2,sig_2)
        val_2 = comp_auc(model,variables,train_2) 
        model.fit(var_3,sig_3)
        val_3 = comp_auc(model,variables,train_3) 
        model.fit(var_4,sig_4)
        val_4 = comp_auc(model,variables,train_4)
        
        val =(val_1+val_2+val_3+val_4)/4
        
        print("Average AUC is: " + str(val))
                        
            
        model.fit(train[variables], train['signal'])
        #...add the model trained on all the data to the candidates
        candidate_models[model] = val
        print('passed')
    else:
        print('failed')




from sklearn.ensemble import GradientBoostingClassifier

test_model(GradientBoostingClassifier())

print('-----')
test_model(GradientBoostingClassifier(learning_rate=0.2, n_estimators=200, 
                                max_depth=5))

print('-----')
test_model(GradientBoostingClassifier(learning_rate=0.1, n_estimators=300, 
                                max_depth=10,max_features = 10))
           
print('-----')
test_model(GradientBoostingClassifier(learning_rate=0.2, n_estimators=200, 
                                max_depth=15))

print('-----')
test_model(GradientBoostingClassifier(learning_rate=0.1, n_estimators=200, 
                                max_depth=6,max_features = 6))

print('-----')
test_model(GradientBoostingClassifier(learning_rate=0.05, n_estimators=300, 
                                max_depth=4))

print('-----')
test_model(GradientBoostingClassifier(learning_rate=0.3, n_estimators=200, 
                                max_depth=6,                                  
                                 warm_start=True))




from sklearn.linear_model import LogisticRegression

for c in range(0,5):
    print('-----')
    test_model(LogisticRegression(C=0.8 + c*0.2))




from sklearn.naive_bayes import GaussianNB

test_model(GaussianNB())




from sklearn.neighbors import KNeighborsClassifier


test_model(KNeighborsClassifier())
           
print('-----')
test_model(KNeighborsClassifier(n_neighbors=8,  leaf_size=20))           




from sklearn.tree import DecisionTreeClassifier

print('-----')
test_model(DecisionTreeClassifier())

print('-----')
test_model(DecisionTreeClassifier(max_depth = 10,max_features=5))

print('-----')
test_model(DecisionTreeClassifier(max_depth = 12,max_features=8))

print('-----')
test_model(DecisionTreeClassifier(max_depth = 8,max_features=10))




from sklearn.ensemble import RandomForestClassifier

test_model(RandomForestClassifier())

print('-----')
test_model(RandomForestClassifier(max_depth = 10,max_features=5))

print('-----')
test_model(RandomForestClassifier(max_depth = 12,max_features=7))

print('-----')
test_model(RandomForestClassifier(max_depth = 8,max_features=10))

print('-----')
test_model(RandomForestClassifier(max_depth = 10,max_features=5,n_estimators=20))




from sklearn.neural_network import MLPClassifier

test_model(MLPClassifier())

print('-----')
test_model(MLPClassifier(hidden_layer_sizes=(150,) max_iter=200))




candidate_models




best_model = max(candidate_models, key=candidate_models.get)
type(best_model)




candidate_models[best_model]




pred_file(best_model,variables)

