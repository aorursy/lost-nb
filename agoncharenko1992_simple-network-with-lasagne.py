#!/usr/bin/env python
# coding: utf-8

# In[1]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn.preprocessing import StandardScaler
from sklearn.cross_validation import train_test_split
from sklearn.preprocessing import LabelEncoder
import theano
import theano.tensor as T
import lasagne
import time
import pickle
from lasagne.layers import DenseLayer, NonlinearityLayer, DropoutLayer
from lasagne.nonlinearities import softmax
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))

# Any results you write to the current directory are saved as output.


# In[2]:


def train_data_load():
    Xy_train_df = pd.read_csv("../input/train.csv")
    Xy_train_df['species'] = Xy_train_df.species.astype('category')
    cat_names = list(Xy_train_df.species.cat.categories)
    cat_code = range(len(Xy_train_df.species.cat.categories))
    species_dict = dict(zip(cat_code, cat_names))
    Xy_train_df['species'] = Xy_train_df.species.cat.codes
    y_train_df = Xy_train_df.species
    Xy_train_df.drop(labels='species', axis=1, inplace=True)
    X_train_df = Xy_train_df
    del Xy_train_df
    X = X_train_df.drop(labels='id', axis=1, inplace=False).values
    y = y_train_df.values
    scaler = StandardScaler()
    X = scaler.fit_transform(X, y)
    X, X_test, y, y_test = train_test_split(X, y, test_size=0.1)
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.1)
    return X_train, y_train, X_val, y_val, X_test, y_test, scaler, species_dict 
def gen_minibatches(X, y, batch_size, shuffle=False):
    ex_count = X.shape[0]
    assert ex_count==y.shape[0], "Training data sizes don't match"
    if shuffle:
        ids = np.random.permutation(ex_count)
    else:
        ids = np.arange(ex_count)
    for start_idx in range(0, ex_count - batch_size + 1, batch_size):
        ii = ids[start_idx:start_idx + batch_size]
        yield X[ii], y[ii]


# In[3]:


input_var = T.matrix('inputs', dtype=theano.config.floatX)
def model_construction(input_var, num_units_list, features_count, classes_count):
    network = lasagne.layers.InputLayer(shape=(None, features_count), input_var=input_var)
    for idx, curr_layer_num_units in enumerate(num_units_list):
        network = DenseLayer(network, num_units=curr_layer_num_units, 
                   W=lasagne.init.HeNormal(gain='relu'), b=lasagne.init.Constant(0.), 
                   nonlinearity=lasagne.nonlinearities.rectify, name='layer' + str(idx))
        network = DropoutLayer(network, p=0.6, name='dropout' + str(idx))
    network = DenseLayer(network, num_units=classes_count, nonlinearity=None, name='last_layer')
    network = NonlinearityLayer(network, softmax, name='probs')    
    return network    


# In[4]:


save_path = 'super_model'
def train(X_train, y_train, X_val, y_val, X_test, y_test, network, num_epochs=500, 
          learning_rate=0.001, learning_rate_decay=0.95, 
          momentum=0.9, momentum_decay=0.95, 
          decay_after_epochs=10, regu=0.002, batch_size=64, updates='adam'):
    
    target_var = T.ivector('target')
    prediction = lasagne.layers.get_output(network)
    loss = lasagne.objectives.categorical_crossentropy(prediction, target_var)
    cross_entr_loss = loss.mean()
    regu_loss = regu * lasagne.regularization.regularize_network_params(
        network, lasagne.regularization.l2)
    loss = cross_entr_loss + regu_loss
    print(updates)
    print("initial learning_rate=%f, decay_value %f per %d epoch, L2_reg coeff value=%f, batch_size=%d" 
          % (learning_rate, learning_rate_decay, decay_after_epochs, regu, batch_size))
    train_acc = T.mean(T.eq(T.argmax(prediction, axis=1), target_var), dtype=theano.config.floatX)
    learning_rate_var = theano.shared(np.float32(learning_rate))
    momentum_var = theano.shared(np.float32(momentum))
    params = lasagne.layers.get_all_params(network, trainable=True)
    if updates=='nesterov':
        updates = lasagne.updates.nesterov_momentum(loss, params, learning_rate=learning_rate_var,
                                                    momentum=momentum_var)
    else:
        updates = lasagne.updates.adam(loss, params)
    test_prediction = lasagne.layers.get_output(network, deterministic=True)
    test_loss = lasagne.objectives.categorical_crossentropy(test_prediction,
                                                            target_var)
    test_loss = test_loss.mean()
    test_acc = T.mean(T.eq(T.argmax(test_prediction, axis=1), target_var),
                      dtype=theano.config.floatX)
    train_fn = theano.function([input_var, target_var], [cross_entr_loss, regu_loss], updates=updates)
    train_acc_fn = theano.function([input_var, target_var], train_acc)
    val_fn = theano.function([input_var, target_var], [test_loss, test_acc])
    print("Training...")
    best_val_acc = 0.0
    best_model = None

    loss_history = []
    train_acc_history = []
    val_acc_history = []
    test_acc_history = []

    for epoch in range(num_epochs):
        train_err = train_batches = cross_loss = weights_sums = 0
        start_time = time.time()
        for X_batch, y_batch in gen_minibatches(X_train, y_train, batch_size, shuffle=True):
            cross_err, weights = train_fn(X_batch, y_batch)
            train_err += (cross_err + weights)
            cross_loss += cross_err
            weights_sums += weights
            train_batches += 1
            loss_history.append(cross_err + weights)
        # training accuracy
        n_acc = len(y_val)
        trval_err = trval_acc = trval_batches = 0
        for X_batch, y_batch in gen_minibatches(X_train[:n_acc], y_train[:n_acc], 
                                                batch_size, shuffle=False):
            err, acc = val_fn(X_batch, y_batch)
            trval_err += err
            trval_acc += acc
            trval_batches += 1
        trval_acc /= trval_batches
        train_acc_history.append(trval_acc)
        # validation accuracy
        val_err = val_acc = val_batches = 0
        for X_batch, y_batch in gen_minibatches(X_val, y_val, batch_size//2, shuffle=False):
            err, acc = val_fn(X_batch, y_batch)
            val_err += err
            val_acc += acc
            val_batches += 1
        val_acc /= val_batches
        val_acc_history.append(val_acc)

        test_err = test_acc = test_batches = 0
        for X_batch, y_batch in gen_minibatches(X_test, y_test, batch_size//2, shuffle=False):
            err, acc = val_fn(X_batch, y_batch)
            test_err += err
            test_acc += acc
            test_batches += 1
        test_acc /= test_batches
        test_acc_history.append(test_acc)
        
        # keep track of the best model based on validation accuracy
        if val_acc > best_val_acc:
            # make a copy of the model
            best_val_acc = val_acc
            best_model = lasagne.layers.get_all_param_values(network)
        if epoch % 50 == 0:
            print('epoch %d / %d in %.1fs: loss %f, cross_loss %f, weights_loss %f, train: %.3f, val %.3f, test %.3f, lr %e mom %e'
                  % (epoch + 1, num_epochs, time.time() - start_time,
                     train_err / train_batches, cross_loss / train_batches,
                     weights_sums / train_batches, trval_acc, val_acc, test_acc, 
                     learning_rate_var.get_value(), momentum_var.get_value()))
        # decay learning rate
        if (epoch + 1) % decay_after_epochs == 0:
            learning_rate_var.set_value(
                np.float32(learning_rate_var.get_value() * learning_rate_decay))
            momentum = (1.0 - (1.0 - momentum_var.get_value()) * momentum_decay)                        .clip(max=0.9999)
            momentum_var.set_value(np.float32(momentum))
        # save model snapshots
        if save_path and (epoch + 1) % 10 == 0:
            model = lasagne.layers.get_all_param_values(network)
            path = '%s.pickle' % (save_path)
            with open(path, 'wb') as f:
                pickle.dump({'model': model}, f, -1)
    return network          


# In[5]:


def predict_proba(network, Xtest_vals, save_path):
    if save_path is not None:
        path = '%s.pickle' % (save_path)
        with open(path, 'rb') as f:
            data_new = pickle.load(f)
            print(len(data_new['model']))
            lasagne.layers.set_all_param_values(network, data_new['model'])
    proba_tensor = lasagne.layers.get_output(network, Xtest_vals, deterministic=True)
    proba_vals = proba_tensor.eval()
    return proba_vals

def main():
    X_train, y_train, X_val, y_val, X_test, y_test, scaler, species_dict = train_data_load()
    network = model_construction(input_var=input_var, num_units_list=[128, 64], 
                                 features_count=X_train.shape[1], classes_count=99)
    network = train(X_train, y_train, X_val, y_val, X_test, y_test, network, num_epochs=700, regu=0.002,
                   batch_size=32)
    X_test = pd.read_csv("../input/test.csv")
    Xtest_vals = X_test.drop(labels='id', axis=1, inplace=False).values
    Xtest_vals = scaler.transform(Xtest_vals)
    print(Xtest_vals.shape)
    probs = predict_proba(network, Xtest_vals, save_path)
    print(probs)
    cols = list(species_dict.values())
    predicted = pd.DataFrame(probs, columns=cols)
    predicted = pd.concat([X_test.id, predicted], axis=1)
    predicted.to_csv("sample_submission.csv", index=False)


# In[6]:


main()


# In[7]:




