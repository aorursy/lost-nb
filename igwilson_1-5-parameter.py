#!/usr/bin/env python
# coding: utf-8



trn_x, val_x, trn_y, val_y = cross_validation.train_test_split(x, y, 
                                                               random_state=42, 
                                                               stratify=y,
                                                               test_size=0.20)

clf = xgb.XGBRegressor(max_depth=10,
                       n_estimators=1500,
                       min_child_weight=9,
                       learning_rate=0.05,
                       nthread=8,
                       subsample=0.80,
                       colsample_bytree=0.80,
                       seed=4242)

clf.fit(trn_x, trn_y, eval_set=[(val_x, val_y)], verbose=True, eval_metric='logloss', 
                                early_stopping_rounds=50)


# so there are a number of paramters for us to vary

random_state
stratify
test_size
max_depth
n_estimators
learning_rate
nthread
subsample
colsample_bytree
seed
early_stopping_rounds=50






## MXNET

model = mx.model.FeedForward.load('model/resnet-50', 0, ctx=mx.cpu(), numpy_batch_size=1)

fea_symbol = model.symbol.get_internals()["flatten0_output"]

feature_extractor = mx.model.FeedForward(ctx=mx.cpu()
					, symbol=fea_symbol
					, numpy_batch_size=64
					, arg_params=model.arg_params
					, aux_params=model.aux_params
					, allow_extra_params=True)
feature_extractor



# so there are a number of paramters for us to vary


# try different internals
model.symbol.get_internals()["flatten0_output"]
numpy_batch_size






# try different models
mx.model.FeedForward.load('model/resnet-50', 0, ctx=mx.cpu(), numpy_batch_size=1)
numpy_batch_size=1






# we can code different permutations of parameters using the code on the link below

https://docs.python.org/3/library/itertools.html#itertools.combinations




# we can try this on 10% data initially."""
So we need to pick out the best combination of parameters for the xgboost model on MXNET.  
We also need to work out if the feature extraction on MXNET can be improved.

** start by getting it working on 5% then validate on 20%.

Use the "best" resize/mask/transformation:

XGBOOST
(1)  Identify xgboost parameters to be varied.
(2)  Identify the range the parameters can take.
(3)  Run all possible xgboost permutations (Tabulate all results)


Use "best" XGBOOST model for the below:

MXNET
(1)  identify paramters to be varied
(2)  indentify the range of possible values each of the paramters can take
(3)  run all possible permutations (Tabulate all results)


"""

trn_x, val_x, trn_y, val_y = cross_validation.train_test_split(x, y, 
                                                               random_state=42, 
                                                               stratify=y,
                                                               test_size=0.20)

clf = xgb.XGBRegressor(max_depth=10,
                       n_estimators=1500,
                       min_child_weight=9,
                       learning_rate=0.05,
                       nthread=8,
                       subsample=0.80,
                       colsample_bytree=0.80,
                       seed=4242)

clf.fit(trn_x, trn_y, eval_set=[(val_x, val_y)], verbose=True, eval_metric='logloss', 
                                early_stopping_rounds=50)


# so there are a number of paramters for us to vary

random_state
stratify
test_size
max_depth
n_estimators
learning_rate
nthread
subsample
colsample_bytree
seed
early_stopping_rounds=50






## MXNET

model = mx.model.FeedForward.load('model/resnet-50', 0, ctx=mx.cpu(), numpy_batch_size=1)

fea_symbol = model.symbol.get_internals()["flatten0_output"]

feature_extractor = mx.model.FeedForward(ctx=mx.cpu()
					, symbol=fea_symbol
					, numpy_batch_size=64
					, arg_params=model.arg_params
					, aux_params=model.aux_params
					, allow_extra_params=True)
feature_extractor



# so there are a number of paramters for us to vary


# try different internals
model.symbol.get_internals()["flatten0_output"]
numpy_batch_size






# try different models
mx.model.FeedForward.load('model/resnet-50', 0, ctx=mx.cpu(), numpy_batch_size=1)
numpy_batch_size=1






# we can code different permutations of parameters using the code on the link below

https://docs.python.org/3/library/itertools.html#itertools.combinations




# we can try this on 10% data initially.







# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))

# Any results you write to the current directory are saved as output.




import numpy as np

np.where([[True, False], [True, True]], [[1, 2], [3, 4]], [[9, 8], [7, 6]])




import numpy as np

x = np.random.randn(20, 3)
x_new = x[np.sum(x, axis=1) > .5]




arr = np.array([[1,2,3,4], [5,6,7,8], [9,10,11,12]])




arr.shape




d = np.random.randint(100, 300, size=(350,300,150))




d.shape




d




x_new = d[np.sum(d, axis=1) > 100]




np.sum([[0, 1], [0, 5], [0, 6]], axis=1)




test = np.array([[0, 1], [0, 5], [0, 6]])




test.shape




d = np.random.randint(100, 300, size=(350,300,150))




d.shape




test=np.sum(d, axis=2)




d[ (3>d[:,1,:]) & (d[:,1,:]>-6) ]




test=d[np.std(d[:,:,:])==57.605366674526657]




test.shape




# we need a table of the imgs_resampled the std and we can remove the imgs_resampled which equal zero.

# are create a new images_resampled file.

