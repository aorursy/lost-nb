#!/usr/bin/env python
# coding: utf-8



# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from subprocess import check_output
print(check_output(["ls", "../working"]).decode("utf8"))
aaa = pd.read_csv('../working/cat_predicts.csv', sep=',')
aaa.head()
# Any results you write to the current directory are saved as output.




from sklearn import datasets
from sklearn.utils import check_array
from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_val_score
from sklearn.base import ClassifierMixin
from collections import Counter
from sklearn.feature_selection import VarianceThreshold
from scipy.stats import pearsonr
from sklearn.feature_selection import RFE
from sklearn.cross_validation import KFold
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import mean_squared_error, make_scorer
from sklearn.ensemble import AdaBoostClassifier
from xgboost import XGBRegressor
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification
from sklearn.decomposition import PCA
from imblearn.over_sampling import RandomOverSampler
import xgboost as xgb
import seaborn as sns
from sklearn.grid_search import GridSearchCV
from sklearn.linear_model import SGDClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn.feature_selection import SelectFromModel
from sklearn.metrics import f1_score
from sklearn.metrics import matthews_corrcoef
from collections import Counter
import itertools
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import BaggingClassifier
from sklearn.metrics import confusion_matrix
from imblearn.datasets import make_imbalance
from imblearn.ensemble import BalancedBaggingClassifier
from imblearn.metrics import classification_report_imbalanced
from sklearn import pipeline, metrics, grid_search
from sklearn.feature_selection import SelectFromModel
from sklearn.ensemble import GradientBoostingClassifier
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt




def gini(solution, submission):
    df = zip(solution, submission, range(len(solution)))
    df = sorted(df, key=lambda x: (x[1],-x[2]), reverse=True)
    rand = [float(i+1)/float(len(df)) for i in range(len(df))]
    totalPos = float(sum([x[0] for x in df]))
    cumPosFound = [df[0][0]]
    for i in range(1,len(df)):
        cumPosFound.append(cumPosFound[len(cumPosFound)-1] + df[i][0])
    Lorentz = [float(x)/totalPos for x in cumPosFound]
    Gini = [Lorentz[i]-rand[i] for i in range(len(df))]
    return sum(Gini)

def normalized_gini(solution, submission):
    normalized_gini = gini(solution, submission)/gini(solution, solution)
    return normalized_gini
gini_scorer = metrics.make_scorer(normalized_gini, greater_is_better = True)




trainDF = pd.read_csv('../input/kaggle-seguro/train/train.csv', sep=',')
testDF = pd.read_csv('../input/dataset/test/test.csv', sep=',')
target = trainDF.pop('target')




plt.figure(figsize=(10,3))
sns.countplot(trainDF['target'],palette='rainbow')
plt.xlabel('Target')
trainDF['target'].value_counts()




cor = trainDF.corr()
plt.figure(figsize=(16,10))
sns.heatmap(cor)




ps_cal = trainDF.columns[trainDF.columns.str.startswith('ps_calc')] 




id_test = testDF['id'].values
trainDF = trainDF.drop(ps_cal,axis =1)
trainDF = trainDF.drop(['id'],axis =1)
testDF = testDF.drop(ps_cal,axis =1)
testDF = testDF.drop(['id'],axis =1)




cor = trainDF.corr()
plt.figure(figsize=(16,10))
sns.heatmap(cor)




# def missing_value(df):
#     col = df.columns
#     for i in col:
#         if df[i].isnull().sum()>0:
#             df[i].fillna(df[i].mode()[0],inplace=True)
# missing_value(trainDF)




trainDF = trainDF.fillna(999)
testDF = testDF.fillna(999)




for c in trainDF.select_dtypes(include=['float64']).columns:
    trainDF[c]=trainDF[c].astype(np.float32)
    testDF[c]=testDF[c].astype(np.float32)
for c in trainDF.select_dtypes(include=['int64']).columns[2:]:
    trainDF[c]=trainDF[c].astype(np.int8)
    testDF[c]=testDF[c].astype(np.int8)  




from catboost import CatBoostClassifier, Pool
y_train = target.values
x_train = trainDF
x_test = testDF

train_data = Pool(x_train, y_train)
test_data = Pool(x_test)




props = {
        	leaf_estimation_method ='Newton',
        	learning_rate=0.057,
          	l2_leaf_reg = 23,
          	depth=6,
          	od_pval=0.0000001,
          	iterations = 877,
          	loss_function='Logloss'
          
        }




from tqdm import tqdm
print('Starting the loop...')
num_ensembles = 6
y_pred = 0.0
for i in tqdm(range(num_ensembles)):
    model = CatBoostClassifier(random_seed = i+200, gradient_iterations = i+1 ,leaf_estimation_method ='Newton', learning_rate=0.057, l2_leaf_reg = 23, depth=6, od_pval=0.0000001, iterations = 877, loss_function='Logloss')
    fit_model = model.fit(train_data)
    y_pred +=  fit_model.predict_proba(test_data)[:,1]
y_pred /= num_ensembles
gc.collect()


# Create a submission file
sub = pd.DataFrame()
sub['id'] = id_test
sub['target'] = y_pred
sub.to_csv('cat_predicts.csv', index=False)




print('done')




cat_col = [col for col in trainDF.columns if '_cat' in col]
print(cat_col)
for c in cat_col:
    trainDF[c] = trainDF[c].astype('uint8')
#     test[c] = test[c].astype('uint8') 




bin_col = [col for col in trainDF.columns if 'bin' in col]
print(bin_col)
for c in bin_col:
    trainDF[c] = trainDF[c].astype('uint8')
#     test[c] = test[c].astype('uint8') 




train_X = trainDF.loc[:,trainDF.columns[:len(trainDF.columns)-1]]
train_y = trainDF.loc[:,['target']].values.ravel()




model = GradientBoostingClassifier(n_estimators=200)
score = cross_val_score(model,train_X,train_y,cv=5, scoring="accuracy")
print(score.mean())




def runXGB(xtrain,xvalid,ytrain,yvalid,xtest,eta=0.1,num_rounds=100,max_depth=4):
    params = {
        'objective':'binary:logistic',        
        'max_depth':max_depth,
        'learning_rate':eta,
        'eval_metric':'auc',
        'min_child_weight':6,
        'subsample':0.8,
        'colsample_bytree':0.8,
        'seed':seed,
        'reg_lambda':1.3,
        'reg_alpha':8,
        'gamma':10,
        'scale_pos_weight':1.6
        #'n_thread':-1
    }
    
    dtrain = xgb.DMatrix(xtrain,label=ytrain)
    dvalid = xgb.DMatrix(xvalid,label=yvalid)
    dtest = xgb.DMatrix(xtest)
    watchlist = [(dtrain,'train'),(dvalid,'test')]
    
    model = xgb.train(params,dtrain,num_rounds,watchlist,early_stopping_rounds=50,verbose_eval=50)
    pred = model.predict(dvalid,ntree_limit=model.best_ntree_limit)
    pred_test = model.predict(dtest,ntree_limit=model.best_ntree_limit)
    return pred_test,model
    




from catboost import CatBoostClassifier, Pool
from tqdm import tqdm
train_data = Pool(train_X, train_y)
test_data = Pool(train_X)
num_ensembles = 6
y_pred = 0.0
for i in tqdm(range(num_ensembles)):
    model = CatBoostClassifier(random_seed = i+200, gradient_iterations = i+1 ,leaf_estimation_method ='Newton', learning_rate=0.057, l2_leaf_reg = 23, depth=6, od_pval=0.0000001, iterations = 877, loss_function='Logloss')
    fit_model = model.fit(train_data)
    y_pred +=  fit_model.predict_proba(test_data)[:,1]
y_pred /= num_ensembles
gc.collect()
# Create a submission file
sub = pd.DataFrame()
sub['id'] = id_test
sub['target'] = y_pred
sub.to_csv('cat_predicts.csv', index=False)

