#!/usr/bin/env python
# coding: utf-8



# Python ≥3.5 is required
import sys
assert sys.version_info >= (3, 5)

# Scikit-Learn ≥0.20 is required
import sklearn
assert sklearn.__version__ >= "0.20"

# Common imports
import numpy as np
import os
import gc
import time

#NLP packages
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.corpus import stopwords
import re
from string import punctuation
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.decomposition import TruncatedSVD

stop_words = set(stopwords.words('english'))


from collections import Counter

import matplotlib.pyplot as plt
import seaborn as sns
palette = sns.color_palette('Paired', 10)

import numpy as np
import pandas as pd
# Pandas display options
pd.set_option('display.float_format', lambda x: '%.3f' % x)


#setting fontsize and style for all the plots
plt.style.use('fivethirtyeight')
plt.rcParams['font.size'] = 18
plt.rcParams['figure.figsize'] = (16,5)

get_ipython().run_line_magic('matplotlib', 'inline')
#plotting directly without requering the plot()

import warnings
warnings.filterwarnings(action="ignore") #ignoring most of warnings, cleaning up the notebook for better visualization

pd.set_option('display.max_columns', 500) #fixing the number of rows and columns to be displayed
pd.set_option('display.max_rows', 500)

print(os.listdir("../input")) #showing all the files in the ../input directory

# Set random seed 
randomseed = 42

# Any results you write to the current directory are saved as output. Kaggle message :D




train = pd.read_table('../input/train.tsv',low_memory=True)
test = pd.read_table('../input/test_stg2.tsv', low_memory=True)
print('Training set shape: {}'.format(train.shape))
print('Testing set shape: {}'.format(test.shape))
train.head()




fig = plt.figure(figsize=(16,5))
plt.subplot(2,1,1)
sns.distplot(train['price'])
plt.subplot(2,1,2)
sns.distplot(np.log1p(train['price']))




fig = plt.figure(figsize=(16,10))
plt.subplot(2,1,1)
sns.countplot(train['shipping'])
plt.subplot(2,1,2)
sns.countplot(train['item_condition_id'])




train.isnull().sum()




train.nunique()




# reference: BuryBuryZymon at https://www.kaggle.com/maheshdadhich/i-will-sell-everything-for-free-0-55
def split_cat(text):
    try: return text.split("/")
    except: return ("missing", "missing", "missing")

train['general_cat'], train['subcat_1'], train['subcat_2'] = zip(*train['category_name'].apply(lambda x: split_cat(x)))
test['general_cat'], test['subcat_1'], test['subcat_2'] = zip(*test['category_name'].apply(lambda x: split_cat(x)))


print('Training set shape: {}'.format(train.shape))
print('Testing set shape: {}'.format(test.shape))
train.head()




def imputing_nan_values(X):
    X.category_name.fillna(value="missing", inplace=True)
    X.brand_name.fillna(value="missing", inplace=True)
    X.item_description.fillna(value="missing", inplace=True)
    return (X)

train = imputing_nan_values(train)
test = imputing_nan_values(test)

train.isnull().sum()




fig = plt.figure(figsize=(16,5))
sns.countplot(train['general_cat'])
plt.xlabel('General Category',fontsize = 15,color='blue')
plt.ylabel('Count',fontsize = 15,color='blue')
plt.xticks(rotation=45)
plt.title('General Category/Count',fontsize = 20,color='blue')




train['brand_name'].value_counts()




train['has_brand'] = 1 #setting all the values to true as default
test['has_brand'] = 1
train['has_description'] = 1 #setting all the values to true as default
test['has_description'] = 1

train.loc[train['item_description'] == 'No description yet', 'has_description'] = 0
train.loc[train['item_description'] == 'No description yet', 'has_description'] = 0
train.loc[train['brand_name'] == 'missing', 'has_brand'] = 0
train.loc[train['brand_name'] == 'missing', 'has_brand'] = 0




sns.catplot(x='has_brand', y='price', data=train, aspect=1.5,alpha=0.8)
plt.xlabel('Has brand: 0 = missing', fontsize = 15,color='blue')
plt.ylabel('Price',fontsize = 15,color='blue')
plt.title('Has Brand X Price',fontsize = 20,color='blue')




sns.catplot(x='has_description', y='price', data=train, aspect=1.5,alpha=0.8)
plt.xlabel('has_description: 0 = No description', fontsize = 15,color='blue')
plt.ylabel('Price',fontsize = 15,color='blue')
plt.title('HAS DESCRIPTION X Price',fontsize = 20,color='blue')




# description related tf-idf features 
# I guess "No dscription present won't affact these features ... So, I am not removing them.
## https://www.kaggle.com/maheshdadhich/i-will-sell-everything-for-free-0-55
start = time.time()
tfidf_vec = TfidfVectorizer(stop_words='english', ngram_range=(1,1))
full_tfidf = tfidf_vec.fit_transform(train['item_description'].values.tolist() + test['item_description'].values.tolist())
train_tfidf = tfidf_vec.transform(train['item_description'].values.tolist())
test_tfidf = tfidf_vec.transform(train['item_description'].values.tolist())

n_comp = 40
svd_obj = TruncatedSVD(n_components=n_comp, algorithm='arpack')
svd_obj.fit(full_tfidf)
train_svd = pd.DataFrame(svd_obj.transform(train_tfidf))
test_svd = pd.DataFrame(svd_obj.transform(test_tfidf))
    
train_svd.columns = ['svd_item_'+str(i) for i in range(n_comp)]
test_svd.columns = ['svd_item_'+str(i) for i in range(n_comp)]
train = pd.concat([train, train_svd], axis=1)
test = pd.concat([test, test_svd], axis=1)
end = time.time()

print("time taken {}".format(end - start))
print('Training set shape: {}'.format(train.shape))
print('Testing set shape: {}'.format(test.shape))
gc.enable()
del tfidf_vec, full_tfidf,train_tfidf,test_tfidf,svd_obj,train_svd,test_svd
gc.collect()
train.head()




lb = LabelEncoder()
label_encoder_columns = ['brand_name','subcat_1','subcat_2']
for name in label_encoder_columns:  
    train[name] = lb.fit_transform(train[name])
    test[name] = lb.fit_transform(test[name])




train.head()




train = pd.get_dummies(train, columns=['general_cat','item_condition_id'])
test = pd.get_dummies(test, columns=['general_cat','item_condition_id'])

print('Training set shape: {}'.format(train.shape))
print('Testing set shape: {}'.format(test.shape))
train.head()




drop_columns = ['general_cat','item_description','category_name','item_condition_id','name',
                           'train_id', 'price']
features_to_be_used = [f for f in train.columns if f not in drop_columns]

train_labels = np.log1p(train['price'].values)
train = train.loc[:,features_to_be_used]
test = test.loc[:,features_to_be_used]




print('Training set shape: {}'.format(train.shape))
print('Testing set shape: {}'.format(test.shape))
train.head()




from sklearn.model_selection import train_test_split

train_final,train_validation, train_y, train_val_y  = train_test_split(train, train_labels,test_size=0.2, shuffle = True, random_state=randomseed)




print(train_final.shape, train_y.shape)




#let's create this function to make it easier and clean to fit the model and use the cross_val_score and obtain results
import time #implementing in this function the time spent on training the model
from sklearn.metrics import mean_squared_error
from catboost import CatBoostRegressor, Pool
import lightgbm as lgb
import xgboost as xgb
import eli5
import gc

def train_model(X_train, x_val, y, y_val, params=None, model_type='lgb', plot_feature_importance=False):
  
    evals_result={}
    
    
    if model_type == 'lgb':
        start = time.time()
        
        model = lgb.LGBMRegressor(**params, n_estimators = 15000, nthread = 4, n_jobs = -1)
        
        model.fit(X_train, y, eval_set=[(X_train, y), (x_val, y_val)], eval_metric='rmse', early_stopping_rounds=200,
                    verbose=50)
            
        y_pred_valid = model.predict(x_val, num_iteration=model.best_iteration_)
        
        end = time.time()
        
        #y_pred = model.predict(X_test, num_iteration=model.best_iteration_)
        
        print('RMSE validation data: {}'.format(np.sqrt(mean_squared_error(y_val,y_pred_valid))))
        
        
        if plot_feature_importance:
            # feature importance
            fig, ax = plt.subplots(figsize=(16,16))
            lgb.plot_importance(model, max_num_features=50, height=0.8,color='c', ax=ax)
            plt.title("LightGBM - Feature Importance", fontsize=14)
            
        print('Total time spent: {}'.format(end-start))
        return model
            
    if model_type == 'xgb':
        start = time.time()
        
        model = xgb.XGBRegressor(**params, nthread = 4, n_jobs = -1)

        model.fit(X_train, y, eval_metric="rmse", 
                      eval_set=[(X_train, y), (x_val, y_val)],verbose=20,
                      early_stopping_rounds=50)
        
        y_pred_valid = model.predict(x_val, ntree_limit=model.best_ntree_limit)
        
        end = time.time()

        print('RMSE validation data: {}'.format(np.sqrt(mean_squared_error(y_val,y_pred_valid))))
        
        print('Total time spent: {}'.format(end-start))
        return model
            
    if model_type == 'cat':
        start = time.time()
        model = CatBoostRegressor(eval_metric='RMSE', **params)
        model.fit(X_train, y, eval_set=(x_val, y_val), 
                  cat_features=[], use_best_model=True)

        y_pred_valid = model.predict(x_val)
        
        print('RMSE validation data: {}'.format(np.sqrt(mean_squared_error(y_val,y_pred_valid))))
        
        end = time.time()
        
        if plot_feature_importance:
            feature_score = pd.DataFrame(list(zip(X_train.dtypes.index, model.get_feature_importance(Pool(X_train, label=y, cat_features=[])))), columns=['Feature','Score'])
            feature_score = feature_score.sort_values(by='Score', kind='quicksort', na_position='last')
            feature_score.plot('Feature', 'Score', kind='barh', color='c', figsize=(16,16))
            plt.title("Catboost Feature Importance plot", fontsize = 14)
            plt.xlabel('')

        print('Total time spent: {}'.format(end-start))
        return model
        
    # Clean up memory
    gc.enable()
    del model, y_pred_valid, X_test,X_train,X_valid, y_pred, y_train, start, end,evals_result, x_val
    gc.collect()




params_cat = {
    'iterations': 1500,
    'max_ctr_complexity': 6,
    'random_seed': 42,
    'od_type': 'Iter',
    'od_wait': 100,
    'verbose': 50,
    'depth': 4
}

#cat_model = train_model(train_final.drop('tokens',axis=1),train_validation.drop('tokens',axis=1),train_y,train_val_y,params_cat,
                        #model_type='cat',plot_feature_importance=True)




params_lgb = {
        "objective" : "regression",
        "metric" : "rmse",
        "num_leaves" : 30,
        "min_child_weight" : 50,
        "learning_rate" : 0.1,
        "bagging_fraction" : 0.7,
        "feature_fraction" : 0.7,
        "bagging_frequency" : 5,
        "bagging_seed" : 42,
        'reg_aplha': 1,
        'reg_lambda': 0.001
}

lgb_model = train_model(train_final,train_validation,train_y,train_val_y,params_lgb,plot_feature_importance=True)
preds_lgb = lgb_model.predict(test)




submission = pd.read_csv('../input/sample_submission_stg2.csv')
submission['price'] = np.expm1(preds_lgb)
submission.to_csv("lgb_model.csv", index=False)

