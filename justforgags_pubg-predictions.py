#!/usr/bin/env python
# coding: utf-8

# In[ ]:


## This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.


# In[ ]:


def reduce_mem_usage(df):
    """ iterate through all the columns of a dataframe and modify the data type
        to reduce memory usage.        
    """
    start_mem = df.memory_usage().sum() / 1024**2
    print('Memory usage of dataframe is {:.2f} MB'.format(start_mem))
    
    for col in df.columns:
        col_type = df[col].dtype
        
        if col_type != object:
            c_min = df[col].min()
            c_max = df[col].max()
            if str(col_type)[:3] == 'int':
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
                elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                    df[col] = df[col].astype(np.int64)  
            else:
                if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:
                    df[col] = df[col].astype(np.float16)
                elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                    df[col] = df[col].astype(np.float32)
                else:
                    df[col] = df[col].astype(np.float64)
        else:
            df[col] = df[col].astype('category')

    end_mem = df.memory_usage().sum() / 1024**2
    print('Memory usage after optimization is: {:.2f} MB'.format(end_mem))
    print('Decreased by {:.1f}%'.format(100 * (start_mem - end_mem) / start_mem))
    
    return df


def import_data(file):
    """create a dataframe and optimize its memory usage"""
    df = pd.read_csv(file, parse_dates=True, keep_date_col=True)
    df = reduce_mem_usage(df)
    return df


# In[ ]:


print('train')
train = import_data('../input/train.csv')
print('test')
test = import_data('../input/test.csv')


# In[ ]:


# Data manipulation
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler

# Visualization
import matplotlib.pyplot as plt
import seaborn as sns
from collections import OrderedDict
# Set a few plotting defaults
get_ipython().run_line_magic('matplotlib', 'inline')
plt.style.use('fivethirtyeight')
plt.rcParams['font.size'] = 18
plt.rcParams['patch.edgecolor'] = 'k'
from scipy.stats import spearmanr
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import Imputer
from sklearn.metrics import f1_score, make_scorer
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_val_score
import lightgbm as lgb
from sklearn.model_selection import StratifiedKFold
import lightgbm as lgb
from IPython.display import display


# In[ ]:


train.head()


# In[ ]:


test.head()


# In[ ]:


train.info()


# In[ ]:


train.isnull().sum().sum()


# In[ ]:


test.isnull().sum()


# In[ ]:


final=pd.concat([train,test],keys=['Train','Test'])


# In[ ]:


final.loc['Train'].winPlacePerc.describe()


# In[ ]:


def find_bucket(x):
    if x>0.9 and x<=1.0:
        return "1"
    if x>0.8 and x<=0.9:
        return "2"
    if x>0.7 and x<=0.8:
        return "3"
    if x>0.6 and x<=0.7:
        return "4"
    if x>0.5 and x<=0.6:
        return "5"
    if x>0.4 and x<=0.5:
        return "6"
    if x>0.3 and x<=0.4:
        return "7"
    if x>0.2 and x<=0.3:
        return "8"
    if x>0.1 and x<=0.2:
        return "9"
    if x>0 and x<=0.1:
        return "10"
    else:
        return '0'
    


# In[ ]:


seri=final.loc['Train'].winPlacePerc.apply(lambda x:find_bucket(x))


# In[ ]:


final.loc[:,'Buckets']=seri 
final.loc['Train'].loc[:,'Buckets']=seri


# In[ ]:


final.head()


# In[ ]:


cols=[i for i in range(0,11)]
bucket_mapping={"1":"0.9-1.0","2":"0.8-0.9","3":"0.7-0.8",
                "4":"0.6-0.7","5":"0.5-0.6","6":"0.4-0.5",
                "7":"0.3-0.4","8":"0.2-0.3","9":"0.1-0.2",
                "10":"0.0-0.1","0":"O (ZERO)"}
                
final.loc['Train']["Buckets"].value_counts().plot.bar(figsize=(8,6),edgecolor='k',linewidth=2)
plt.xticks(cols,bucket_mapping.values(), rotation=60)
plt.ylabel("Number of people")
plt.xlabel("Bucket category")
plt.title("Number of people V/S Percentile they are in")


# In[ ]:


final_2=final.copy()
final_2.head()


# In[ ]:


'''
#we can group by numGroups too
#firstly checking if any match contains more than 100 players
matchId_group=final_2.groupby("matchId")["DBNOs","assists","boosts","damageDealt","headshotKills","heals","killPoints","killStreaks","kills","rideDistance","roadKills","swimDistance",
                           "vehicleDestroys","walkDistance","weaponsAcquired",
                           "winPoints","winPlacePerc","groupId"]
lis=matchId_group.apply(lambda x:len(x))
c=0
for i in lis:
    if i>100:
        c=c+1
        
print("Number matches with more than 100 players : {}".format(c))

'''
        


# In[ ]:


#17
'''
match=["killPlace","matchId","revives","teamKills","numGroups","maxPlace"]
indo=["DBNOs","assists","boosts","damageDealt","headshotKills"]
#sns.scatterplot(y='winPlacePerc',x='killPlace',data=final_2.loc['Train'])

plt.figure(figsize=(20,20))
plt.style.use('fivethirtyeight')
fig, ax = plt.subplots(nrows=5, ncols=1)
for i in range(0,5):
    #print(i)
    ax=plt.subplot(5,1,i+1)
    sns.scatterplot(y='winPlacePerc',x=indo[i],data=final_2.loc['Train'],ax=ax)
plt.subplots_adjust(top=2)'''


# In[ ]:


#adding new feature

#headshot kills
final_2['perc_kill_headshot']=pd.Series()
final_2['perc_kill_headshot']=final['headshotKills']/final['kills']

                                                              
#assist kill

final_2['perc_kill_assist']=pd.Series()
final_2['perc_kill_assist']=final['assists']/final['kills']

final_2['perc_kill_assist']

#road kills

final_2['perc_kill_road']=pd.Series()
final_2['perc_kill_road']=final['roadKills']/final['kills']

# percent of total distance on ride

final_2['perc_dist_ride']=pd.Series()
final_2['perc_dist_ride']=final['rideDistance']/(final['swimDistance']+final['rideDistance']+final['walkDistance'])

#percent swim

final_2['perc_dist_swim']=pd.Series()
final_2['perc_dist_swim']=final['swimDistance']/(final['swimDistance']+final['rideDistance']+final['walkDistance'])

#percent walk

final_2['perc_dist_walk']=pd.Series()
final_2['perc_dist_walk']=final['walkDistance']/(final['swimDistance']+final['rideDistance']+final['walkDistance'])


# In[ ]:



ind=["DBNOs","assists","boosts","damageDealt","headshotKills","heals","killPoints","killStreaks","kills","rideDistance","roadKills","swimDistance","vehicleDestroys","walkDistance",
     "weaponsAcquired","winPoints","perc_dist_ride","perc_dist_swim","perc_dist_walk",'longestKill','perc_kill_headshot','perc_kill_road','perc_kill_assist']

match=["killPlace","matchId","revives","teamKills","numGroups","maxPlace"]

dict_means={}
dict_std={}
for i in ind:
    dict_means[i]=np.mean(final_2[i])
    dict_std[i]=np.std(final_2[i])
for i in match:
    dict_means[i]=np.mean(final_2[i])
    dict_std[i]=np.std(final_2[i])    
#print(dict_means)
#print(dict_std)

#z_score=x-mean/std
final_2.shape

for i in dict_means.keys():
    print(i)
    final_2[i]=final_2[i]-dict_means[i]
    final_2[i]=final_2[i]/dict_std[i]
train=final_2.loc['Train']
test=final_2.loc['Test']


# In[ ]:


train=train.fillna(0)
train.head()


# In[ ]:


index1=[]
index2=[]
for i in dict_means.keys():
    print(i)
    #print(train[i].sort_values()[0])
    index1=train[train[i]>3].index
    index2=train[train[i]<-3].index
    train.drop(index1,inplace=True)
    #train.drop(index2,inplace=True)
    print(index1)
    print(index2)
    


# In[ ]:


test=test.fillna(0)
final_2=pd.concat([train,test],keys=['Train','Test'])


# In[ ]:


sns.scatterplot(y='winPlacePerc',x="perc_dist_swim",data=final_2.loc['Train'])


# In[ ]:


#checking correlation heat map for co related features
corr_matrix=final_2.loc['Train'].corr()
plt.figure(figsize=(30,37))
sns.heatmap(corr_matrix,annot=True, cmap = plt.cm.autumn_r, fmt='.3f');


# In[ ]:


upper=corr_matrix.where(np.triu(np.ones(corr_matrix.shape),k=1).astype(np.bool))
to_drop=[column for column in upper.columns if any(upper[column]>0.95)]
to_drop


# In[ ]:


final_2=final_2.drop(to_drop,axis=1)


# In[ ]:


final_2.loc['Test'].isnull().sum()


# In[ ]:



#dividing data
feats=[i for i in final_2.columns if i != 'winPlacePerc']
train_y=final_2.loc["Train"].winPlacePerc
train_X=final_2.loc["Train"][feats]

test_data=final_2.loc["Test"][feats]

#train_set_d=pipeline.fit_transform(train_X)
#test_set_d=pipeline.transform(test_data)


# In[ ]:


train_set_d=train_X.copy()
test_set_d=test_data.copy()

train_set_d.Id=final.loc['Train'].Id
train_set_d.matchId=final.loc['Train'].matchId
train_set_d.groupId=final.loc['Train'].groupId


test_set_d.Id=final.loc['Test'].Id
test_set_d.matchId=final.loc['Test'].matchId
test_set_d.groupId=final.loc['Test'].groupId


# In[ ]:


train_copy_1=train_set_d.copy()
train_copy_2=train_set_d.copy()
test_copy_1=test_set_d.copy()
test_copy_2=test_set_d.copy()


# In[ ]:



scorer=make_scorer(f1_score,greater_is_better=True,average='macro')


# In[ ]:


from sklearn.model_selection import StratifiedKFold
import lightgbm as lgb
from IPython.display import display

def model_gbm(features, labels, test_features, test_ids, 
              nfolds = 5, return_preds = True, hyp = None):
    """Model using the GBM and cross validation.
       Trains with early stopping on each fold.
       Hyperparameters probably need to be tuned."""
    
    feature_names = list(features.columns)
    print(feature_names)

    # Option for user specified hyperparameters
    if hyp is not None:
        # Using early stopping so do not need number of esimators
        if 'n_estimators' in hyp:
            del hyp['n_estimators']
        params = hyp
    
    else:
        # Model hyperparameters
        params = {'boosting_type': 'dart', 
                  'colsample_bytree': 0.88, 
                  'learning_rate': 0.028, 
                   'min_child_samples': 10, 
                   'num_leaves': 36, 'reg_alpha': 0.76, 
                   'reg_lambda': 0.43, 
                   'subsample_for_bin': 40000, 
                   'subsample': 0.54, 
                   'class_weight': 'balanced'}
    
    # Build the model
    model = lgb.LGBMRegressor(**params, objective = 'multiclass', 
                               n_jobs = -1, n_estimators = 10000,
                               random_state = 10)
    
    # Using stratified kfold cross validation
    strkfold = StratifiedKFold(n_splits = nfolds, shuffle = True)
    
    # Hold all the predictions from each fold
    predictions = pd.DataFrame()
    importances = np.zeros(len(feature_names))# to the size of features present
    
    # Convert to arrays for indexing
    features = np.array(features)
    print(features)
    test_features = np.array(test_features)
    labels = np.array(labels).reshape((-1 ))
    
    valid_scores = []
    modeld=lgb.LGBMRegressor(**params, objective = 'multiclass', 
                               n_jobs = -1, n_estimators = 10000,
                               random_state = 10)
    # Iterate through the folds
    for i, (train_indices, valid_indices) in enumerate(strkfold.split(features, labels)):
        
        # Dataframe for fold predictions
        fold_predictions = pd.DataFrame()
        
        # Training and validation data
        X_train = features[train_indices]
        X_valid = features[valid_indices]
        y_train = labels[train_indices]
        y_valid = labels[valid_indices]
        modeld=model
        # Train with early stopping
        model.fit(X_train, y_train, early_stopping_rounds = 100, 
                  eval_metric = macro_f1_score,
                  eval_set = [(X_train, y_train), (X_valid, y_valid)],
                  eval_names = ['train', 'valid'],#(X_train, y_train) >>>>>> train and (X_valid, y_valid)>>>>>>>>>>>
                  verbose = 200)
        display(model)
        display(model.best_score_)
        # Record the validation fold score
        valid_scores.append(model.best_score_['valid']['macro_f1'])
        
        # Make predictions from the fold as probabilities
        fold_predictions = model.predict(test_features) #Returns prediction probabilities for each class of each output.
        display(fold_predictions)
        # Record each prediction for each class as a separate column
        for j in range(4):
            fold_predictions[(j + 1)] = fold_predictions[:, j]
        display(fold_predictions)    
        # Add needed information for predictions 
        fold_predictions['Id'] = test_ids
        fold_predictions['fold'] = (i+1)
        
        # Add the predictions as new rows to the existing predictions
        predictions = predictions.append(fold_predictions)
        
        # Feature importances
        importances += model.feature_importances_ / nfolds   
        display(model.feature_importances_)
        display(importances)
        # Display fold information
        display(f'Fold {i + 1}, Validation Score: {round(valid_scores[i], 5)}, Estimators Trained: {model.best_iteration_}')

    # Feature importances dataframe
    feature_importances = pd.DataFrame({'feature': feature_names,
                                        'importance': importances})
    display("feature_importances")
    display(feature_importances)
    valid_scores = np.array(valid_scores)
    display(f'{nfolds} cross validation score: {round(valid_scores.mean(), 5)} with std: {round(valid_scores.std(), 5)}.')
    display(valid_scores)
    # If we want to examine predictions don't average over folds
    if return_preds:
        predictions['Target'] = predictions[[1, 2, 3, 4]].idxmax(axis = 1)
        predictions['confidence'] = predictions[[1, 2, 3, 4]].max(axis = 1)
        return predictions, feature_importances
    


# In[ ]:


test_ids=test_data.Id
train_set_d=train_set_d.drop(["Id",'groupId','matchId'],axis=1)
test_set_d=test_set_d.drop(["Id",'groupId','matchId'],axis=1)


# In[ ]:


get_ipython().run_cell_magic('capture', '--no-display', '%%capture --no-display\n#predictions, gbm_fi = model_gbm(train_set_d, train_y, test_set_d, test_ids, return_preds=True)')


# In[ ]:


params = {'boosting_type': 'gbdt', 
                 'colsample_bytree': 1, 
                 'learning_rate': 0.028, 
                  'min_child_samples': 10, 
                  'num_leaves': 36, 'reg_alpha': 0.76, 
                  'reg_lambda': 0.43, 
                  'subsample_for_bin': 40000, 
                  'subsample': 0.54, 
                  'class_weight': 'balanced'}

model = lgb.LGBMRegressor(**params,n_jobs = -1, n_estimators = 10000,random_state = 10)
# Using stratified kfold cross validation


# In[ ]:


train_copy_1=train_copy_1.drop(['Buckets'],axis=1)
train_copy_1=train_copy_1.drop(['Id','matchId','groupId'],axis=1)


# In[ ]:


model.fit(train_copy_1,train_y)


# In[ ]:


test_set_d=test_set_d.drop(["Buckets"],axis=1)
predictions=model.predict(test_set_d)


# In[ ]:


test_set


# In[ ]:


submission = import_data('../input/sample_submission.csv')
submission.Id=final.loc['Test'].Id
submission.winPlacePerc=predictions


# In[ ]:


submission.to_csv('submission_2.csv', index=False)


# In[ ]:




