#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd 
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from math import pi
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')

import lightgbm as lgb
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import KFold

import warnings
warnings.filterwarnings("ignore")

# Print all rows and columns
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
IS_LOCAL = False

import os

if(IS_LOCAL):
    PATH="../input/santander-value-prediction-challenge"
else:
    PATH="../input"
print(os.listdir(PATH))


# In[ ]:


train_df = pd.read_csv(PATH+"/train.csv")
test_df = pd.read_csv(PATH+"/test.csv")


# In[ ]:


print("Santander Value Prediction Challenge train -  rows:",train_df.shape[0]," columns:", train_df.shape[1])


# In[ ]:


print("Santander Value Prediction Challenge test -  rows:",test_df.shape[0]," columns:", test_df.shape[1])


# In[ ]:


train_df.head()


# In[ ]:


test_df.head()


# In[ ]:


def check_nulls(df):
    nulls = df.isnull().sum(axis=0).reset_index()
    nulls.columns = ['column', 'missing']
    nulls = nulls[nulls['missing']>0]
    nulls = nulls.sort_values(by='missing')
    return nulls    


check_nulls(train_df)


# In[ ]:


check_nulls(test_df)


# In[ ]:


def check_sparsity(df):
    non_zeros = (df.ne(0).sum(axis=1)).sum()
    total = df.shape[1]*df.shape[0]
    zeros = total - non_zeros
    sparsity = round(zeros / total * 100,2)
    density = round(non_zeros / total * 100,2)

    print(" Total:",total,"\n Zeros:", zeros, "\n Sparsity [%]: ", sparsity, "\n Density [%]: ", density)
    return density

d1 = check_sparsity(train_df)


# In[ ]:


d2 = check_sparsity(test_df)


# In[ ]:


dtype_df = train_df.dtypes.reset_index()
dtype_df.columns = ["Count", "Column Type"]
dtype_df.groupby("Column Type").aggregate('count').reset_index()


# In[ ]:


data = []
for feature in train_df.columns:
    # Defining the role
    if feature == 'target':
        use = 'target'
    elif feature == 'ID':
        use = 'id'
    else:
        use = 'input'
         
        
    # Initialize keep to True for all variables except for `ID`
    keep = True
    if feature == 'ID':
        keep = False
    
    # Defining the data type 
    dtype = train_df[feature].dtype
    
    
    
    # Creating a Dict that contains all the metadata for the variable
    feature_dictionary = {
        'varname': feature,
        'use': use,
        'keep': keep,
        'dtype': dtype,
    }
    data.append(feature_dictionary)
    
# Create the metadata
metadata = pd.DataFrame(data, columns=['varname', 'use', 'keep', 'dtype'])
metadata.set_index('varname', inplace=True)

# Sample the metadata
metadata.head(10)


# In[ ]:


pd.DataFrame({'count' : metadata.groupby(['dtype'])['dtype'].size()}).reset_index()


# In[ ]:


int_data = []
var = metadata[(metadata.dtype == 'int64') & (metadata.use == 'input')].index
d3 = check_sparsity(train_df[var])


# In[ ]:


d4 = check_sparsity(test_df[var])


# In[ ]:


var = metadata[(metadata.dtype == 'float64') & (metadata.use == 'input')].index
d5 = check_sparsity(train_df[var])


# In[ ]:


d6 = check_sparsity(test_df[var])


# In[ ]:


data = {'Dataset': ['Train', 'Test'], 'All': [d1, d2], 'Integer': [d3,d4], 'Float': [d5,d6]}
    
density_data = pd.DataFrame(data)
density_data.set_index('Dataset', inplace=True)
density_data


# In[ ]:


# check constant columns
colsConstant = []
columnsList = [x for x in train_df.columns if not x in ['ID','target']]

for col in columnsList:
    if train_df[col].std() == 0: 
        colsConstant.append(col)
print("There are", len(colsConstant), "constant columns in the train set.")


# In[ ]:


metadata['keep'].loc[colsConstant] = False


# In[ ]:


# Plot distribution of one feature
def plot_distribution(df,feature,color):
    plt.figure(figsize=(10,6))
    plt.title("Distribution of %s" % feature)
    sns.distplot(df[feature].dropna(),color=color, kde=True,bins=100)
    plt.show()   
    
plot_distribution(train_df, "target", "blue")


# In[ ]:


def plot_log_distribution(df,feature,color):
    plt.figure(figsize=(10,6))
    plt.title("Distribution of %s" % feature)
    sns.distplot(np.log1p(df[feature]).dropna(),color=color, kde=True,bins=100)
    plt.title("Distribution of log(target)")
    plt.show()   

plot_log_distribution(train_df, "target", "green")  


# In[ ]:


non_zeros = (train_df.ne(0).sum(axis=1))

plt.figure(figsize=(10,6))
plt.title("Distribution of log(number of non-zeros per row) - train set")
sns.distplot(np.log1p(non_zeros),color="red", kde=True,bins=100)
plt.show()


# In[ ]:


non_zeros = (test_df.ne(0).sum(axis=1))

plt.figure(figsize=(10,6))
plt.title("Distribution of log(number of non-zeros per row) - test set")
sns.distplot(np.log1p(non_zeros),color="magenta", kde=True,bins=100)
plt.show()


# In[ ]:


var = metadata[(metadata.dtype == 'float64') & (metadata.use == 'input')].index
non_zeros = (train_df[var].ne(0).sum(axis=1))

plt.figure(figsize=(10,6))
plt.title("Distribution of log(number of non-zeros per row) - floats only - train set")
sns.distplot(np.log1p(non_zeros),color="green", kde=True,bins=100)
plt.show()


# In[ ]:


non_zeros = (test_df[var].ne(0).sum(axis=1))

plt.figure(figsize=(10,6))
plt.title("Distribution of log(number of non-zeros per row) - floats only - test set")
sns.distplot(np.log1p(non_zeros),color="blue", kde=True,bins=100)
plt.show()


# In[ ]:


var = metadata[(metadata.dtype == 'int64') & (metadata.use == 'input')].index
non_zeros = (train_df[var].ne(0).sum(axis=1))

plt.figure(figsize=(10,6))
plt.title("Distribution of log(number of non-zeros per row) - integers only -  train set")
sns.distplot(np.log1p(non_zeros),color="yellow", kde=True,bins=100)
plt.show()


# In[ ]:


non_zeros = (test_df[var].ne(0).sum(axis=1))

plt.figure(figsize=(10,6))
plt.title("Distribution of log(number of non-zeros per row) - integers only - train set")
sns.distplot(np.log1p(non_zeros),color="cyan", kde=True,bins=100)
plt.show()


# In[ ]:


non_zeros = (train_df.ne(0).sum(axis=0))

plt.figure(figsize=(10,6))
plt.title("Distribution of log(number of non-zeros per column) - train set")
sns.distplot(np.log1p(non_zeros),color="darkblue", kde=True,bins=100)
plt.show()


# In[ ]:


non_zeros = (test_df.ne(0).sum(axis=0))

plt.figure(figsize=(10,6))
plt.title("Distribution of log(number of non-zeros per column) - test set")
sns.distplot(np.log1p(non_zeros),color="darkgreen", kde=True,bins=100)
plt.show()


# In[ ]:


var = metadata[(metadata.dtype == 'float64') & (metadata.use == 'input')].index
val = train_df[var].sum()

fig, (ax1, ax2) = plt.subplots(ncols=2, figsize=(12,6))
sns.boxplot(val, palette="Blues",  showfliers=False,ax=ax1)
sns.boxplot(val, palette="Greens",  showfliers=True,ax=ax2)
plt.show();


# In[ ]:


var = metadata[(metadata.dtype == 'int64') & (metadata.use == 'input')].index
val = train_df[var].sum()

fig, (ax1, ax2) = plt.subplots(ncols=2, figsize=(12,6))
sns.boxplot(val, palette="Reds",  showfliers=False,ax=ax1)
sns.boxplot(val, palette="Blues",  showfliers=True,ax=ax2)
plt.show();


# In[ ]:


labels = []
values = []
for col in train_df.columns:
    if col not in ["ID", "target"]:
        labels.append(col)
        values.append(np.corrcoef(train_df[col].values, train_df["target"].values)[0,1])
corr_df = pd.DataFrame({'columns_labels':labels, 'corr_values':values})
corr_df = corr_df.sort_values(by='corr_values')
 
corr_df = corr_df[(corr_df['corr_values']>0.25) | (corr_df['corr_values']<-0.25)]
ind = np.arange(corr_df.shape[0])
width = 0.9
fig, ax = plt.subplots(figsize=(10,6))
rects = ax.barh(ind, np.array(corr_df.corr_values.values), color='gold')
ax.set_yticks(ind)
ax.set_yticklabels(corr_df.columns_labels.values, rotation='horizontal')
ax.set_xlabel("Correlation coefficient")
ax.set_title("Correlation coefficient of the variables")
plt.show()


# In[ ]:


temp_df = train_df[corr_df.columns_labels.tolist()]
corrmat = temp_df.corr(method='pearson')
f, ax = plt.subplots(figsize=(12, 12))

# Draw the heatmap using seaborn
sns.heatmap(corrmat, vmax=1., square=True, cmap="YlOrRd")
plt.title("Important variables correlation map", fontsize=15)
plt.show()


# In[ ]:


corrmat


# In[ ]:


var = temp_df.columns.values

i = 0
sns.set_style('whitegrid')
plt.figure()
fig, ax = plt.subplots(5,4,figsize=(12,15))

for feature in var:
    i += 1
    plt.subplot(5,4,i)
    sns.kdeplot(train_df[feature], bw=0.5,label="train")
    sns.kdeplot(test_df[feature], bw=0.5,label="test")
    plt.xlabel(feature, fontsize=12)
    locs, labels = plt.xticks()
    plt.tick_params(axis='both', which='major', labelsize=12)
plt.show();


# In[ ]:


sns.set_style('whitegrid')
plt.figure()
s = sns.lmplot(x='429687d5a', y='e4159c59e',data=train_df, fit_reg=True,scatter_kws={'s':2})
s = sns.lmplot(x='6b119d8ce', y='e8d9394a0',data=train_df, fit_reg=True,scatter_kws={'s':2})
s = sns.lmplot(x='cbbc9c431', y='f296082ec',data=train_df, fit_reg=True,scatter_kws={'s':2})
s = sns.lmplot(x='cbbc9c431', y='51707c671',data=train_df, fit_reg=True,scatter_kws={'s':2})
plt.show()


# In[ ]:


var = metadata[(metadata.keep == False) & (metadata.use == 'input')].index
train_df.drop(var, axis=1, inplace=True)  
test_df.drop(var, axis=1, inplace=True)  


# In[ ]:


print("Santander Value Prediction Challenge train -  rows:",train_df.shape[0]," columns:", train_df.shape[1])
print("Santander Value Prediction Challenge test -  rows:",test_df.shape[0]," columns:", test_df.shape[1])


# In[ ]:


# Replace 0 with NAs
train_df.replace(0, np.nan, inplace=True)
test_df.replace(0, np.nan, inplace=True)


# In[ ]:


all_features = [f for f in train_df.columns if f not in ['target', 'ID']]
for df in [train_df, test_df]:
    df['nans'] = df[all_features].isnull().sum(axis=1)
    # All of the stats will be computed without the 0s 
    df['median'] = df[all_features].median(axis=1)
    df['mean'] = df[all_features].mean(axis=1)
    df['sum'] = df[all_features].sum(axis=1)
    df['std'] = df[all_features].std(axis=1)
    df['kurtosis'] = df[all_features].kurtosis(axis=1)


# In[ ]:


features = all_features + ['nans', 'median', 'mean', 'sum', 'std', 'kurtosis']


# In[ ]:


print("Santander Value Prediction Challenge train -  rows:",train_df.shape[0]," columns:", train_df.shape[1])
print("Santander Value Prediction Challenge test -  rows:",test_df.shape[0]," columns:", test_df.shape[1])


# In[ ]:


# Create folds
folds = KFold(n_splits=5, shuffle=True, random_state=1)


# Convert to lightgbm Dataset
dtrain = lgb.Dataset(data=train_df[features], label=np.log1p(train_df['target']), free_raw_data=False)
# Construct dataset so that we can use slice()
dtrain.construct()
# Init predictions
sub_preds = np.zeros(test_df.shape[0])
oof_preds = np.zeros(train_df.shape[0])


# In[ ]:


lgb_params = {
    'objective': 'regression',
    'num_leaves': 100,
    'subsample': 0.8,
    'colsample_bytree': 0.75,
    'verbose': -1,
    'seed': 2018,
    'boosting_type': 'gbdt',
    'max_depth': 10,
    'learning_rate': 0.04,
    'metric': 'l2',
}


# In[ ]:


# Run KFold
for trn_idx, val_idx in folds.split(train_df):
    # Train lightgbm
    clf = lgb.train(
        params=lgb_params,
        train_set=dtrain.subset(trn_idx),
        valid_sets=dtrain.subset(val_idx),
        num_boost_round=10000, 
        early_stopping_rounds=100,
        verbose_eval=50
    )
    # Predict Out Of Fold and Test targets
    # Using lgb.train, predict will automatically select the best round for prediction
    oof_preds[val_idx] = clf.predict(dtrain.data.iloc[val_idx])
    sub_preds += clf.predict(test_df[features]) / folds.n_splits
    # Display current fold score
    print('Current fold score : %9.6f' % mean_squared_error(np.log1p(train_df['target'].iloc[val_idx]), 
                             oof_preds[val_idx]) ** .5)
    
# Display Full OOF score (square root of a sum is not the sum of square roots)
print('Full Out-Of-Fold score : %9.6f' 
      % (mean_squared_error(np.log1p(train_df['target']), oof_preds) ** .5))


# In[ ]:


fig, ax = plt.subplots(figsize=(14,10))
lgb.plot_importance(clf, max_num_features=50, height=0.8,color="tomato",ax=ax)
plt.show()


# In[ ]:


sub = test_df[['ID']].copy()
sub['target'] = np.expm1(sub_preds)
sub[['ID', 'target']].to_csv('submission.csv', index=False)

