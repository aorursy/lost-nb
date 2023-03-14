#!/usr/bin/env python
# coding: utf-8

# In[1]:


import matplotlib.pyplot as plt 
import time
import numpy as np
import seaborn as sns 
import time


# In[2]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import make_scorer, mean_squared_error
from sklearn.model_selection import cross_val_predict , KFold, cross_val_score
from mlxtend.plotting import plot_sequential_feature_selection as plot_sfs
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from mlxtend.feature_selection import SequentialFeatureSelector as SFS
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import AdaBoostRegressor
from math import log1p
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import SelectFromModel
from sklearn.metrics import accuracy_score
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory
from datetime import datetime
import os
print(os.listdir("../input"))
from xgboost import plot_importance

# Any results you write to the current directory are saved as output.


# In[3]:


#import sys
#sys.path.append('../input/feature-selector/')
#from feature_selector import FeatureSelector


# In[4]:


train= pd.read_csv("../input/train.csv", parse_dates=["first_active_month"])
test = pd.read_csv("../input/test.csv", parse_dates=["first_active_month"])
print("{} observations and {} features in train set.".format(train.shape[0],train.shape[1]))
print("{} observations and {} features in test set.".format(test.shape[0],test.shape[1]))


# In[5]:


print(len(list(test.keys())))
print(len(list(train.keys())))


# In[6]:


history = pd.read_csv("../input/historical_transactions.csv", low_memory=True)


# In[7]:


#history.head(n=5)


# In[8]:


#fig, ax = plt.subplots(figsize=(12, 3))
#sns.boxplot(x='target', data=train)


# In[9]:


#fig, ax = plt.subplots(figsize=(16, 5))
#sns.distplot(train.target, ax=ax)


# In[10]:


train['feature_1'].unique()


# In[11]:


def missing_data_function(frame):

    total = frame.isnull().sum().sort_values(ascending=False)
    percent = (frame.isnull().sum()*100 / frame.isnull().count()).sort_values(ascending=False)
    missing_data = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])
    return missing_data


# In[12]:


def reduce_mem_usage_func(df):
    """ Memory saving function credit to https://www.kaggle.com/gemartin/load-data-reduce-memory-usage
        iterate through all the columns of a dataframe and modify the data type
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

    end_mem = df.memory_usage().sum() / 1024**2
    print('Memory usage after optimization is: {:.2f} MB'.format(end_mem))
    print('Decreased by {:.1f}%'.format(100 * (start_mem - end_mem) / start_mem))
    return df


# In[13]:


history=reduce_mem_usage_func(history)


# In[14]:


history.reset_index(inplace=True)


# In[15]:


new_transactions = pd.read_csv('../input/new_merchant_transactions.csv',  low_memory=True)


# In[16]:


new_transactions.head(n=5)


# In[17]:


new_transactions=reduce_mem_usage_func(new_transactions)


# In[18]:


#time.sleep(30)


# In[19]:


new_transactions.reset_index(inplace=True)


# In[20]:


#history.head()


# In[21]:


def merge_train_test(train, test , df ):
    
    train=pd.merge(left=train , right=df, how = 'left', on ='card_id')
    test=pd.merge(left=test , right=df, how = 'left', on ='card_id')
    return train, test
    


# In[22]:


history['purchase_date'] = pd.to_datetime(history['purchase_date'])


# In[23]:


new_transactions['purchase_date'] = pd.to_datetime(new_transactions['purchase_date'])


# In[24]:


def features(df):
    df['feature_sum'] = df['feature_1'] + df['feature_2'] + df['feature_3']
    df['feature_mean'] = df['feature_sum']/3
    df['feature_max'] = df[['feature_1', 'feature_2', 'feature_3']].max(axis=1)
    df['feature_min'] = df[['feature_1', 'feature_2', 'feature_3']].min(axis=1)
    df['feature_var'] = df[['feature_1', 'feature_2', 'feature_3']].std(axis=1)
    return df 



    
    
    
    


# In[25]:


train=features(train)


# In[26]:


test=features(test)


# In[27]:


def deal_missing(df):
    df['category_2'].fillna(1.0,inplace=True)
    df['category_3'].fillna('A',inplace=True)
    df['merchant_id'].fillna('M_ID_00a6ca8a8a',inplace=True)
    df['installments'].replace(-1, np.nan,inplace=True)
    df['installments'].replace(999, np.nan,inplace=True)
    df['purchase_amount'] = df['purchase_amount'].apply(lambda x: min(x, 0.8))
    return df


# In[28]:


history=deal_missing(history)


# In[29]:


new_transactions=deal_missing(new_transactions)


# In[30]:


def mapping(df):
    df['authorized_flag'] = df['authorized_flag'].map({'Y': 1, 'N': 0}).astype('int8')
    df['category_1'] = df['category_1'].map({'Y': 1, 'N': 0}).astype('int8')
    df['category_3'] = df['category_3'].map({'A':0, 'B':1, 'C':2}).astype('int8')
    return df


# In[31]:


history=mapping(history)


# In[32]:


new_transactions=mapping(new_transactions)


# In[33]:


def new_features(df):
    df['month'] = df['purchase_date'].dt.month.astype('int8')
    df['day'] = df['purchase_date'].dt.day.astype('int8')
    df['hour'] = df['purchase_date'].dt.hour.astype('int8')
    df['weekofyear'] = df['purchase_date'].dt.weekofyear.astype('int8')
    df['weekday'] = df['purchase_date'].dt.weekday.astype('int8')
    df['weekend'] = (df['purchase_date'].dt.weekday >=5).astype('int8')
    time.sleep(60)
    df['price'] = df['purchase_amount'] / (1+df['installments'])
    df['month_diff'] = (((datetime.today() - df['purchase_date']).dt.days)//30).astype('int8')
    df['month_diff'] += history['month_lag']
    time.sleep(60)
    df['Christmas_Day_2017']=(pd.to_datetime('2017-12-25')-df['purchase_date']).dt.days.apply(lambda x: x if x > 0 and x < 100 else 0).astype('int8')
    df['Mothers_Day_2017']=(pd.to_datetime('2017-06-04')-df['purchase_date']).dt.days.apply(lambda x: x if x > 0 and x < 100 else 0).astype('int8')
    df['Valentine_Day_2017']=(pd.to_datetime('2017-06-12')-df['purchase_date']).dt.days.apply(lambda x: x if x > 0 and x < 100 else 0).astype('int8')
    df['Black_Friday_2017']=(pd.to_datetime('2017-11-24') - df['purchase_date']).dt.days.apply(lambda x: x if x > 0 and x < 100 else 0).astype('int8')
    df['Mothers_Day_2018']=(pd.to_datetime('2018-05-13')-df['purchase_date']).dt.days.apply(lambda x: x if x > 0 and x < 100 else 0).astype('int8')
    time.sleep(60)
    df['activity_100_days']=(df['purchase_date'].max()-df['purchase_date']).dt.days.apply(lambda x: 1 if x > 0 and x < 100 else 0).astype('int8')
    df['duration'] = df['purchase_amount']*df['month_diff']
    df['amount_month_ratio'] = df['purchase_amount']/df['month_diff']
    df['price'] = df['purchase_amount'] / df['installments']

    return df 
#


# In[34]:


history=new_features(history)


# In[35]:


#history=reduce_mem_usage_func(history)


# In[36]:


history.head(n=5)


# In[37]:


new_transactions=new_features(new_transactions)


# In[38]:


for col in ['category_2','category_3']:
    history[col+'_mean'] = history.groupby([col])['purchase_amount'].transform('mean')
    new_transactions[col+'_mean'] = new_transactions.groupby([col])['purchase_amount'].transform('mean')


# In[39]:


def aggregation(frame,name):
    agg_func = {
        'subsector_id':['nunique']
        ,'merchant_id':['nunique']
        ,'merchant_category_id':['nunique']
        ,'month':['nunique', 'mean', 'min', 'max']
        ,'hour':['nunique', 'mean', 'min', 'max']
        ,'weekofyear':['nunique', 'mean', 'min', 'max']
        ,'weekday':['nunique', 'mean', 'min', 'max']
        ,'day':['nunique', 'mean', 'min', 'max']
        ,'purchase_amount': ['sum','max','min','mean','var','skew']
        ,'installments' :['sum','max','mean','var','skew']
        ,'purchase_date' : ['max','min']
        ,'month_lag' :['max','min','mean','var','skew']
        ,'month_diff':['mean','var','skew']
        ,'weekend':['mean']
        ,'month':['mean', 'min', 'max']
        ,'weekday':['mean', 'min', 'max']
        ,'category_1' :['mean','std']
        ,'category_2': ['mean','std']
        ,'category_3': ['mean','std']
        ,'card_id':['size','count']
        ,'price': ['mean','max','min','var']
        ,'Christmas_Day_2017': ['mean']
        ,'Black_Friday_2017': ['mean']
        ,'Mothers_Day_2018':['mean']
        ,'activity_100_days':['mean']
        ,'category_2_mean':['mean']
        ,'category_3_mean':['mean']
        ,'duration' : ['mean','min','max','var','skew']
        ,'amount_month_ratio': ['mean','min','max','var','skew']
        }
    agg_new_trans = frame.groupby(['card_id']).agg(agg_func)
    agg_new_trans.columns = [str(name) + '_'.join(col).strip() 
                           for col in agg_new_trans.columns.values]
    agg_new_trans.reset_index(inplace=True)
    return agg_new_trans
    
    



# In[40]:


time.sleep(60)


# In[41]:


#df = aggregation(history,'history_')


# In[42]:


train , test = merge_train_test(train, test , aggregation(history,'history_') )


# In[43]:


time.sleep(30)


# In[44]:


time.sleep(20)


# In[45]:


train , test = merge_train_test(train, test ,aggregation(new_transactions,'new_') )


# In[46]:


new_transactions.head(n=10)


# In[47]:


new_transactions.keys()


# In[48]:





# In[48]:





# In[48]:


#train=train.drop(columns=['amount_mean_x', 'amount_std_x', 'amount_max_x','amount_min_x', 'amount_sum_x','amount_mean_y', 'amount_std_y', 'amount_max_y',
                         # 'amount_min_y', 'amount_sum_y'])


# In[49]:


train.keys()


# In[50]:





# In[50]:


#istory['installments']=history['installments'].replace({-1:0,999:0}, inplace=True)


# In[51]:


def shopping_days(df, name):
    days_of_shopping=df.groupby('card_id')['purchase_date'].max()-df.groupby('card_id')['purchase_date'].min()
    days_of_shopping=days_of_shopping.reset_index()
    days_of_shopping.columns = ['card_id',str(name)+'_'+'shopping_days']
    days_of_shopping[str(name)+'_'+'shopping_days']=days_of_shopping[str(name)+'_'+'shopping_days'].dt.days
    days_of_shopping.reset_index(inplace=True)
    return days_of_shopping


# In[52]:


history_shopping=shopping_days(history,'history')


# In[53]:


train , test = merge_train_test(train, test , history_shopping )


# In[54]:


new_shopping=shopping_days(new_transactions, 'new')


# In[55]:


new_shopping.head(n=10)


# In[56]:


train , test =merge_train_test(train, test , new_shopping )


# In[57]:



    


# In[57]:


#history['day_of_week']=history['purchase_date'].dt.weekday
#history['day_of_month']=history['purchase_date'].dt.day
#history['month_of_year']=history['purchase_date'].dt.month


# In[58]:


#most_frequent_day_of_week=history.groupby('card_id')['day_of_week'].agg(mode).reset_index(name='most_frequent_day_of_week')


# In[59]:


#train , test = merge_train_test(train, test , most_frequent_day_of_week )


# In[60]:


#most_frequent_day_of_month=history.groupby('card_id')['day_of_month'].agg(mode).reset_index(name='most_frequent_day_of_month')


# In[61]:


#train , test = merge_train_test(train, test , most_frequent_day_of_month )


# In[62]:


#most_frequent_month=history.groupby('card_id')['month_of_year'].agg(mode).reset_index(name='most_frequent_month')


# In[63]:


#train , test = merge_train_test(train, test , most_frequent_month )


# In[64]:


last_buy = history.groupby('card_id')['purchase_date'].max()
last_buy = last_buy.reset_index(name='last_one')
last_buy['last_one']=(datetime.today() - last_buy['last_one']).dt.days.reset_index()


# In[65]:


train , test = merge_train_test(train, test , last_buy )


# In[66]:


First_buy = new_transactions.groupby('card_id')['purchase_date'].min().reset_index(name='first_one')
First_buy['first_one']=(datetime.today() - First_buy['first_one']).dt.days.reset_index()


# In[67]:


train , test = merge_train_test(train, test , First_buy )


# In[68]:


train['between']=train['first_one'] - train['last_one']
test['between']=test['first_one'] - test['last_one']


# In[69]:


train['from']=(datetime.today() - train['first_active_month']).dt.days
test['from']=(datetime.today() - test['first_active_month']).dt.days


# In[70]:


train['activation_month'] = train["first_active_month"].dt.month
test['activation_month'] = test ["first_active_month"].dt.month


# In[71]:


train['activation_year'] = train["first_active_month"].dt.year
test['activation_year'] = test ["first_active_month"].dt.year


# In[72]:


def aggregate_per_month(df,history):
    

    agg_func = {
            'purchase_amount': ['count', 'mean', 'min', 'max', 'std'],
            'installments': ['count', 'sum', 'mean', 'min', 'max', 'std'],
            }

    intermediate_group = history[history['authorized_flag']==1].groupby(['card_id', 'month_lag']).agg(agg_func)
    intermediate_group.columns = ['_'.join(col).strip() for col in intermediate_group.columns.values]
    intermediate_group.reset_index(inplace=True)

    final_group = intermediate_group.groupby('card_id').agg(['mean', 'std'])
    final_group.columns = ['_'.join(col).strip() for col in final_group.columns.values]
    final_group.reset_index(inplace=True)
    final_group = pd.merge(df, final_group, on='card_id', how='left')
    return final_group


# In[73]:


import time
time.sleep(60)


# In[74]:


train = aggregate_per_month(train,history)


# In[75]:


import time
time.sleep(60)


# In[76]:


test=aggregate_per_month(test,history)


# In[77]:


import time
time.sleep(60)


# In[78]:


def successive_aggregates(tr,df, field1, field2, name ):
    t = df.groupby(['card_id', field1])[field2].mean()
    u = pd.DataFrame(t).reset_index().groupby('card_id')[field2].agg(['mean', 'min', 'max', 'std'])
    u.columns = [str(name)+field1 + '_' + field2 + '_' + col for col in u.columns.values]
    u.reset_index(inplace=True)
    tr=pd.merge(tr, u, on='card_id', how='left')
    return tr


# In[79]:





# In[79]:


train = successive_aggregates(train,new_transactions, 'category_1', 'purchase_amount','new')
test  = successive_aggregates(test,new_transactions, 'category_1', 'purchase_amount','new')

train = successive_aggregates(train,new_transactions, 'category_2', 'purchase_amount','new')
test  = successive_aggregates(test,new_transactions, 'category_2', 'purchase_amount','new')

train = successive_aggregates(train,new_transactions, 'category_3', 'purchase_amount','new')
test  = successive_aggregates(test,new_transactions, 'category_3', 'purchase_amount','new')


train = successive_aggregates(train,new_transactions,  'installments', 'purchase_amount','new')
test  = successive_aggregates(test,new_transactions,  'installments', 'purchase_amount','new')

train = successive_aggregates(train,new_transactions, 'city_id', 'purchase_amount','new')
test  = successive_aggregates(test,new_transactions, 'city_id', 'purchase_amount','new')

train = successive_aggregates(train,new_transactions, 'category_1', 'installments','new')
test  = successive_aggregates(test,new_transactions, 'category_1', 'installments','new')

train = successive_aggregates(train,new_transactions, 'category_2', 'installments','new')
test  = successive_aggregates(test,new_transactions, 'category_2', 'installments','new')

train = successive_aggregates(train,new_transactions, 'category_3', 'installments','new')
test  = successive_aggregates(test,new_transactions, 'category_3', 'installments','new')


# In[80]:


time.sleep(15)


# In[81]:


train = successive_aggregates(train,history, 'category_1', 'purchase_amount','history')
test  = successive_aggregates(test,history, 'category_1', 'purchase_amount','history')

train = successive_aggregates(train,history, 'category_2', 'purchase_amount','history')
test  = successive_aggregates(test,history, 'category_2', 'purchase_amount','history')

train = successive_aggregates(train,history, 'category_3', 'purchase_amount','history')
test  = successive_aggregates(test,history, 'category_3', 'purchase_amount','history')


# In[82]:


import time
time.sleep(15)


# In[83]:


train = successive_aggregates(train,history, 'category_1', 'installments','history')
test  = successive_aggregates(test,history, 'category_1', 'installments','history')

train = successive_aggregates(train,history, 'category_2', 'installments','history')
test  = successive_aggregates(test,history, 'category_2', 'installments','history')

train = successive_aggregates(train,history, 'category_3', 'installments','history')
test  = successive_aggregates(test,history, 'category_3', 'installments','history')


# In[84]:


time.sleep(15)


# In[85]:


train = successive_aggregates(train,history,  'installments', 'purchase_amount','history')
test  = successive_aggregates(test,history,  'installments', 'purchase_amount','history')

train = successive_aggregates(train,history, 'city_id', 'purchase_amount','history')
test  = successive_aggregates(test,history, 'city_id', 'purchase_amount','history')


# In[86]:


train['outliers'] = 0
train.loc[train['target'] < -30, 'outliers'] = 1


# In[87]:





# In[87]:


#for f in ['feature_1','feature_2','feature_3']:
#    order_label = train.groupby([f])['outliers'].mean()
#    train[f+'map'] = train[f].map(order_label)
#    test[f+'map'] =  test[f].map(order_label)


# In[88]:


train.head(n=15)


# In[89]:


import time
time.sleep(60)


# In[90]:


def aggregate_new_transactions(new_trans): 
    new_trans['authorized_flag'] =     new_trans['authorized_flag'].map({'Y':1, 'N':0})
    agg_func = {
        'authorized_flag': ['sum', 'mean'],
        'merchant_id': ['nunique'],
        'city_id': ['nunique'],
        'purchase_amount': ['sum', 'median', 'max', 'min', 'std'],
        'installments': ['sum', 'median', 'max', 'min', 'std'],
        'month_lag': ['min', 'max','mean','std']
        }
    agg_new_trans = new_trans.groupby(['card_id']).agg(agg_func)
    agg_new_trans.columns = ['new_' + '_'.join(col).strip() 
                           for col in agg_new_trans.columns.values]
    agg_new_trans.reset_index(inplace=True)
    
    df = (new_trans.groupby('card_id')
          .size()
          .reset_index(name='new_transactions_count'))
    
    #agg_new_trans = pd.merge(df, agg_new_trans, on='card_id', how='left')
    
    return agg_new_trans


# In[91]:


#df = aggregate_new_transactions(new_transactions)


# In[92]:


import time
time.sleep(20)


# In[93]:


#train , test = merge_train_test(train, test , df )


# In[94]:


import time
time.sleep(10)


# In[95]:


#train = other_features (train )
#test= other_features (test )


# In[96]:


import time
time.sleep(120)


# In[97]:


#history = pd.get_dummies(history, columns=['category_2', 'category_3'])


# In[98]:


import time
time.sleep(60)


# In[99]:


def convert(df):
    for i in df.columns:
        if df[i].dtype =='uint8' : df[i]=df[i].astype(float)
    return df 


# In[100]:





# In[100]:


import time
time.sleep(120)


# In[101]:


new_transactions.head(n=5)


# In[102]:


def aggregate_transactions(df,frame,name):
    
    #frame.loc[:, 'purchase_date'] = pd.DatetimeIndex(frame['purchase_date']).\
                                     # astype(np.int64) * 1e-9
    
    agg_func = {
    #'category_1_N': [ 'mean'],
    #'category_1_Y': ['mean'],   
    'category_2_1.0': ['mean'],
    'category_2_2.0': ['mean'],
    'category_2_3.0': ['mean'],
    'category_2_4.0': ['mean'],
    'category_2_5.0': ['mean'],
    'category_3_A': ['mean'],
    'category_3_B': ['mean'],
    'category_3_C': ['mean'],
    
    }
    
    agg_history = frame.groupby(['card_id']).agg(agg_func)
    agg_history.columns = [str(name)+'_'.join(col).strip() for col in agg_history.columns.values]
    agg_history.reset_index(inplace=True)
    agg_new_trans = pd.merge(df, agg_history, on='card_id', how='left')
    
    return agg_new_trans


# In[103]:


history.keys()


# In[104]:


#train.head(n=5)


# In[105]:


print(test.shape)
print(train.shape)


# In[106]:





# In[106]:





# In[106]:


#train = aggregate_per_month(train,history)


# In[107]:


#test=aggregate_per_month(test,history)


# In[108]:





# In[108]:


history.keys()


# In[109]:


history.head(n=10)


# In[110]:


for f in ['feature_1','feature_2','feature_3']:
    order_label = train.groupby([f])['outliers'].mean()
    train[f] = train[f].map(order_label)
    test[f] = test[f].map(order_label)


# In[111]:


X = train.drop(columns=['first_active_month','target','card_id'])
test_X = test.drop(columns=['first_active_month','card_id'])

Y=train['target']


# In[112]:


#X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.33, random_state=7)
X.shape


# In[113]:


corr_matrix = X.corr().abs()


# In[114]:


plt.matshow(corr_matrix)


# In[115]:


upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(np.bool))


# In[116]:


to_drop = [column for column in upper.columns if any(upper[column] > 0.95)]


# In[117]:


to_drop


# In[118]:


#X = X.drop(columns=to_drop)
#test_X = test_X.drop(columns=to_drop)


# In[119]:


X.shape


# In[120]:


test_X.shape


# In[121]:





# In[121]:





# In[121]:


from sklearn.model_selection import StratifiedKFold


# In[122]:


folds = StratifiedKFold(n_splits=5, shuffle=True, random_state=15)


# In[123]:


for fold_, (trn_idx, val_idx) in enumerate(folds.split(train,train['outliers'].values)):
    print(trn_idx, val_idx)


# In[124]:


xgb_params = {'eta': 0.01, 'max_depth': 5, 'subsample': 0.8, 'colsample_bytree': 0.8, 'lambda' : 0.1 , 'alpha' : 0.4,'min_child_weight':1,
          'objective': 'reg:linear', 'eval_metric': 'rmse', 'silent': True, 'nthread': 4}


# In[125]:


#xg_reg = xgb.XGBRegressor(objective ='reg:linear', colsample_bytree = 0.3, learning_rate = 0.1,
               # max_depth = 5, alpha = 10, lamda=2 , n_estimators = 10, eval_metric='rmse')


# In[126]:


import lightgbm as lgb


# In[127]:


param = {'num_leaves': 32,
         'min_data_in_leaf': 149, 
         'objective':'regression',
         'max_depth': 4,
         'learning_rate': 0.005,
         "boosting": "gbdt",
         "feature_fraction": 0.7522,
         "bagging_freq": 1,
         "bagging_fraction": 0.7083 ,
         "bagging_seed": 11,
         "metric": 'rmse',
         "lambda_l1": 0.45,
         "lambda_l2":0.15,
         "random_state": 133,
         "verbosity": -1}


# In[128]:


features = [c for c in X.columns if c not in ['card_id', 'first_active_month', 'target','month_lag_mean_y','month_lag_std_y','outliers',
                                             'history_purchase_date_max', 'history_purchase_date_min', 'new_purchase_date_max', 'new_purchase_date_min']]
categorical_feats = ['feature_1','feature_2','feature_3']


# In[129]:


len(list(X.keys()))


# In[130]:


Y.head()


# In[131]:


#missing_data_function(X)


# In[132]:


#X=X.fillna(-1)


# In[133]:


#missing_data_function(test_X)


# In[134]:


#test_X=test_X.fillna(-1)


# In[135]:





# In[135]:


X.shape


# In[136]:


X=X.drop(columns=['history_purchase_date_max', 'history_purchase_date_min', 'new_purchase_date_max', 'new_purchase_date_min'])


# In[137]:


rest=features


# In[138]:


folds = StratifiedKFold(n_splits=5, shuffle=True, random_state=15)
oof_lgbm = np.zeros(len(train))
predictions_lgbd = np.zeros(len(test))
start = time.time()
feature_importance_df = pd.DataFrame()

for fold_, (trn_idx, val_idx) in enumerate(folds.split(train,train['outliers'].values)):
    print("fold n°{}".format(fold_))
    trn_data = lgb.Dataset(X.iloc[trn_idx][features],
                           label=Y.iloc[trn_idx],
                           categorical_feature=categorical_feats
                          )
    val_data = lgb.Dataset(X.iloc[val_idx][features],
                           label=Y.iloc[val_idx],
                           categorical_feature=categorical_feats
                          )

    num_round = 10000
    clf = lgb.train(param,
                    trn_data,
                    num_round,
                    valid_sets = [trn_data, val_data],
                    verbose_eval=100,
                    early_stopping_rounds = 100)
    
    oof_lgbm[val_idx] = clf.predict(X.iloc[val_idx][features], num_iteration=clf.best_iteration)
    
    
    
    predictions_lgbd += clf.predict(test_X[features], num_iteration=clf.best_iteration) / folds.n_splits


# In[139]:


test_X.keys()


# In[140]:


folds = KFold(n_splits=5, shuffle=True, random_state=15)
oof = np.zeros(len(train))
oof_xgb_3 = np.zeros(len(train))
predictions_xg = np.zeros(len(test))

start = time.time()


for fold_, (trn_idx, val_idx) in enumerate(folds.split(train,train['outliers'].values)):
        
        X_train, X_valid = X.iloc[trn_idx][features], X.iloc[val_idx][features]
        y_train, y_valid = Y.iloc[trn_idx], Y.iloc[val_idx]
        
        
        
        train_data = xgb.DMatrix(data=X_train, label=y_train)
        valid_data = xgb.DMatrix(data=X_valid, label=y_valid)
        watchlist = [(train_data, 'train'), (valid_data, 'valid_data')]
        clf_xg = xgb.train(dtrain=train_data, num_boost_round=20000, evals=watchlist, early_stopping_rounds=100, verbose_eval=500, params=xgb_params)
        y_pred_valid = clf_xg.predict(xgb.DMatrix(X_valid), ntree_limit=clf_xg.best_ntree_limit)
        oof_xgb_3[val_idx] = clf_xg.predict(xgb.DMatrix(X_valid), ntree_limit=clf_xg.best_ntree_limit)
        y_pred = clf_xg.predict(xgb.DMatrix(test_X[features]), ntree_limit=clf_xg.best_ntree_limit)
        predictions_xg += y_pred / folds.n_splits
        
        
       


# In[141]:


#predictions_xg


# In[142]:


param_outliers = {'num_leaves': 8,
         'min_data_in_leaf': 30, 
         'objective':'binary',
         'max_depth': 3,
         'learning_rate': 0.01,
         "boosting": "rf",
         "feature_fraction": 0.9,
         "bagging_freq": 1,
         "bagging_fraction": 0.9 ,
         "bagging_seed": 11,
         "metric": 'binary_logloss',
         "lambda_l1": 0.15,
         "lambda_l2": 0.0,        
         "verbosity": -1,
         "random_state": 2333}


# In[143]:


#predictions_xg


# In[144]:


folds = KFold(n_splits=5, shuffle=True, random_state=15)
oof_outliers = np.zeros(len(train))
predictions_outliers = np.zeros(len(test))
feature_importance_df = pd.DataFrame()
target=train['outliers']
start = time.time()


for fold_, (trn_idx, val_idx) in enumerate(folds.split(train.values, target.values)):
    print("fold n°{}".format(fold_))
    trn_data = lgb.Dataset(train.iloc[trn_idx][features], label=target.iloc[trn_idx], categorical_feature=categorical_feats)
    val_data = lgb.Dataset(train.iloc[val_idx][features], label=target.iloc[val_idx], categorical_feature=categorical_feats)

    num_round = 10000
    clf = lgb.train(param_outliers, trn_data, num_round, valid_sets = [trn_data, val_data], verbose_eval=100, early_stopping_rounds = 200)
    oof_outliers[val_idx] = clf.predict(train.iloc[val_idx][features], num_iteration=clf.best_iteration)
    
    fold_importance_df = pd.DataFrame()
    fold_importance_df["feature"] = features
    fold_importance_df["importance"] = clf.feature_importance()
    fold_importance_df["fold"] = fold_ + 1
    feature_importance_df = pd.concat([feature_importance_df, fold_importance_df], axis=0)
    
    predictions_outliers += clf.predict(test[features], num_iteration=clf.best_iteration) / folds.n_splits

#print("CV score: {:<8.5f}".format(log_loss(target, oof))


# In[145]:


from sklearn.linear_model import Ridge


# In[146]:


train_stack = np.vstack([oof_lgbm, oof_xgb_3,oof_outliers]).transpose()
test_stack = np.vstack([predictions_lgbd, predictions_xg,predictions_outliers]).transpose()

folds = KFold(n_splits=5, shuffle=True, random_state=15)
oof = np.zeros(train_stack.shape[0])
predictions = np.zeros(test_stack.shape[0])

for fold_, (trn_idx, val_idx) in enumerate(folds.split(train,train['outliers'].values)):
    print("fold n°{}".format(fold_))
    X_train, X_valid = train_stack[trn_idx], train_stack[val_idx]
    y_train, y_valid = Y.iloc[trn_idx], Y.iloc[val_idx]

    train_data = xgb.DMatrix(data=X_train, label=y_train)
    valid_data = xgb.DMatrix(data=X_valid, label=y_valid)
    watchlist = [(train_data, 'train'), (valid_data, 'valid_data')]
    clf_xg = xgb.train(dtrain=train_data, num_boost_round=20000, evals=watchlist, early_stopping_rounds=200, verbose_eval=500, params=xgb_params)
    y_pred_valid = clf_xg.predict(xgb.DMatrix(X_valid), ntree_limit=clf_xg.best_ntree_limit)
    #oof_xgb_3[val_idx] = clf_xg.predict(xgb.DMatrix(train.iloc[val_idx][rest]), ntree_limit=clf_xg.best_ntree_limit)
    y_pred = clf_xg.predict(xgb.DMatrix(test_stack), ntree_limit=clf_xg.best_ntree_limit)
    predictions += y_pred / folds.n_splits
    
    #oof[val_idx] = clf.predict(X_valid)
    #predictions += clf.predict(test_stack) / folds.n_splits"""


# In[147]:


predictions_xg


# In[148]:


sub_df = pd.DataFrame({"card_id":test["card_id"].values})
sub_df["target"] =predictions_xg
sub_df.to_csv("submit.csv", index=False)


# In[149]:


sub_df.head(n=100)

