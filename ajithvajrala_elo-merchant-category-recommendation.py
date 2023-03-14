#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import datetime
import gc
import matplotlib.pyplot as plt
import seaborn as sns
import lightgbm as lgb
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import mean_squared_error
import warnings
warnings.filterwarnings('ignore')
np.random.seed(49)
import os
print(os.listdir("../input"))


# In[2]:


train_df = pd.read_csv("../input/train.csv")
test_df = pd.read_csv("../input/test.csv")
merchants_df = pd.read_csv("../input/merchants.csv")


# In[3]:


new_merchant_transactions_df = pd.read_csv("../input/new_merchant_transactions.csv", )
historical_transactions_df = pd.read_csv("../input/historical_transactions.csv")


# In[4]:


#ref https://www.kaggle.com/chauhuynh/my-first-kernel-3-699/ 
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


# In[5]:


historical_transactions_df = reduce_mem_usage(historical_transactions_df)
new_merchant_transactions_df = reduce_mem_usage(new_merchant_transactions_df)


# In[6]:


gc.collect()


# In[7]:


merchants_df = reduce_mem_usage(merchants_df)


# In[8]:


train_df.head()


# In[9]:


test_df.head()


# In[10]:


historical_transactions_df.head()


# In[11]:


merchants_df.head()


# In[12]:


gc.collect()


# In[13]:


#check for null values
for df in[historical_transactions_df, new_merchant_transactions_df]:
    print(df.isna().sum())


# In[14]:


for df in [historical_transactions_df,new_merchant_transactions_df]:
    df['category_2'].fillna(1.0,inplace=True)
    df['category_3'].fillna('A',inplace=True)
    df['merchant_id'].fillna('M_ID_00a6ca8a8a',inplace=True)


# In[15]:


merchants_df.isna().sum()


# In[16]:


merchants_df.apply(lambda x : len(x.unique()))


# In[17]:


merchants_df['avg_purchases_lag3'].value_counts().head()


# In[18]:


merchants_df['avg_purchases_lag6'].value_counts().head()


# In[19]:


merchants_df['avg_purchases_lag12'].value_counts().head()


# In[20]:


merchants_df['category_2'].value_counts().head()


# In[21]:


merchants_df['avg_sales_lag3'].fillna(merchants_df['avg_sales_lag3'].mean(),inplace=True)
merchants_df['avg_sales_lag6'].fillna(merchants_df['avg_sales_lag6'].mean(),inplace=True)
merchants_df['avg_sales_lag12'].fillna(merchants_df['avg_sales_lag12'].mean(),inplace=True)
merchants_df['category_2'].fillna(1.0,inplace=True)


# In[22]:


merchants_df.isnull().sum()


# In[23]:


merchants_df.head()


# In[24]:


merchants_df['avg_sales_lag6'].describe()


# In[25]:


#del merchants_df['avg_sales_lag3']
#del merchants_df['avg_sales_lag6']
#del merchants_df['avg_sales_lag12']


# In[26]:


merchants_df.columns = ['merch_' + str(col) for col in merchants_df.columns]


# In[27]:


merchants_df.columns


# In[28]:


merchants_df.columns = ['merchant_id', 'merch_merchant_group_id',
       'merch_merchant_category_id', 'merch_subsector_id', 'merch_numerical_1',
       'merch_numerical_2', 'merch_category_1',
       'merch_most_recent_sales_range', 'merch_most_recent_purchases_range',
       'merch_avg_sales_lag3', 'merch_avg_purchases_lag3',
       'merch_active_months_lag3', 'merch_avg_sales_lag6',
       'merch_avg_purchases_lag6', 'merch_active_months_lag6',
       'merch_avg_sales_lag12', 'merch_avg_purchases_lag12',
       'merch_active_months_lag12', 'merch_category_4', 'merch_city_id',
       'merch_state_id', 'merch_category_2']


# In[29]:


merchants_df['merch_most_recent_sales_range'].value_counts()


# In[30]:


merchants_df.head()


# In[31]:


merchants_df['merch_category_1'] = merchants_df['merch_category_1'].map({'Y':1, 'N':0})
merchants_df['merch_most_recent_sales_range'] = merchants_df['merch_most_recent_sales_range'].astype('category').cat.codes
merchants_df['merch_most_recent_purchases_range'] = merchants_df['merch_most_recent_purchases_range'].astype('category').cat.codes


# In[32]:


merchants_df = merchants_df.drop_duplicates(subset='merchant_id', keep="last")


# In[33]:


merchants_df.shape


# In[34]:


merchants_df.head()


# In[35]:


merchants_df['merch_category_4'].value_counts()


# In[36]:


merchants_df['merch_category_4'] = merchants_df['merch_category_4'].map({'Y':1, 'N':0})


# In[37]:


merchants_df.dtypes


# In[38]:


merchants_df['merch_avg_sales_lag3'] = merchants_df['merch_avg_sales_lag3'].astype('float16')
merchants_df['merch_avg_purchases_lag3'] = merchants_df['merch_avg_purchases_lag3'].astype('float16')
merchants_df['merch_avg_sales_lag6'] = merchants_df['merch_avg_sales_lag6'].astype('float16')
merchants_df['merch_avg_purchases_lag6'] = merchants_df['merch_avg_purchases_lag6'].astype('float16')
merchants_df['merch_avg_sales_lag12'] = merchants_df['merch_avg_sales_lag12'].astype('float16')
merchants_df['merch_avg_purchases_lag12'] = merchants_df['merch_avg_purchases_lag12'].astype('float16')


# In[39]:


gc.collect()


# In[40]:


historical_transactions_df = historical_transactions_df.merge(merchants_df,on='merchant_id', how='left')


# In[41]:


gc.collect()


# In[42]:


historical_transactions_df.shape


# In[43]:


historical_transactions_df.head()


# In[44]:


gc.collect()


# In[45]:


new_merchant_transactions_df = new_merchant_transactions_df.merge(merchants_df,on='merchant_id', how='left')


# In[46]:


gc.collect()


# In[47]:


for df in[historical_transactions_df, new_merchant_transactions_df]:
    del df['merch_city_id']
    del df['merch_category_2']
    del df['merch_state_id']
    del df['merch_merchant_category_id']
    del df['merch_subsector_id']
    del df['merch_category_1']


# In[48]:


gc.collect()


# In[49]:


del merchants_df
gc.collect()


# In[50]:


new_merchant_transactions_df.head()


# In[51]:


gc.collect()


# In[52]:


def get_new_columns(name,aggs):
    return [name + '_' + k + '_' + agg for k in aggs.keys() for agg in aggs[k]]


# In[53]:


for df in [historical_transactions_df,new_merchant_transactions_df]:
    df['purchase_date'] = pd.to_datetime(df['purchase_date'])
    df['year'] = df['purchase_date'].dt.year
    df['weekofyear'] = df['purchase_date'].dt.weekofyear
    df['month'] = df['purchase_date'].dt.month
    df['dayofweek'] = df['purchase_date'].dt.dayofweek
    df['weekend'] = (df.purchase_date.dt.weekday >=5).astype(int)
    df['hour'] = df['purchase_date'].dt.hour
    df['authorized_flag'] = df['authorized_flag'].map({'Y':1, 'N':0})
    df['category_1'] = df['category_1'].map({'Y':1, 'N':0}) 
    df['month_diff'] = ((datetime.datetime.today() - df['purchase_date']).dt.days)//30
    df['month_diff'] += df['month_lag']


# In[54]:


historical_transactions_df.dtypes


# In[55]:


gc.collect()


# In[56]:


for df in [historical_transactions_df,new_merchant_transactions_df]:
    df['merch_category_4'] = df['merch_category_4'].astype('int8')
    df['year'] = df['year'].astype('int8')
    df['weekofyear'] = df['weekofyear'].astype('int8')
    df['month'] = df['month'].astype('int8')
    df['dayofweek'] = df['dayofweek'].astype('int8')
    df['weekend'] = df['weekend'].astype('int8')
    df['hour'] = df['hour'].astype('int8')
    df['month_diff'] = df['month_diff'].astype('int8')


# In[57]:


gc.collect()


# In[58]:


historical_transactions_df.head()


# In[59]:


for df in[historical_transactions_df, new_merchant_transactions_df]:
    df['category_3'] = df['category_3'].astype('category').cat.codes


# In[60]:


gc.collect()


# In[61]:


historical_transactions_df.dtypes


# In[62]:


gc.collect()


# In[63]:


historical_transactions_df.head()


# In[64]:


aggs = {}
for col in ['month','hour','weekofyear','dayofweek','year','subsector_id','merchant_id','merchant_category_id']:
    aggs[col] = ['nunique']
    
    
    
aggs['purchase_amount'] = ['sum','max','min','mean','var']
aggs['installments'] = ['sum','max','min','mean','var']
aggs['purchase_date'] = ['max','min']
aggs['month_lag'] = ['max','min','mean','var']
aggs['month_diff'] = ['mean']
aggs['authorized_flag'] = ['sum', 'mean']
aggs['weekend'] = ['sum', 'mean']
aggs['category_1'] = ['sum', 'mean']
aggs['card_id'] = ['size']

aggs['merch_numerical_1'] = ['sum', 'mean']
aggs['merch_numerical_2'] = ['sum', 'mean']
aggs['merch_most_recent_sales_range'] = ['sum', 'mean']
aggs['merch_most_recent_purchases_range'] = ['sum', 'mean']
aggs['merch_avg_purchases_lag3'] = ['sum', 'mean']
aggs['merch_active_months_lag3'] = ['sum', 'mean']
aggs['merch_avg_purchases_lag6'] = ['sum', 'mean']
aggs['merch_active_months_lag6'] = ['sum', 'mean']
aggs['merch_avg_purchases_lag12'] = ['sum', 'mean']
aggs['merch_active_months_lag12'] = ['sum', 'mean']
aggs['merch_category_4'] = ['sum', 'mean']


# In[65]:


for col in ['category_1', 'category_2','category_3','installments', 'state_id', 'month', 'dayofweek', 'hour']:
    historical_transactions_df[col+'_mean'] = historical_transactions_df.groupby([col])['purchase_amount'].transform('mean')
    aggs[col+'_mean'] = ['mean']   
    
historical_transactions_df.head()


# In[66]:


gc.collect()


# In[67]:


new_columns = get_new_columns('hist',aggs)


# In[68]:


df_hist_trans_group = historical_transactions_df.groupby('card_id').agg(aggs)
df_hist_trans_group.columns = new_columns
df_hist_trans_group.reset_index(drop=False,inplace=True)
df_hist_trans_group['hist_purchase_date_diff'] = (df_hist_trans_group['hist_purchase_date_max'] - df_hist_trans_group['hist_purchase_date_min']).dt.days
df_hist_trans_group['hist_purchase_date_average'] = df_hist_trans_group['hist_purchase_date_diff']/df_hist_trans_group['hist_card_id_size']
df_hist_trans_group['hist_purchase_date_uptonow'] = (datetime.datetime.today() - df_hist_trans_group['hist_purchase_date_max']).dt.days


# In[69]:


del historical_transactions_df
gc.collect()


# In[70]:


gc.collect()


# In[71]:


train_df = train_df.merge(df_hist_trans_group,on='card_id',how='left')
test_df = test_df.merge(df_hist_trans_group,on='card_id',how='left')
del df_hist_trans_group
gc.collect()


# In[72]:


gc.collect()


# In[73]:


train_df.head()


# In[74]:


test_df.head()


# In[75]:


aggs = {}
for col in ['month','hour','weekofyear','dayofweek','year','subsector_id','merchant_id','merchant_category_id']:
    aggs[col] = ['nunique']
aggs['purchase_amount'] = ['sum','max','min','mean','var']
aggs['installments'] = ['sum','max','min','mean','var']
aggs['purchase_date'] = ['max','min']
aggs['month_lag'] = ['max','min','mean','var']
aggs['month_diff'] = ['mean']
aggs['weekend'] = ['sum', 'mean']
aggs['category_1'] = ['sum', 'mean']
aggs['card_id'] = ['size']

for col in ['category_1', 'category_2','category_3', 'installments', 'state_id', 'month', 'dayofweek', 'hour']:
    new_merchant_transactions_df[col+'_mean'] = new_merchant_transactions_df.groupby([col])['purchase_amount'].transform('mean')
    aggs[col+'_mean'] = ['mean']
    
new_columns = get_new_columns('new_hist',aggs)
df_hist_trans_group = new_merchant_transactions_df.groupby('card_id').agg(aggs)
df_hist_trans_group.columns = new_columns
df_hist_trans_group.reset_index(drop=False,inplace=True)
df_hist_trans_group['new_hist_purchase_date_diff'] = (df_hist_trans_group['new_hist_purchase_date_max'] - df_hist_trans_group['new_hist_purchase_date_min']).dt.days
df_hist_trans_group['new_hist_purchase_date_average'] = df_hist_trans_group['new_hist_purchase_date_diff']/df_hist_trans_group['new_hist_card_id_size']
df_hist_trans_group['new_hist_purchase_date_uptonow'] = (datetime.datetime.today() - df_hist_trans_group['new_hist_purchase_date_max']).dt.days
train_df = train_df.merge(df_hist_trans_group,on='card_id',how='left')
test_df = test_df.merge(df_hist_trans_group,on='card_id',how='left')
del df_hist_trans_group
gc.collect()


# In[76]:


train_df.head(5)


# In[77]:


del new_merchant_transactions_df
gc.collect()


# In[78]:


gc.collect()


# In[79]:


train_df['outliers'] = 0
train_df.loc[train_df['target'] < -30, 'outliers'] = 1
train_df['outliers'].value_counts()


# In[80]:


for df in [train_df,test_df]:
    df['first_active_month'] = pd.to_datetime(df['first_active_month'])
    df['dayofweek'] = df['first_active_month'].dt.dayofweek
    df['weekofyear'] = df['first_active_month'].dt.weekofyear
    df['month'] = df['first_active_month'].dt.month
    df['elapsed_time'] = (datetime.datetime.today() - df['first_active_month']).dt.days
    df['hist_first_buy'] = (df['hist_purchase_date_min'] - df['first_active_month']).dt.days
    df['new_hist_first_buy'] = (df['new_hist_purchase_date_min'] - df['first_active_month']).dt.days
    
    for f in ['hist_purchase_date_max','hist_purchase_date_min','new_hist_purchase_date_max','new_hist_purchase_date_min']:
        df[f] = df[f].astype(np.int64) * 1e-9
    
    df['card_id_total'] = df['new_hist_card_id_size']+df['hist_card_id_size']
    df['purchase_amount_total'] = df['new_hist_purchase_amount_sum']+df['hist_purchase_amount_sum']

for f in ['feature_1','feature_2','feature_3']:
    order_label = train_df.groupby([f])['outliers'].mean()
    train_df[f] = train_df[f].map(order_label)
    test_df[f] = test_df[f].map(order_label)


# In[81]:


train_df.head()


# In[82]:


test_df.head()


# In[83]:


gc.collect()


# In[84]:


df_train_columns = [c for c in train_df.columns if c not in ['card_id', 'first_active_month','target','outliers']]
target = train_df['target']
del train_df['target']


# In[85]:


param = {'num_leaves': 31,
        'min_data_in_leaf': 30, 
        'objective':'regression',
        'max_depth': -1,
        'learning_rate': 0.01,
        "min_child_samples": 20,
        "boosting": "gbdt",
        "feature_fraction": 0.9,
        "bagging_freq": 1,
        "bagging_fraction": 0.9 ,
        "bagging_seed": 11,
        "metric": 'rmse',
        "lambda_l1": 0.1,
        "verbosity": -1,
        "nthread": 4,
        "random_state": 49}
   
folds = StratifiedKFold(n_splits=5, shuffle=True, random_state=49)
oof = np.zeros(len(train_df))
predictions = np.zeros(len(test_df))
feature_importance_df = pd.DataFrame()

for fold_, (trn_idx, val_idx) in enumerate(folds.split(train_df,train_df['outliers'].values)):
   print("fold {}".format(fold_))
   trn_data = lgb.Dataset(train_df.iloc[trn_idx][df_train_columns], label=target.iloc[trn_idx])#, categorical_feature=categorical_feats)
   val_data = lgb.Dataset(train_df.iloc[val_idx][df_train_columns], label=target.iloc[val_idx])#, categorical_feature=categorical_feats)

   num_round = 10000
   clf = lgb.train(param, trn_data, num_round, valid_sets = [trn_data, val_data], verbose_eval=100, early_stopping_rounds = 100)
   oof[val_idx] = clf.predict(train_df.iloc[val_idx][df_train_columns], num_iteration=clf.best_iteration)
   
   fold_importance_df = pd.DataFrame()
   fold_importance_df["Feature"] = df_train_columns
   fold_importance_df["importance"] = clf.feature_importance()
   fold_importance_df["fold"] = fold_ + 1
   feature_importance_df = pd.concat([feature_importance_df, fold_importance_df], axis=0)
   
   predictions += clf.predict(test_df[df_train_columns], num_iteration=clf.best_iteration) / folds.n_splits

np.sqrt(mean_squared_error(oof, target))


# In[86]:


cols = (feature_importance_df[["Feature", "importance"]]
        .groupby("Feature")
        .mean()
        .sort_values(by="importance", ascending=False)[:1000].index)

best_features = feature_importance_df.loc[feature_importance_df.Feature.isin(cols)]

plt.figure(figsize=(14,25))
sns.barplot(x="importance",
            y="Feature",
            data=best_features.sort_values(by="importance",
                                           ascending=False))
plt.title('LightGBM Features (avg over folds)')
plt.tight_layout()


# In[87]:


sub_df = pd.DataFrame({"card_id":test_df["card_id"].values})
sub_df["target"] = predictions
sub_df.to_csv("submission.csv", index=False)


# In[88]:


sub_df.head()


# In[89]:




