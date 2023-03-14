#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns 
import numpy as np
from scipy import stats 
get_ipython().run_line_magic('matplotlib', 'inline')
import plotly.offline as py #可视化
py.init_notebook_mode(connected = True)
import plotly.graph_objs as go
import plotly.tools as tls
import plotly.figure_factory as ff
from sklearn.model_selection import KFold
from sklearn.metrics import mean_absolute_error
import warnings
warnings.filterwarnings("ignore")
import gc #内存管理
sns.set_style('white')


# In[2]:


debug = False
if debug == True:
    train_df = pd.read_csv('../input/train_V2.csv',nrows = 1000000)
    test_df = pd.read_csv('../input/test.csv')
else:
    train_df = pd.read_csv('../input/train_V2.csv')
    test_df = pd.read_csv('../input/test_V2.csv')


# In[3]:


print("train data's shape is:",train_df.shape)
print("test data's shape is:",test_df.shape)


# In[4]:


train_df.head()


# In[5]:


# Memory saving function credit to https://www.kaggle.com/gemartin/load-data-reduce-memory-usage
def reduce_mem_usage(df):
    """ iterate through all the columns of a dataframe and modify the data type
        to reduce memory usage. 
        ,通过各列的取值范围确定对应的int类型
    """
    start_mem = df.memory_usage().sum() / 1024**2
    print('Memory usage of dataframe before optimization is {:.2f} MB'.format(start_mem))

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


# In[6]:


train_df = reduce_mem_usage(train_df)
test_df = reduce_mem_usage(test_df)


# In[7]:


train_df.info()


# In[8]:


train_df.head(5)


# In[9]:


train_df.loc[:,train_df.isnull().any()].columns.tolist()


# In[10]:


train_df[train_df['winPlacePerc'].isnull()]


# In[11]:


train_df.drop(2744604,inplace = True)


# In[12]:


test_df.loc[:,train_df.isnull().any()].columns.tolist()


# In[13]:


qt_df = pd.DataFrame(train_df.quantile(0.99));
qt_df


# In[14]:


train_df['killsWithoutMoving'] = (train_df['kills'] > 0) & (train_df['walkDistance'] == 0)                                                         & (train_df['swimDistance'] == 0)
train_df[train_df['killsWithoutMoving'] == True].head()


# In[15]:


train_df.drop(train_df[train_df['killsWithoutMoving'] == True].index,inplace = True)


# In[16]:


train_df.drop('killsWithoutMoving',axis = 1,inplace = True)


# In[17]:


train_df['roadKills'].value_counts().sort_values(ascending = False)


# In[18]:


train_df[train_df['roadKills'] > 7].head()


# In[19]:


train_df.drop(train_df[train_df['roadKills'] > 7].index,inplace = True)


# In[20]:


sns.set_style('darkgrid')


# In[21]:


train_df['headshotRate'] = train_df['headshotKills'] / train_df['kills']
train_df['headshotRate'] = train_df['headshotRate'].fillna(0)
train_df['headshotRate'].replace(np.inf, 0, inplace=True)


# In[22]:


train_df[(train_df['headshotRate'] == 1) & (train_df['kills'] > 7)].head()


# In[23]:


sns.countplot(train_df['kills'][(train_df['headshotRate'] == 1) & (train_df['kills'] > 7)])


# In[24]:


train_df.drop(train_df[(train_df['headshotRate'] == 1) & (train_df['kills'] > 7)].index,inplace = True)


# In[25]:


plt.figure(figsize=(12,4))
_ = sns.distplot(train_df['longestKill'], bins=10)


# In[26]:


train_df[['Id','longestKill','winPlacePerc']][train_df['longestKill'] > 800].head()


# In[27]:


train_df.drop(train_df[train_df['longestKill'] > 800].index,inplace = True)


# In[28]:


sns.distplot(train_df['rideDistance'],bins = 10)


# In[29]:


train_df[train_df['rideDistance'] > 25000].head()


# In[30]:


train_df.drop(train_df[train_df['rideDistance'] > 25000].index,inplace = True)


# In[31]:


sns.distplot(train_df['swimDistance'],bins = 10)


# In[32]:


train_df.drop(train_df[train_df['swimDistance'] > 2000].index,inplace = True)


# In[33]:


fig = plt.figure(figsize = (8,4))
sns.distplot(train_df['walkDistance'],bins = 10)


# In[34]:


train_df.drop(train_df[train_df['walkDistance'] > 15000].index,inplace = True)


# In[35]:


train_df.shape


# In[36]:


train_df.groupby('matchType')['maxPlace'].agg('mean')


# In[37]:


train_df['playersJoined'] = train_df.groupby('matchId')['matchId'].transform('count')
plt.figure(figsize=(10,4))
sns.countplot(train_df[train_df['playersJoined'] >= 70]['playersJoined'],saturation = 1,palette = 'RdBu')
plt.title('playersJoined')
plt.show()


# In[38]:


train_df['killsNorm'] = train_df['kills']*((100-train_df['playersJoined'])/100 + 1)
train_df['damageDealtNorm'] = train_df['damageDealt']*((100-train_df['playersJoined'])/100 + 1)
train_df['maxPlaceNorm'] = train_df['maxPlace']*((100-train_df['playersJoined'])/100 + 1)
train_df['matchDurationNorm'] = train_df['matchDuration']*((100-train_df['playersJoined'])/100 + 1)


# In[39]:


_ = sns.distplot(train_df['headshotRate'],bins = 10,kde = False)


# In[40]:


data = train_df.copy()
data['headshotRate'] = pd.cut(data['headshotRate'], [0, 0.2, 0.4, 0.6, 0.8,1], labels=['0-.2','.2-.4', '.4-.6', '.6-.8','.8-1.0'])


# In[41]:


plt.figure(figsize=(10,6))
sns.boxplot(x="headshotRate", y="winPlacePerc", data= data)
plt.show()


# In[42]:


del data


# In[43]:


train_df['totalDistance'] = train_df['walkDistance']+ train_df['rideDistance']                                         +train_df['swimDistance']


# In[44]:


sns.set_palette("RdBu")
_ = sns.jointplot(x = 'totalDistance',y = 'winPlacePerc',data = train_df                  ,kind = 'scatter',s = 5)


# In[45]:


train_df['healsAndBoosts'] = train_df['heals'] + train_df['boosts']


# In[46]:


sns.set_palette("tab20c")
sns.pairplot(train_df[['healsAndBoosts','heals','boosts','winPlacePerc']],
            plot_kws =dict(s = 4))


# In[47]:


train_df['killPlaceInternal'] = train_df['killPlace'] / train_df['maxPlace']
train_df['killPlaceInternal'].fillna(0, inplace=True)
train_df['killPlaceInternal'].replace(np.inf, 0, inplace=True)


# In[48]:


data = train_df.copy()
data['killPlaceInternal'] = pd.cut(data['killPlaceInternal'], [0, 0.2, 0.4, 0.6, 0.8,1], labels=['0-.2','.2-.4', '.4-.6', '.6-.8','.8-1.0'])


# In[49]:


plt.figure(figsize=(10,6))
sns.boxplot(x="killPlaceInternal", y="winPlacePerc", data= data)
plt.show()


# In[50]:


train_df['walkPerboost'] = train_df['walkDistance'] / train_df['boosts']
train_df['walkPerboost'].fillna(0,inplace = True)
train_df['walkPerboost'].replace(np.inf, 0, inplace=True)


# In[51]:


sns.jointplot(x = 'walkPerboost',y = 'winPlacePerc',data = train_df                  ,kind = 'scatter',s = 5)


# In[52]:


train_df['walkPerHeal'] = train_df['walkDistance'] / train_df['heals']
train_df['walkPerHeal'].fillna(0,inplace = True)
train_df['walkPerHeal'].replace(np.inf, 0, inplace=True)


# In[53]:


sns.jointplot(x = 'walkPerHeal',y = 'winPlacePerc',data = train_df                  ,kind = 'scatter',s = 5)


# In[54]:


import shap
import random
from sklearn.model_selection import train_test_split
shap.initjs()


# In[55]:


target = 'winPlacePerc'
cols_drop = ['Id', 'groupId', 'matchId', 'matchType', target]
cols_fit = [col for col in train_df.columns if col not in cols_drop]


# In[56]:


from lightgbm import LGBMRegressor
params = {
    'n_estimators': 100,
    'learning_rate': 0.3, 
    'num_leaves': 20,
    'objective': 'regression_l2', 
    'metric': 'mae',
    'verbose': -1,
}
train_X, val_X, train_y,val_y = train_test_split(train_df[cols_fit],train_df[target],test_size = 0.1,random_state = 1)


# In[57]:


model = LGBMRegressor(**params)
model.fit(
    train_X,train_y,
    eval_set=[(val_X,val_y)],
    eval_metric='mae',
    verbose = 20,
)


# In[58]:


explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(val_X)


# In[59]:


shap.summary_plot(shap_values, val_X, plot_type='bar')


# In[60]:


shap.summary_plot(shap_values, val_X, feature_names=cols_fit)


# In[61]:


train_df.drop(['kills','matchDuration','heals','headshotRate','damageDealtNorm',               'swimDistance','walkPerHeal','teamKills','roadKills','vehicleDestroys'],
              axis = 1,inplace = True)


# In[62]:


train_df.shape


# In[63]:


def feature_engineering(data,is_train = True):
    test_idx = None
    if is_train:
        print('processing train.csv')
        df = data
    else:
        print('processing test.csv')
        df = pd.read_csv('../input/test_V2.csv')
        test_idx = df.Id
        df['playersJoined'] = df.groupby('matchId')['matchId'].transform('count')
    
    print('remove some columns')
    target = 'winPlacePerc'
    features = list(df.columns)
    #使用remove函数可以去除list中的第一个指定元素
    features.remove('Id')
    features.remove('matchId')
    features.remove('groupId')
    features.remove('matchType')
    y = None
    
    if is_train:
        print('get target')
        #这样可以获得每场比赛每只队伍的winPlacePerc
        y = np.array(df.groupby(['matchId','groupId'])[target].agg('mean'),dtype = np.float64)
        features.remove(target)
    
    print('get Group mean feature')
    #得到各局比赛中每一组各特征的平均值，注意，接下来都是以组为单位了
    agg = df.groupby(['matchId','groupId'])[features].agg('mean')
    #各局比赛中每一组各特征的局内排名
    agg_rank = agg.groupby('matchId')[features].rank(pct = True).reset_index()
    
    #agg 训练集里组号相同的ID已经合并了，但训练集还没有
    if is_train:
         #经过groupby之后matchId和groupId已经自动按组排序好了
        df_out = agg.reset_index()[['matchId','groupId']]
    else:
        df_out = df[['matchId','groupId']]
    
    #这里merge的结果不是只有matchId和groupId，还有agg中的其他属性
    df_out = df_out.merge(agg.reset_index(),suffixes = ["",""],how = 'left',on = ['matchId','groupId'])
    #每局比赛每组的平均成绩和平均成绩百分比排名
    df_out = df_out.merge(agg_rank,suffixes = ["_mean","_mean_rank"],how = 'left',on = ['matchId','groupId'])
    
     #每局比赛每组的最大值和最大值百分比排名
    print('get group max feature')
    agg = df.groupby(['matchId','groupId'])[features].agg('max')
    agg_rank = agg.groupby('matchId')[features].rank(pct = True).reset_index()
    df_out = df_out.merge(agg.reset_index(),suffixes = ["",""],how = 'left',on = ['matchId','groupId'])
    df_out = df_out.merge(agg_rank, suffixes=["_max", "_max_rank"], how='left', on=['matchId', 'groupId'])
    
    #每局比赛每组的最小值和最小值百分比排名
    print("get group min feature")
    agg = df.groupby(['matchId','groupId'])[features].agg('min')
    agg_rank = agg.groupby('matchId')[features].rank(pct = True).reset_index()
    df_out = df_out.merge(agg.reset_index(),suffixes = ["",""],how = 'left',on = ['matchId','groupId'])
    df_out = df_out.merge(agg_rank,suffixes = ["_min","_min_rank"],how = 'left',on = ['matchId','groupId'])
    
    #每局比赛所有人的平均值
    print('get match mean feature')
    agg =  df.groupby(['matchId'])[features].agg('mean').reset_index()
    df_out = df_out.merge(agg,suffixes = ["","_match_mean"], how = 'left',on = ['matchId'])
    
    #每局比赛的人数
    print('get match size feature')
    agg = df.groupby(['matchId']).size().reset_index(name = 'match_size')
    df_out = df_out.merge(agg,how = 'left', on = ['matchId'])
    
    #每局比赛每组的人数
    agg = df.groupby(['matchId','groupId']).size().reset_index(name = 'group_size')
    df_out = df_out.merge(agg,how = 'left',on = ['matchId','groupId'])
    
    df_out.drop(['matchId','groupId'],axis = 1,inplace = True)
    X = df_out
    
    feature_names = list(df_out.columns)
    
    del df,df_out,agg,agg_rank
    gc.collect()
    
    return X,y,feature_names,test_idx


# In[64]:


#x_train是训练数据，y_train是标签，这里x_train和x_test都是没有matchId和groupId的
x_train,y_train,train_columns,_ = feature_engineering(train_df,True)


# In[65]:


x_test,_,_,test_idx = feature_engineering(train_df,False)


# In[66]:


x_train = reduce_mem_usage(x_train)
x_test = reduce_mem_usage(x_test)


# In[67]:


x_train = reduce_mem_usage(x_train)
x_test = reduce_mem_usage(x_test)


# In[68]:


x_train.info()


# In[69]:


del train_df
gc.collect()


# In[70]:


x_test['killsNorm'] = x_test['kills']*((100- x_test['playersJoined'])/100 + 1)
x_test['maxPlaceNorm'] = x_test['maxPlace']*((100-x_test['playersJoined'])/100 + 1)
x_test['matchDurationNorm'] = x_test['matchDuration']*((100-x_test['playersJoined'])/100 + 1)

x_test['totalDistance'] = x_test['walkDistance']+ x_test['rideDistance']                                         + x_test['swimDistance']

x_test['healsAndBoosts'] = x_test['heals'] + x_test['boosts']

x_test['killPlaceInternal'] = x_test['killPlace'] / x_test['maxPlace']
x_test['killPlaceInternal'].fillna(0, inplace=True)
x_test['killPlaceInternal'].replace(np.inf, 0, inplace=True)

x_test['walkPerboost'] = x_test['walkDistance'] / x_test['boosts']
x_test['walkPerboost'].fillna(0,inplace = True)
x_test['walkPerboost'].replace(np.inf, 0, inplace=True)


# In[71]:


x_test.drop(['kills','matchDuration','heals',               'swimDistance','teamKills','roadKills','vehicleDestroys'],
              axis = 1,inplace = True)


# In[72]:


gc.collect()


# In[73]:


import os
import time
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
import lightgbm as lgb


# In[74]:


folds = KFold(n_splits = 3,random_state = 6)
oof_preds = np.zeros(x_train.shape[0])
sub_preds = np.zeros(x_test.shape[0])

start = time.time()
valid_score = 0

feature_importance_df = pd.DataFrame()

for n_fold,(trn_idx,val_idx) in enumerate(folds.split(x_train,y_train)):
    trn_x,trn_y = x_train.iloc[trn_idx],y_train[trn_idx]
    val_x,val_y = x_train.iloc[val_idx],y_train[val_idx]
    
    #将数据加载到lightGBM的dataset对象中
    train_data = lgb.Dataset(data = trn_x,label = trn_y)
    valid_data = lgb.Dataset(data = val_x, label = val_y)
    
    params = {
        "application" : "regression",
        "metric" : "mae", 
        'n_estimators': 15000,
        "early_stopping_rounds": 100, 
        "num_leaves": 31,
        "learning_rate": 0.05,
        "bagging_fraction" : 0.8, 
        'bagging_seed':0,
        "num_threads":4,
        "feature_fraction":0.7 
    }
    lgb_model = lgb.train(params,train_data,valid_sets = [train_data,valid_data],verbose_eval = 1000)
    #oof_preds是整个训练集的预测结果
    oof_preds[val_idx] = lgb_model.predict(val_x, num_iteration = lgb_model.best_iteration)
    oof_preds[oof_preds > 1] = 1
    oof_preds[oof_preds < 0] = 0
    #x_test是测试集，怎么会在交叉验证中使用
    sub_pred = lgb_model.predict(x_test, num_iteration = lgb_model.best_iteration)
    sub_pred[sub_pred > 1] = 1
    sub_pred[sub_pred < 0] = 0
    sub_preds += sub_pred / folds.n_splits
    
    fold_importance_df = pd.DataFrame()
    fold_importance_df["feature"] = train_columns
    fold_importance_df["importance"] = lgb_model.feature_importance()
    fold_importance_df['fold'] = n_fold + 1
    feature_importance_df = pd.concat([feature_importance_df, fold_importance_df], axis = 0)
    
gc.collect()
end = time.time()
print("Take time:",(end - start))


# In[75]:


cols = feature_importance_df[['feature','importance']].groupby("feature").                            mean().sort_values(by = 'importance',ascending = False)[:50].index
#目的是找出前五十个最好的特征(最大，最小，平均)
best_features = feature_importance_df.loc[feature_importance_df.feature.isin(cols)]


# In[76]:


plt.figure(figsize = (14,10))
sns.barplot(x = "importance",y = 'feature',            data = best_features.sort_values(by = "importance",ascending = False))
plt.title('LightGBM Features (avg over folds)')
plt.tight_layout()
plt.savefig('lgbm_importance.png')


# In[77]:


f, ax = plt.subplots(figsize=(14, 14))
plt.scatter(y_train, oof_preds)
plt.xlabel("y")
plt.ylabel("predict_y")
plt.show()


# In[78]:


df_test = pd.read_csv('../input/test_V2.csv')
pred = sub_preds
print("fix winPlacePerc")
for i in range(len(df_test)):
    winPlacePerc = pred[i]
    maxPlace = int(df_test.iloc[i]['maxPlace'])
    if maxPlace == 0:
        winPlacePerc = 0.0
    elif maxPlace == 1:
        winPlacePerc = 1.0
    else:
        #排名最大距离
        gap = 1.0 / (maxPlace - 1)
        #winPlacePerc = winPlace / (maxPlace - 1),但这样的意义大吗？
        winPlacePerc = round(winPlacePerc / gap) * gap
    
    if winPlacePerc < 0: winPlacePerc = 0.0
    if winPlacePerc > 1: winPlacePerc = 1.0    
    pred[i] = winPlacePerc

    if (i + 1) % 100000 == 0:
        print(i, flush=True, end=" ")

df_test['winPlacePerc'] = pred

submission = df_test[['Id', 'winPlacePerc']]
submission.to_csv('submission.csv', index=False)


# In[79]:




