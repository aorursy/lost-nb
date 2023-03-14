#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import os
import pathlib

from tqdm.notebook import tnrange, tqdm

from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from skmultilearn.problem_transform import BinaryRelevance

from sklearn.multioutput import MultiOutputClassifier
from sklearn.metrics import classification_report, cohen_kappa_score, coverage_error,label_ranking_average_precision_score, make_scorer
from sklearn.metrics import hamming_loss, accuracy_score

import catboost
from catboost import Pool, cv

pd.set_option('display.max_columns', 1000)
pd.set_option('display.max_rows', 500)


# In[2]:


for dirname, _, filenames in os.walk('../input'):
    for filename in filenames:
        # getting POSIX path for Windows machine; filenames with /s rather than \s
        print(pathlib.Path(os.path.join(dirname,filename)).as_posix())


# In[3]:


train = pd.read_csv('../input/data-science-bowl-2019/train.csv')


# In[4]:


test = pd.read_csv('../input/data-science-bowl-2019/test.csv')
train_labels = pd.read_csv('../input/data-science-bowl-2019/train_labels.csv')


# In[5]:


print(train.shape)
print(test.shape)


# In[ ]:





# In[6]:


def to_category(df, c_list):
    for c in c_list:
        df[c] = df[c].astype('category')
    return df


# In[7]:


def to_int32(df,c_list):
    for c in c_list:
        df[c] = df[c].astype('int32')
    return df


# In[8]:


def to_int8(df,c_list):
    for c in c_list:
        df[c] = df[c].astype('int8')
    return df


# In[9]:


def to_float32(df,c_list):
    for c in c_list:
        df[c] = df[c].astype('float32')
    return df


# In[10]:


def onehot(df,c_list):
    output = pd.DataFrame(dtype=int)
    k = {}
    for idx, val in enumerate(c_list):
        k[val] = pd.get_dummies(df[val])
    output = pd.concat(list(k.values()), axis=1)
    return output   


# In[ ]:





# In[11]:


train_ids = train_labels['installation_id'].unique().tolist()


# In[12]:


train.where(train['installation_id'].isin(train_ids), inplace=True)
train.dropna(inplace=True)


# In[13]:


train.installation_id.nunique()


# In[14]:


train = to_int32(train,['event_code','event_count','game_time'])


# In[15]:


# feature engineering using event_count, game_time
df_1=train[['installation_id', 'game_session','event_count', 'game_time']].         groupby(['installation_id','game_session'], as_index=False, sort=False).        agg({'event_count': 'max', 'game_time': ['mean', 'max']})
df_1.columns= [''.join(col).strip() for col in df_1.columns.values]


# In[16]:


print(df_1.columns)
print(df_1.shape)


# In[17]:


# one-hot encoding event_code using pd.get_dummies
df_2 = train[['installation_id', 'game_session', 'event_code']]

df_2=pd.get_dummies(df_2, columns=['event_code']).             groupby(['installation_id', 'game_session'], as_index=False, sort=False).             agg(sum)


# In[18]:


df_3 = train[['installation_id','game_session', 'title', 'type','world']].             groupby(['installation_id', 'game_session'], as_index=False, sort=False).first()


# In[ ]:





# In[19]:


df_labels = train_labels.groupby(['installation_id', 'title'], as_index=False, sort=False).             agg({'num_correct': ['sum', 'mean'], 'num_incorrect': ['sum', 'mean'], 'accuracy':['max','mean'],'accuracy_group':['max','mean']})
df_labels.columns = ['installation_id','title']+['_'.join(col).strip() for col in df_labels.columns[2:].values]


# In[20]:


train_labels.head()


# In[21]:


df_labels.head()


# In[22]:


df_labels.columns


# In[23]:


target =df_labels[['installation_id', 'title','accuracy_group_max']]


# In[24]:


target.set_index(['installation_id','title'], inplace=True)


# In[25]:


target = target.unstack()


# In[26]:


target.columns= [''.join(col).strip() for col in target.columns.values]


# In[27]:


target.isna().sum()


# In[28]:


target.shape


# In[29]:


target.head(10)


# In[30]:


target_col_names = target.columns.tolist()
# target_col_names=['accuracy_group_maxBird Measurer (Assessment)',
#  'accuracy_group_maxCart Balancer (Assessment)',
#  'accuracy_group_maxCauldron Filler (Assessment)',
#  'accuracy_group_maxChest Sorter (Assessment)',
#  'accuracy_group_maxMushroom Sorter (Assessment)']
short_target_col_names = ['bm', 'cb', 'cf', 'cs', 'ms']
target.columns = short_target_col_names


# In[31]:


target


# In[32]:


#contains independent variables
final_train= df_1.join(df_2.drop(['installation_id','game_session'],axis=1)).                join(df_3.drop(['installation_id','game_session'],axis=1))


# In[33]:


# contains dependent & independent variables
final_train_merged = final_train.join(target, on='installation_id')


# In[34]:


final_train.columns


# In[35]:


# final_train_merged.dtypes


# In[36]:


print(final_train_merged.shape)


# In[37]:


final_train_merged = final_train_merged[final_train_merged['event_countmax']>=25]


# In[ ]:





# In[38]:


# final_train_merged.dtypes


# In[39]:


final_train_merged = to_category(final_train_merged,['title', 'type','world'])
final_train_merged = to_int8(final_train_merged, ['event_countmax',
        'event_code_2000', 'event_code_2010', 'event_code_2020',
       'event_code_2025', 'event_code_2030', 'event_code_2035',
       'event_code_2040', 'event_code_2050', 'event_code_2060',
       'event_code_2070', 'event_code_2075', 'event_code_2080',
       'event_code_2081', 'event_code_2083', 'event_code_3010',
       'event_code_3020', 'event_code_3021', 'event_code_3110',
       'event_code_3120', 'event_code_3121', 'event_code_4010',
       'event_code_4020', 'event_code_4021', 'event_code_4022',
       'event_code_4025', 'event_code_4030', 'event_code_4031',
       'event_code_4035', 'event_code_4040', 'event_code_4045',
       'event_code_4050', 'event_code_4070', 'event_code_4080',
       'event_code_4090', 'event_code_4095', 'event_code_4100',
       'event_code_4110', 'event_code_4220', 'event_code_4230',
       'event_code_4235', 'event_code_5000', 'event_code_5010'
        ])
final_train_merged =to_int32(final_train_merged,['game_timemean', 'game_timemax'])


# In[40]:


target.columns.tolist()


# In[41]:


# contains results of installation_id who took the relevant assessment
filtered_target ={}
for x in ['bm', 'cb', 'cf', 'cs', 'ms']:
    filtered_target[x] = target[[x]].dropna().reset_index()
    print(filtered_target[x].shape)


# In[42]:


filtered_target['ms']['installation_id'].nunique()


# In[43]:


# contains train data of installation_id who took the relevant assessment
filtered_train  = {}
for x in ['bm', 'cb', 'cf', 'cs', 'ms']:
    filtered_train[x] = final_train.where(final_train['installation_id'].                            isin(filtered_target[x]['installation_id'].unique().tolist())                           ).dropna()
    print(filtered_train[x].shape)


# In[44]:


filtered_train['cf']


# In[45]:


# changing dtypes


# In[46]:


for x in filtered_train.keys():
    filtered_train[x] = to_category(filtered_train[x],['title', 'type','world'])
    filtered_train[x] = to_int8(filtered_train[x], [ 'event_code_2000',
            'event_code_2010', 'event_code_2020',
           'event_code_2025', 'event_code_2030', 'event_code_2035',
           'event_code_2040', 'event_code_2050', 'event_code_2060',
           'event_code_2070', 'event_code_2075', 'event_code_2080',
           'event_code_2081', 'event_code_2083', 'event_code_3010',
           'event_code_3020', 'event_code_3021', 'event_code_3110',
           'event_code_3120', 'event_code_3121', 'event_code_4010',
           'event_code_4020', 'event_code_4021', 'event_code_4022',
           'event_code_4025', 'event_code_4030', 'event_code_4031',
           'event_code_4035', 'event_code_4040', 'event_code_4045',
           'event_code_4050', 'event_code_4070', 'event_code_4080',
           'event_code_4090', 'event_code_4095', 'event_code_4100',
           'event_code_4110', 'event_code_4220', 'event_code_4230',
           'event_code_4235', 'event_code_5000', 'event_code_5010'
            ])
    filtered_train[x] =to_int32(filtered_train[x],['game_timemean', 'game_timemax','event_countmax'])


# In[47]:


filtered_train['bm']


# In[48]:


# contains dependent variable & target variable of installation_id who took the relevant assessment
filtered_train_merged={}
for x in ['bm', 'cb', 'cf', 'cs', 'ms']:
    filtered_train_merged[x] = filtered_train[x].join(filtered_target[x].                                                      set_index('installation_id'),
                                                      on='installation_id')
    print(filtered_train_merged[x].shape)


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[49]:


X_train={}
X_valid={}
y_train={}
y_valid={}
for x in ['bm', 'cb', 'cf', 'cs', 'ms']:
    X_train[x], X_valid[x], y_train[x], y_valid[x] = train_test_split(filtered_train_merged[x].                                                    drop(['game_session',x],axis=1),
                                                        filtered_train_merged[x][[x]],
                                                          test_size=0.2,
                                                           shuffle=True)
    
    X_train[x].set_index('installation_id', drop=True, inplace=True)
    X_valid[x].set_index('installation_id', drop=True, inplace=True)


# In[ ]:





# In[ ]:





# In[ ]:





# In[50]:


classifier_bm = (catboost.CatBoostClassifier(iterations=1000,
                                         cat_features=['title','world','type'],
                                         learning_rate=None,
                                         depth=None,
                                         l2_leaf_reg=None,
                                         metric_period=10,
                                         custom_loss=['WKappa'],
                                         model_size_reg=None))


# In[51]:


train_pool = Pool(data=X_train['bm'],
                 label=y_train['bm'],
                 cat_features=['title', 'world', 'type'])
valid_pool = Pool(data=X_valid['bm'],
                 label=y_valid['bm'],
                 cat_features=['title','world','type'])


# In[52]:


params ={
    'iterations': 100,
    'learning_rate': 0.5,    
    }

cv_data =cv(
    params = params,
    pool = train_pool,
    fold_count=5,
    shuffle=True,
    partition_random_seed=0,
    plot=True,
)


# In[ ]:





# In[ ]:





# In[53]:


classifier_bm.fit(train_pool,eval_set=valid_pool, plot=True)


# In[54]:


# X_valid


# In[ ]:





# In[55]:


y_pred = classifier_bm.predict(X_valid['bm'])


# In[56]:


print(classification_report(y_valid['bm'],y_pred))


# In[ ]:





# In[ ]:





# In[57]:


classifier_cb = (catboost.CatBoostClassifier(iterations=1000,
                                         cat_features=['title','world','type'],
                                         learning_rate=None,
                                         depth=None,
                                         l2_leaf_reg=None,
                                         metric_period=100,
                                         custom_loss=['AUC'],
                                         model_size_reg=None))


# In[58]:


train_pool = Pool(data=X_train['cb'],
                 label=y_train['cb'],
                 cat_features=['title', 'world', 'type'])
valid_pool = Pool(data=X_valid['cb'],
                 label=y_valid['cb'],
                 cat_features=['title','world','type'])


# In[59]:


params ={
    'iterations': 100,
    'learning_rate': 0.5,    
    }

cv_data =cv(
    params = params,
    pool = train_pool,
    fold_count=5,
    shuffle=True,
    partition_random_seed=0,
    plot=True,
)


# In[ ]:





# In[ ]:





# In[60]:


classifier_cb.fit(train_pool,eval_set=valid_pool, plot=True)


# In[ ]:





# In[61]:


y_pred = classifier_cb.predict(X_valid['cb'])


# In[62]:


print(classification_report(y_valid['cb'],y_pred))


# In[ ]:





# In[ ]:





# In[63]:


classifier_cf = (catboost.CatBoostClassifier(iterations=1000,
                                         cat_features=['title','world','type'],
                                         learning_rate=None,
                                         depth=None,
                                         l2_leaf_reg=None,
                                         metric_period=100,
                                         custom_loss=['AUC'],
                                         model_size_reg=None))


# In[64]:


train_pool = Pool(data=X_train['cf'],
                 label=y_train['cf'],
                 cat_features=['title', 'world', 'type'])
valid_pool = Pool(data=X_valid['cf'],
                 label=y_valid['cf'],
                 cat_features=['title','world','type'])


# In[65]:


# params ={
#     'iterations': 100,
#     'learning_rate': 0.5,    
#     }

# cv_data =cv(
#     params = params,
#     pool = train_pool,
#     fold_count=5,
#     shuffle=True,
#     partition_random_seed=0,
#     plot=True,
# )


# In[ ]:





# In[66]:


classifier_cf.fit(train_pool,eval_set=valid_pool, plot=True)


# In[ ]:





# In[67]:


y_pred = classifier_cf.predict(X_valid['cf'])


# In[68]:


print(classification_report(y_valid['cf'],y_pred))


# In[ ]:





# In[69]:


classifier_cs = (catboost.CatBoostClassifier(iterations=1000,
                                         cat_features=['title','world','type'],
                                         learning_rate=None,
                                         depth=None,
                                         l2_leaf_reg=None,
                                         metric_period=100,
                                         custom_loss=['AUC'],
                                         model_size_reg=None))


# In[70]:


train_pool = Pool(data=X_train['cs'],
                 label=y_train['cs'],
                 cat_features=['title', 'world', 'type'])
valid_pool = Pool(data=X_valid['cs'],
                 label=y_valid['cs'],
                 cat_features=['title','world','type'])


# In[71]:


# params ={
#     'iterations': 100,
#     'learning_rate': 0.5,    
#     }

# cv_data =cv(
#     params = params,
#     pool = train_pool,
#     fold_count=5,
#     shuffle=True,
#     partition_random_seed=0,
#     plot=True,
# )


# In[ ]:





# In[ ]:





# In[72]:


classifier_cs.fit(train_pool,eval_set=valid_pool, plot=True)


# In[ ]:





# In[73]:


y_pred = classifier_cs.predict(X_valid['cs'])


# In[74]:


print(classification_report(y_valid['cs'],y_pred))


# In[ ]:





# In[ ]:





# In[ ]:





# In[75]:


classifier_ms = (catboost.CatBoostClassifier(iterations=1000,
                                         cat_features=['title','world','type'],
                                         learning_rate=None,
                                         depth=None,
                                         l2_leaf_reg=None,
                                         metric_period=100,
                                         custom_loss=['WKappa'],
                                         model_size_reg=None))


# In[76]:


train_pool = Pool(data=X_train['ms'],
                 label=y_train['ms'],
                 cat_features=['title', 'world', 'type'])
valid_pool = Pool(data=X_valid['ms'],
                 label=y_valid['ms'],
                 cat_features=['title','world','type'])


# In[77]:


# params ={
#     'iterations': 100,
#     'learning_rate': 0.5,    
#     }

# cv_data =cv(
#     params = params,
#     pool = train_pool,
#     fold_count=5,
#     shuffle=True,
#     partition_random_seed=0,
#     plot=True,
# )


# In[ ]:





# In[ ]:





# In[78]:


classifier_ms.fit(train_pool,eval_set=valid_pool, plot=True)


# In[79]:


# X_valid


# In[80]:


y_pred = classifier_ms.predict(X_valid['ms'])


# In[81]:


print(classification_report(y_valid['ms'],y_pred))


# In[ ]:





# In[82]:


classifier_bm.save_model('classifier_bm',
           format="cbm",
           export_parameters=None,
           pool=None)


# In[83]:


classifier_cb.save_model('classifier_cb',
           format="cbm",
           export_parameters=None,
           pool=None)


# In[84]:


classifier_cf.save_model('classifier_cf',
           format="cbm",
           export_parameters=None,
           pool=None)


# In[85]:


classifier_cs.save_model('classifier_cs',
           format="cbm",
           export_parameters=None,
           pool=None)


# In[86]:


classifier_ms.save_model('classifier_ms',
           format="cbm",
           export_parameters=None,
           pool=None)


# In[ ]:





# In[ ]:





# In[ ]:





# In[87]:


test = to_int32(test,['event_code','event_count','game_time'])


# In[88]:


df_1=test[['installation_id', 'game_session','event_count', 'game_time']].         groupby(['installation_id','game_session'], as_index=False, sort=False).        agg({'event_count': 'max', 'game_time': ['mean', 'max']})
df_1.columns= [''.join(col).strip() for col in df_1.columns.values]


# In[89]:


df_1.columns


# In[90]:


df_2 = test[['installation_id', 'game_session', 'event_code']]


# In[91]:


df_2=pd.get_dummies(df_2, columns=['event_code']).             groupby(['installation_id', 'game_session'], as_index=False, sort=False).             agg(sum)


# In[92]:


df_3 = test[['installation_id','game_session', 'title', 'type','world']].             groupby(['installation_id', 'game_session'], as_index=False, sort=False).first()


# In[93]:


df_3.shape


# In[94]:


final_test= df_1.join(df_2.drop(['installation_id','game_session'],axis=1)).join(df_3.drop(['installation_id','game_session'],axis=1))


# In[95]:


final_test.set_index('installation_id', inplace=True)


# In[96]:


final_test.drop('game_session', axis=1, inplace=True)


# In[ ]:





# In[97]:


predictions ={}
predictions['bm']=pd.DataFrame(classifier_bm.predict(final_test))
predictions['cb']=pd.DataFrame(classifier_cb.predict(final_test))
predictions['cf']=pd.DataFrame(classifier_cf.predict(final_test))
predictions['cs']=pd.DataFrame(classifier_cs.predict(final_test))
predictions['ms']=pd.DataFrame(classifier_ms.predict(final_test))


# In[98]:


predictions_all =pd.concat([predictions[x] for x in predictions.keys()],axis=1)


# In[99]:


predictions_all.index =  final_test.index


# In[100]:


predictions_all.columns = predictions.keys()


# In[101]:


predictions_all


# In[102]:


predictions_all_g=predictions_all.groupby('installation_id',sort=False, as_index=True).agg(lambda x:x.value_counts().index[0])


# In[103]:


predictions_all_g = to_int32(predictions_all_g,predictions_all_g.columns.tolist())


# In[104]:


test_last = test.groupby('installation_id').last()


# In[105]:


assess_map  = {'Bird Measurer (Assessment)':'bm',
                 'Cart Balancer (Assessment)':'cb',
                 'Cauldron Filler (Assessment)':'cf',
                 'Chest Sorter (Assessment)': 'cs',
                 'Mushroom Sorter (Assessment)':'ms'}


# In[106]:


test_last['title'] = test_last['title'].map(assess_map) 


# In[107]:


submit = {}
count = 0
for x,y in zip(test_last.index, test_last['title']):
    submit[x] = predictions_all_g.loc[x][y]


# In[108]:


test_last['title'].shape[0] ==predictions_all_g.shape[0]


# In[109]:


submit_df = pd.DataFrame.from_dict(submit,orient='index')
submit_df.reset_index(level=0, inplace=True)


# In[110]:


submit_df.columns= ['installation_id','accuracy_group']


# In[111]:


submit_df.head(10)


# In[112]:


submit_df.to_csv('submission.csv',index=False)


# In[ ]:





# In[ ]:




