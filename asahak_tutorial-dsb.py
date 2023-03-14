#!/usr/bin/env python
# coding: utf-8

# In[1]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.


# In[ ]:





# In[2]:


import pandas as pd
sample_submission = pd.read_csv("../input/data-science-bowl-2019/sample_submission.csv")
specs = pd.read_csv("../input/data-science-bowl-2019/specs.csv")
test = pd.read_csv("../input/data-science-bowl-2019/test.csv")
train = pd.read_csv("../input/data-science-bowl-2019/train.csv")
train_labels = pd.read_csv("../input/data-science-bowl-2019/train_labels.csv")


# In[3]:



import pandas as pd
import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import cohen_kappa_score
from scipy.stats import mode
from sklearn.feature_selection import SelectFromModel
from sklearn.model_selection import train_test_split


import xgboost as xgb
from xgboost import XGBClassifier
from xgboost import plot_importance
from matplotlib import pyplot
import shap


# In[4]:


os.listdir('../input/data-science-bowl-2019')


# In[5]:


get_ipython().run_cell_magic('time', '', "keep_cols = ['event_id', 'game_session', 'installation_id', 'event_count',\n             'event_code','title' ,'game_time', 'type', 'world','timestamp']\ntrain=pd.read_csv('../input/data-science-bowl-2019/train.csv',usecols=keep_cols)\ntrain_labels=pd.read_csv('../input/data-science-bowl-2019/train_labels.csv',\n                         usecols=['installation_id','game_session','accuracy_group'])\ntest=pd.read_csv('../input/data-science-bowl-2019/test.csv',usecols=keep_cols)\nsubmission=pd.read_csv('../input/data-science-bowl-2019/sample_submission.csv')")


# In[6]:


get_ipython().run_cell_magic('time', '', 'train.shape,train_labels.shape')


# In[7]:


get_ipython().run_cell_magic('time', '', "x=train_labels['accuracy_group'].value_counts()\nsns.barplot(x.index,x)")


# In[8]:


train_labels.head()


# In[9]:


train.head()


# In[10]:


not_req=(set(train.installation_id.unique()) - set(train_labels.installation_id.unique()))
#labelにないデータidの集合を作成


# In[11]:


train_new=~train['installation_id'].isin(not_req)
train.where(train_new,inplace=True)#
train.dropna(inplace=True)
train['event_code']=train.event_code.astype(int)


# In[12]:


train.shape


# In[13]:


#時間の特徴量を丁寧に記述する

def extract_time_features(df):
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df['month'] = df['timestamp'].dt.month
    df['hour'] = df['timestamp'].dt.hour
    df['year'] = df['timestamp'].dt.year
    df['dayofweek'] = df['timestamp'].dt.dayofweek
    df['weekofyear'] = df['timestamp'].dt.weekofyear
    return df


# In[14]:


time_features=['month','hour','year','dayofweek','weekofyear']
def prepare_data(df):
    df=extract_time_features(df)
    
    df=df.drop('timestamp',axis=1)
    #df['timestamp']=pd.to_datetime(df['timestamp'])
    #df['hour_of_day']=df['timestamp'].map(lambda x : int(x.hour))
    

    join_one=pd.get_dummies(df[['event_code','installation_id','game_session']],
                            columns=['event_code']).groupby(['installation_id','game_session'],
                                                            as_index=False,sort=False).agg(sum)

    agg={'event_count':sum,'game_time':['sum','mean'],'event_id':'count'}

    join_two=df.drop(time_features,axis=1).groupby(['installation_id','game_session']
                                                   ,as_index=False,sort=False).agg(agg)
    
    join_two.columns= [' '.join(col).strip() for col in join_two.columns.values]
    

    join_three=df[['installation_id','game_session','type','world','title']].groupby(
                ['installation_id','game_session'],as_index=False,sort=False).first()
    
    join_four=df[time_features+['installation_id','game_session']].groupby(['installation_id',
                'game_session'],as_index=False,sort=False).agg(mode)[time_features].applymap(lambda x: x.mode[0])
    
    join_one=join_one.join(join_four)
    
    join_five=(join_one.join(join_two.drop(['installation_id','game_session'],axis=1))).                         join(join_three.drop(['installation_id','game_session'],axis=1))
    
    return join_five


# In[15]:


join_train=prepare_data(train)
cols=join_train.columns.to_list()[2:-3]
join_train[cols]=join_train[cols].astype('int16')


# In[16]:


join_test=prepare_data(test)
cols=join_test.columns.to_list()[2:-3]
join_test[cols]=join_test[cols].astype('int16')


# In[17]:


cols=join_test.columns[2:-12].to_list()
cols.append('event_id count')
cols.append('installation_id')


# In[18]:


final_train=pd.merge(train_labels,join_train,on=['installation_id','game_session'],
                                         how='left').drop(['game_session'],axis=1)

final_test=join_test.groupby('installation_id',as_index=False,sort=False).last().drop(['game_session','installation_id'],axis=1)
#final_test=(df.join(df_two)).join(df_three.join(df_four)).drop('installation_id',axis=1)


# In[19]:


final_train.shape,final_test.shape


# In[20]:


set(final_train.columns[i] for i in range(len(final_train.columns)))-set(final_test.columns[i] for i in range(len(final_test.columns)))


# In[21]:


final_train.drop(['installation_id'],axis=1)


# In[22]:


final_train.shape,final_test.shape


# In[23]:


final=pd.concat([final_train,final_test])
encoding=['type','world','title']
for col in encoding:
    lb=LabelEncoder()
    lb.fit(final[col])
    final[col]=lb.transform(final[col])
    
final_train=final[:len(final_train)]
final_test=final[len(final_train):]


# In[24]:


X_train=final_train.drop('accuracy_group',axis=1)
y_train=final_train['accuracy_group']


# In[25]:


def model(X_train,y_train,final_test,n_splits=3):
    scores=[]
    pars = {
        'colsample_bytree': 0.8,                 
        'learning_rate': 0.08,
        'max_depth': 10,
        'subsample': 1,
        'objective':'multi:softprob',
        'num_class':4,
        'eval_metric':'mlogloss',
        'min_child_weight':3,
        'gamma':0.25,
        'n_estimators':500
    }

    kf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    y_pre=np.zeros((len(final_test),4),dtype=float)
    final_test=xgb.DMatrix(final_test.drop('accuracy_group',axis=1))


    for train_index, val_index in kf.split(X_train, y_train):
        train_X = X_train.iloc[train_index]
        val_X = X_train.iloc[val_index]
        train_y = y_train[train_index]
        val_y = y_train[val_index]
        xgb_train = xgb.DMatrix(train_X, train_y)
        xgb_eval = xgb.DMatrix(val_X, val_y)

        xgb_model = xgb.train(pars,
                      xgb_train,
                      num_boost_round=1000,
                      evals=[(xgb_train, 'train'), (xgb_eval, 'val')],
                      verbose_eval=False,
                      early_stopping_rounds=20
                     )

        val_X=xgb.DMatrix(val_X)
        pred_val=[np.argmax(x) for x in xgb_model.predict(val_X)]
        score=cohen_kappa_score(pred_val,val_y,weights='quadratic')
        scores.append(score)
        print('choen_kappa_score :',score)

        pred=xgb_model.predict(final_test)
        y_pre+=pred

    pred = np.asarray([np.argmax(line) for line in y_pre])
    print('Mean score:',np.mean(scores))
    
    return xgb_model,pred


# In[26]:


X_train=X_train.drop('installation_id',axis=1)


# In[27]:


X_train.columns


# In[28]:


final_test.columns


# In[29]:


final_test=final_test.drop('installation_id',axis=1)


# In[30]:


xgb_model,pred=model(X_train,y_train,final_test,5)


# In[31]:


sub=pd.DataFrame({'installation_id':submission.installation_id,'accuracy_group':pred})
sub.to_csv('submission.csv',index=False)


# In[ ]:




