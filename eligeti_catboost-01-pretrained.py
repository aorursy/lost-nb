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


from catboost import Pool, cv, CatBoostClassifier


pd.set_option('display.max_columns', 1000)
pd.set_option('display.max_rows', 500)


# In[2]:




for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.


# In[3]:


test = pd.read_csv('../input/data-science-bowl-2019/test.csv')


# In[4]:


test.head()


# In[5]:


model_bm = CatBoostClassifier()
model_cb = CatBoostClassifier()
model_cf = CatBoostClassifier()
model_cs = CatBoostClassifier()
model_ms = CatBoostClassifier()


# In[6]:


model_bm.load_model("../input/trainedmodels/classifier_bm")
model_cb.load_model("../input/trainedmodels/classifier_cb")
model_cf.load_model("../input/trainedmodels/classifier_cf")
model_cs.load_model("../input/trainedmodels/classifier_cs")
model_ms.load_model("../input/trainedmodels/classifier_ms")


# In[7]:


def to_category(df, c_list):
    for c in c_list:
        df[c] = df[c].astype('category')
    return df

def to_int32(df,c_list):
    for c in c_list:
        df[c] = df[c].astype('int32')
    return df

def to_int8(df,c_list):
    for c in c_list:
        df[c] = df[c].astype('int8')
    return df

def to_float32(df,c_list):
    for c in c_list:
        df[c] = df[c].astype('float32')
    return df


# In[ ]:





# In[8]:


test = to_int32(test,['event_code','event_count','game_time'])

df_1=test[['installation_id', 'game_session','event_count', 'game_time']].         groupby(['installation_id','game_session'], as_index=False, sort=False).        agg({'event_count': 'max', 'game_time': ['mean', 'max']})
df_1.columns= [''.join(col).strip() for col in df_1.columns.values]


df_2 = test[['installation_id', 'game_session', 'event_code']]

df_2=pd.get_dummies(df_2, columns=['event_code']).             groupby(['installation_id', 'game_session'], as_index=False, sort=False).             agg(sum)

df_3 = test[['installation_id','game_session', 'title', 'type','world']].             groupby(['installation_id', 'game_session'], as_index=False, sort=False).first()

final_test= df_1.join(df_2.drop(['installation_id','game_session'],axis=1)).join(df_3.drop(['installation_id','game_session'],axis=1))

final_test.set_index('installation_id', inplace=True)

final_test.drop('game_session', axis=1, inplace=True)



predictions ={}
predictions['bm']=pd.DataFrame(model_bm.predict(final_test))
predictions['cb']=pd.DataFrame(model_cb.predict(final_test))
predictions['cf']=pd.DataFrame(model_cf.predict(final_test))
predictions['cs']=pd.DataFrame(model_cs.predict(final_test))
predictions['ms']=pd.DataFrame(model_ms.predict(final_test))


predictions_all =pd.concat([predictions[x] for x in predictions.keys()],axis=1)

predictions_all.index =  final_test.index

predictions_all.columns = predictions.keys()

predictions_all

predictions_all_g=predictions_all.groupby('installation_id',sort=False, as_index=True).agg(lambda x:x.value_counts().index[0])

predictions_all_g = to_int32(predictions_all_g,predictions_all_g.columns.tolist())


test_last = test.groupby('installation_id').last()

assess_map  = {'Bird Measurer (Assessment)':'bm',
                 'Cart Balancer (Assessment)':'cb',
                 'Cauldron Filler (Assessment)':'cf',
                 'Chest Sorter (Assessment)': 'cs',
                 'Mushroom Sorter (Assessment)':'ms'}


test_last['title'] = test_last['title'].map(assess_map) 

submit = {}
count = 0
for x,y in zip(test_last.index, test_last['title']):
    submit[x] = predictions_all_g.loc[x][y]

test_last['title'].shape[0] ==predictions_all_g.shape[0]

submit_df = pd.DataFrame.from_dict(submit,orient='index')
submit_df.reset_index(level=0, inplace=True)

submit_df.columns= ['installation_id','accuracy_group']


# In[9]:


submit_df.head(10)


# In[10]:


submit_df.to_csv('submission.csv',index=False)


# In[ ]:




