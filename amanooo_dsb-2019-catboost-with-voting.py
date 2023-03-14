#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
pd.set_option('display.max_columns', None)
from scipy import stats

import os, sys, datetime
from time import time
from tqdm import tqdm_notebook as tqdm

from collections import Counter

from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import cohen_kappa_score

from catboost import CatBoostClassifier
import category_encoders as ce


# In[2]:


Kaggle = True

if Kaggle:
    DIR = '../input/data-science-bowl-2019'
    task_type = 'CPU'
else:
    DIR = './data-science-bowl-2019'
    task_type = 'GPU'


# In[3]:


train = pd.read_csv(os.path.join(DIR,'train.csv'))
train_labels = pd.read_csv(os.path.join(DIR,'train_labels.csv'))
specs = pd.read_csv(os.path.join(DIR,'specs.csv'))
test = pd.read_csv(os.path.join(DIR,'test.csv'))


# In[4]:


print('train:\t\t',train.shape)
print('train_labels:\t',train_labels.shape)
print('specs:\t\t',specs.shape)
print('test:\t\t',test.shape)


# In[5]:


train.head()


# In[6]:


train[['event_id','game_session','installation_id',
       'title','type','world']].describe()


# In[7]:


event_code_n = train['event_code'].nunique()
print("num of unique 'event_code':", event_code_n)
print("'event_code': ",
      train['event_code'].min(), "-", train['event_code'].max())


# In[8]:


# 'event_data' exsample
print(train['event_data'][40])
print(train['event_data'][41])
print(train['event_data'][43])


# In[9]:


train_labels.head()


# In[10]:


train_labels[['game_session','installation_id', 'title']].describe()


# In[11]:


# unique 'title' list
train_labels['title'].unique()


# In[12]:


specs.head()


# In[13]:


specs.describe()


# In[14]:


# 'info' exsample
print(specs['info'][0])
print(specs['info'][6])
print(specs['info'][7])


# In[15]:


# 'args' exsample
print(specs['args'][0])
print(specs['args'][1])


# In[16]:


test.head(8)


# In[17]:


test[['event_id','game_session','installation_id',
       'title','type','world']].describe()


# In[18]:


# make 'title' and 'event_code' list
title_list = list(set(train['title'].value_counts().index)                    .union(set(test['title'].value_counts().index)))
event_code_list = list(set(train['event_code'].value_counts().index)                    .union(set(test['event_code'].value_counts().index)))


# In[19]:


# makes dict 'title to number(integer)'
title2num = dict(zip(title_list, np.arange(len(title_list))))
# makes dict 'number to title'
num2title = dict(zip(np.arange(len(title_list)), title_list))
# makes dict 'title to win event_code' 
# (4100 except 'Bird Measurer' and 4110 for 'Bird Measurer'))
title2win_code = dict(zip(title2num.values()                     ,(np.ones(len(title2num))).astype('int') * 4100))
title2win_code[title2num['Bird Measurer (Assessment)']] = 4110


# In[20]:


# Convert 'title' to the number
train['title'] = train['title'].map(title2num)
test['title'] = test['title'].map(title2num)
train_labels['title'] = train_labels['title'].map(title2num)

# Convert 'timestamp' to datetime
train['timestamp'] = pd.to_datetime(train['timestamp'])
test['timestamp'] = pd.to_datetime(test['timestamp'])


# In[21]:


# Convert the raw data into processed features
def get_data(user_sample, test_set=False):
    '''
    user_sample : DataFrame from train/test group by 'installation_id'
    test_set    : related with the labels processing
    '''
    # Constants and parameters declaration
    user_assessments = []
    last_type = 0
    types_count = {'Clip':0, 'Activity':0, 'Assessment':0, 'Game':0}
    time_first_activity = float(user_sample['timestamp'].values[0])
    time_spent_each_title = {title:0 for title in title_list}
    event_code_count = {code:0 for code in event_code_list}
    accuracy_groups = {0:0, 1:0, 2:0, 3:0}
    
    accumu_accuracy_group = 0
    accumu_accuracy=0
    accumu_win_n = 0 
    accumu_loss_n = 0 
    accumu_actions = 0
    counter = 0
    durations = []
    
    # group by 'game_session'
    for i, session in user_sample.groupby('game_session', sort=False):
        # i      : game_session_id
        # session: DataFrame from user_sample group by 'game_session'
        session_type = session['type'].iloc[0]  # Game/Assessment/Activity/Clip
        session_title = session['title'].iloc[0]
        
        if session_type != 'Assessment':
            time_spent = int(session['game_time'].iloc[-1] / 1000)   # [sec]
            time_spent_each_title[num2title[session_title]] += time_spent
        
        if (session_type == 'Assessment') & (test_set or len(session)>1):
            # search for event_code 4100(4110)
            all_4100 = session.query(f'event_code ==                                          {title2win_code[session_title]}')
            # numbers of wins and losses
            win_n = all_4100['event_data'].str.contains('true').sum()
            loss_n = all_4100['event_data'].str.contains('false').sum()

            # init features and then update
            features = types_count.copy()
            features.update(time_spent_each_title.copy())
            features.update(event_code_count.copy())
            features['session_title'] = session_title
            features['accumu_win_n'] = accumu_win_n
            features['accumu_loss_n'] = accumu_loss_n
            accumu_win_n += win_n
            accumu_loss_n += loss_n
            
            features['day_of_the_week'] = (session['timestamp'].iloc[-1]).                                             strftime('%A')    # Mod 2019-11-17

            if durations == []:
                features['duration_mean'] = 0
            else:
                features['duration_mean'] = np.mean(durations)
            durations.append((session.iloc[-1, 2] - session.iloc[0, 2] ).seconds)

            # average of the all accuracy of this player
            features['accuracy_ave'] = accumu_accuracy / counter                                                 if counter > 0 else 0
            accuracy = win_n / (win_n + loss_n)                                    if (win_n + loss_n) > 0 else 0
            accumu_accuracy += accuracy
            if accuracy == 0:
                features['accuracy_group'] = 0
            elif accuracy == 1:
                features['accuracy_group'] = 3
            elif accuracy == 0.5:
                features['accuracy_group'] = 2
            else:
                features['accuracy_group'] = 1
            features.update(accuracy_groups)
            accuracy_groups[features['accuracy_group']] += 1
            # average of accuracy_groups of this player
            features['accuracy_group_ave'] =                     accumu_accuracy_group / counter if counter > 0 else 0
            accumu_accuracy_group += features['accuracy_group']
            
            # how many actions the player has done in this game_session
            features['accumu_actions'] = accumu_actions
            
            # if test_set, all sessions belong to the final dataset
            # elif train, needs to be passed throught this clausule
            if test_set or (win_n + loss_n) > 0:
                user_assessments.append(features)
                
            counter += 1
        
        # how many actions was made in each event_code
        event_codes = Counter(session['event_code'])
        for key in event_codes.keys():
            event_code_count[key] += event_codes[key]

        # how many actions the player has done
        accumu_actions += len(session)
        if last_type != session_type:
            types_count[session_type] += 1
            last_type = session_type
            
    # if test_set, only the last assessment must be predicted,
    # the previous are scraped
    if test_set:
        return user_assessments[-1]
    return user_assessments


# In[22]:


# get_data function is applyed to each installation_id
compiled_data = []
installation_n = train['installation_id'].nunique()
for i, (ins_id, user_sample) in tqdm(enumerate(train.groupby(                                      'installation_id', sort=False)),
                                     total=installation_n):
    # user_sample : DataFrame group by 'installation_id'
    compiled_data += get_data(user_sample)


# In[23]:


# the compiled_data is converted to DataFrame and deleted to save memmory
new_train = pd.DataFrame(compiled_data)
del compiled_data


# In[24]:


new_train.head(10)


# In[25]:


# process test set, the same that was done with the train set
new_test = []
for ins_id, user_sample in tqdm(test.groupby('installation_id',sort=False),
                                total=1000):
    new_test.append(get_data(user_sample, test_set=True))
    
new_test = pd.DataFrame(new_test)


# In[26]:


new_test.head(10)


# In[27]:


# all_features but 'accuracy_group', that is the label y
all_features = [x for x in new_train.columns if x not in ['accuracy_group']]
# categorical feature
categorical_features = ['session_title','day_of_the_week']


# In[28]:


# Encode categorical_features to integer(for use with LightGB,XGBoost,etc)

# concatnate train and test data
temp_df = pd.concat([new_train[all_features], new_test[all_features]])
# encode
encoder = ce.ordinal.OrdinalEncoder(cols = categorical_features)
temp_df = encoder.fit_transform(temp_df)
# dataset
X, y = temp_df.iloc[:len(new_train),:], new_train['accuracy_group']
X_test = temp_df.iloc[len(new_train):,:]


# In[29]:


X.head()


# In[30]:


y.head()


# In[31]:


X_test.head()


# In[32]:


# makes the model and set the parameters
def make_classifier():
    model = CatBoostClassifier(
        loss_function='MultiClass',
        eval_metric="WKappa",
        task_type=task_type,
        thread_count=-1,
        od_type="Iter",
        early_stopping_rounds=500,
        random_seed=42,
        
        border_count=110,
        l2_leaf_reg=7,
        iterations=1800,
        learning_rate=0.2,
        depth=5
    )
    return model


# In[33]:


# Train and make 5 models
start_time = time()

NFOLDS = 5
folds = StratifiedKFold(n_splits=NFOLDS, shuffle=True, random_state=42)
models = []
scores = []
for fold, (train_ids, test_ids) in enumerate(folds.split(X, y)):
    print('● Fold :', fold+1)
    model = make_classifier()
    model.fit(X.loc[train_ids, all_features], y.loc[train_ids], 
              eval_set=(X.loc[test_ids, all_features], y.loc[test_ids]),
              use_best_model=False,     # The meaning of this parameter does not fall into the trap
              verbose=500,
              cat_features=categorical_features)    
    models.append(model)
    scores.append(model.get_best_score()['validation']['WKappa'])
    print('\n')
    
print('-' * 50)
print("Average 'WKappa' Score =", np.mean(scores))
print('-' * 50)
print('finished in {}'.format( 
    str(datetime.timedelta(seconds=time() - start_time))))


# In[34]:


# Check the effect of 'voting'
predictions = []
for model in models:
    predictions.append(model.predict(X).astype(int))
predictions = np.concatenate(predictions, axis=1)
df = pd.DataFrame(predictions)

vote = stats.mode(predictions, axis=1)[0].reshape(-1)
df['vote'] = vote
df['y'] = y
df.head(10)


# In[35]:


kappa_score = []
for col in df.columns[:NFOLDS+1]:
    kappa_score.append(cohen_kappa_score(df['y'], df[col]))
print('kappa_score:\n',kappa_score)
print('average score:',np.mean(kappa_score[:NFOLDS]))
print('voting score :',kappa_score[-1],'\n')
print('Improved from',np.mean(kappa_score[:NFOLDS]),'to',
      kappa_score[-1],"by 'voting'")


# In[36]:


predictions = []
for model in models:
    predictions.append(model.predict(X_test))
predictions = np.concatenate(predictions, axis=1)
# Voting
predictions = stats.mode(predictions, axis=1)[0].reshape(-1)
print(predictions.shape)


# In[37]:


submission = pd.read_csv(os.path.join(DIR,'sample_submission.csv'))
submission['accuracy_group'] = np.round(predictions).astype('int')
submission.head(10)


# In[38]:


submission['accuracy_group'].plot(kind='hist')


# In[39]:


submission.to_csv('submission.csv', index=None)

