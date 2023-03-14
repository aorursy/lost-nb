#!/usr/bin/env python
# coding: utf-8

# In[1]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import scipy.stats as stats
import seaborn as sns

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.


# In[2]:


import IPython

def display(*dfs):
    for df in dfs:
        IPython.display.display(df)

import cufflinks


# In[3]:


## Function to reduce the DF size
def reduce_mem_usage(df, verbose=True):
    numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
    start_mem = df.memory_usage().sum() / 1024**2    
    for col in df.columns:
        col_type = df[col].dtypes
        if col_type in numerics:
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
    if verbose: print('Mem. usage decreased to {:5.2f} Mb ({:.1f}% reduction)'.format(end_mem, 100 * (start_mem - end_mem) / start_mem))
    return df


# In[4]:


get_ipython().run_line_magic('time', "df = pd.read_csv('/kaggle/input/data-science-bowl-2019/train.csv', engine='c')")
labels = pd.read_csv('/kaggle/input/data-science-bowl-2019/train_labels.csv')


# In[5]:


df_test = pd.read_csv('/kaggle/input/data-science-bowl-2019/test.csv')


# In[6]:


df = reduce_mem_usage(df)
labels = reduce_mem_usage(labels)
df_test = reduce_mem_usage(df_test)


# In[7]:


import gc
gc.collect()


# In[8]:


train_id = df.installation_id.unique()
test_id = df_test.installation_id.unique()

print('# of unique ids in train:', train_id.shape[0])
print('# of unique ids in test:', test_id.shape[0])
print('# of unique ids from test set in train set:', np.isin(test_id, train_id).sum())


# In[9]:


time = df.timestamp.copy()


# In[10]:


df['timestamp'] = pd.to_datetime(df['timestamp'])
df_test['timestamp'] = pd.to_datetime(df_test['timestamp'])


# In[11]:


df.timestamp.min(), df_test.timestamp.min()


# In[12]:


df.timestamp.max(), df_test.timestamp.max()


# In[13]:


df.timestamp.max() - df.timestamp.min(), df_test.timestamp.max() - df_test.timestamp.min()


# In[14]:


df.game_session.nunique() == df.drop_duplicates(['game_session', 'installation_id']).game_session.nunique()


# In[15]:


df_test.game_session.nunique() == df_test.drop_duplicates(['game_session', 'installation_id']).game_session.nunique()


# In[16]:


def event_finished(df):
    df_wt_BM = df[(df.event_code == 4100) & (df.title.str.find('Bird Measurer')==-1)]
    df_BM = df[(df.event_code == 4110) & (df.title.str.find('Bird Measurer')!=-1)]
    return df_wt_BM.append(df_BM)


# In[17]:


# for test
assessment = df_test[df_test.type=='Assessment']
assessment_finished = event_finished(assessment)


# In[18]:


print('# ids in TEST\t # ids in assessment\t # ids in finished assessments')
print(test_id.shape,'\t', assessment.installation_id.unique().shape, '\t\t',
      assessment_finished.installation_id.unique().shape)


# In[19]:


# for train
assessment = df[df.type=='Assessment']
assessment_finished = event_finished(assessment)


# In[20]:


print('# ids in TRAIN\t # ids in assessment\t # ids in finished assessments')
print(train_id.shape,'\t', assessment.installation_id.unique().shape, '\t\t',
      assessment_finished.installation_id.unique().shape)


# In[21]:


# we will look only on users which have info about attempts
df = df[df.installation_id.isin(labels.installation_id)]


# In[22]:


test_last = df_test.groupby('installation_id').last()
print(test_last.event_count.nunique(), test_last.event_code.nunique(), test_last.type.nunique())
test_last


# In[23]:


train_last = df.groupby('installation_id').last()
print(train_last.event_count.nunique(), train_last.event_code.nunique(), train_last.type.nunique())
train_last


# In[24]:


def check_attempt(df):
    ''' indicate each line in assessment are finished; don't care it's correct ot not '''
    
    df['attempt'] = ((df.type == 'Assessment') &
                       (((df.event_code == 4100) & (df.title != 'Bird Measurer (Assessment)')) |
                        ((df.event_code == 4110)&(df.title == 'Bird Measurer (Assessment)')) )
                   ).astype('int8')

def check_attempt_pre_assessment(df):
    ''' indicate which assessments have finished code;
    don't care it's correct ot not, don't care was 1 attempt or more
    
    !!! Need info about attempts - use check_attempt(df) before '''
    
    df['got_attempt'] =(df.groupby('game_session').attempt.transform('sum') >0).astype('int8')

def check_correct(df):
    ''' indicate that attempt was correct'''
    
    df['correct'] = 0
    df.loc[df.attempt == 1, 'correct'] = df[df.attempt == 1].event_data.str.contains('"correct":true')                                                                             .astype('int8')
    

def check_start_assessment(df):
    ''' indicate when each assessment is started'''
    df['assessment_start'] = 0
    df.loc[(df.type == 'Assessment') & (df.event_count == 1), 'assessment_start'] = 1
    assert df.assessment_start.sum(), df[df.type == 'Assessment'].game_session.nunique()
    
    
def get_last_assessment(df, test=False):
    if test:
        groups = df.groupby('installation_id')
    else:
        groups = df[df.attempt == 1].groupby('installation_id')
    
    df = df.merge(groups.title.last().rename('last_title'), on='installation_id', how='outer')
    df = df.merge(groups.game_session.last().rename('last_game_session'), on='installation_id', how='outer')
    df = df.merge(groups.timestamp.last().rename('last_timestamp'), on='installation_id', how='outer')
    return df


# In[25]:


get_ipython().run_cell_magic('time', '', 'check_start_assessment(df_test)\ncheck_attempt(df_test)\ncheck_attempt_pre_assessment(df_test)\ncheck_correct(df_test)\n\ncheck_start_assessment(df)\ncheck_attempt(df)\ncheck_attempt_pre_assessment(df)\ncheck_correct(df)')


# In[26]:


# check that at each game session can be only 1 correct attempt
assert (df.groupby('game_session').correct.sum() > 1).sum() == 0
assert (df_test.groupby('game_session').correct.sum() > 1).sum() == 0


# In[27]:


df_ini = df.copy()


# In[28]:


# df = df_ini.copy()


# In[29]:


print(df.shape)
df = get_last_assessment(df)
df_test = get_last_assessment(df_test, test=True)


# In[30]:


df['to_cut'] = 0
df.loc[(df.timestamp >= df.last_timestamp) & (df.attempt != 1), 'to_cut'] = 1
df = df[df.to_cut == 0]
assert df.to_cut.sum() == 0
df = df.drop('to_cut', axis=1)

df.shape


# In[31]:


user_id = '01242218'

user = df_test[df_test.installation_id == user_id].copy()
user[user.title == user.last_title]


# In[32]:


test_group = df_test.groupby('installation_id')
train_group = df.groupby('installation_id')


# In[33]:


import matplotlib.pyplot as plt
import plotly.graph_objects as go
from plotly.subplots import make_subplots


# In[34]:


fig = make_subplots(rows=1, cols=2)
fig.add_trace(go.Histogram(x=test_group.attempt.sum(), name='test'), row=1,col=1)
fig.add_trace(go.Histogram(x=train_group.attempt.sum(), name='train'), row=1, col=2)
fig.update_xaxes(title_text='number of attempt',row=1, col=1)
fig.update_xaxes(title_text='number of attempt',row=1,col=2)
fig.update_layout(title='Histogram of number attempt per user',
                  font=dict(family="Courier New, monospace",
                           size=18,color="#7f7f7f"))


# In[35]:


'Mode:', test_group.attempt.sum().mode()[0],  train_group.attempt.sum().mode()[0]


# In[36]:


temp = test_group.attempt.sum().astype('int')
temp = (temp.value_counts().sort_index()/temp.shape[0]*100).head(15)
print('will be with nan (%)')
temp.cumsum().shift()


# In[37]:


fig = make_subplots(rows=1, cols=2)
fig.add_trace(go.Histogram(x=test_group.correct.sum(), name='test'), row=1,col=1)
fig.add_trace(go.Histogram(x=train_group.correct.sum(), name='train'), row=1, col=2)

fig.update_xaxes(title_text='number of correct attempt',row=1, col=1)
fig.update_xaxes(title_text='number of correct attempt',row=1,col=2)
fig.update_layout(title='Histogram of correct attempt number per user',
                  font=dict(family="Courier New, monospace",
                           size=18,color="#7f7f7f"))


# In[38]:


'Mode:', test_group.correct.sum().mode()[0],  train_group.correct.sum().mode()[0]


# In[39]:


def count_last_assessment_repeat(df):
    '''df - DataFrmae with info about lase assessment: last_title and last_game_session'''
    
    temp = df[df.type == 'Assessment']
    grouped = temp.groupby(['installation_id', 'game_session',  'title',
                           'last_title', 'last_game_session'], as_index=False)\
                                        [['assessment_start', 'got_attempt','correct']].sum()
    grouped.loc[grouped.got_attempt > 0, 'got_attempt'] = 1

    repeated = grouped[(grouped.title == grouped.last_title) & (grouped.game_session != grouped.last_game_session)]
    results = repeated.pivot_table(index=['installation_id', 'last_game_session'], columns='title', 
                                values=['assessment_start', 'got_attempt','correct'], 
                                aggfunc=np.sum, fill_value=0)

    totals = grouped.groupby('title')[['assessment_start', 'got_attempt', 'correct']].sum()
    
    assert (totals.sum().values ==            np.array([df.assessment_start.sum(), 
                     df.groupby('game_session').got_attempt.last().sum(), 
                     df.correct.sum()])).all()
    return results, totals


# In[40]:


total_repeat_test, total_assessment_test = count_last_assessment_repeat(df_test) 
total_repeat_train, total_assessment_train = count_last_assessment_repeat(df) 

lst_assessment_repeat_test = (total_repeat_test != 0).sum().unstack(level=0)[['assessment_start', 'got_attempt', 'correct']]
lst_assessment_repeat_train = (total_repeat_train != 0).sum().unstack(level=0)[['assessment_start', 'got_attempt', 'correct']]


# In[41]:


display(lst_assessment_repeat_test, total_assessment_test)


# In[42]:


def set_annotation(ax):
    for p in ax.patches:
        ax.annotate(format(p.get_height(), '.2f'), (p.get_x() + p.get_width()/2, p.get_height()),
               ha='center', va='center', xytext=(0,5), textcoords='offset points')
    return ax


# In[43]:


fig = plt.figure(figsize=(15,5))
ax = plt.subplot(121)
sns.barplot(y=lst_assessment_repeat_test.sum() / total_assessment_test.sum() * 100,
            x=lst_assessment_repeat_test.columns, ax=ax);
ax.set(ylabel='%')
ax.set_title('Test\nRelation number of repeated assessments to its initial number, %')
ax = set_annotation(ax)    

ax = plt.subplot(122)
sns.barplot(y=lst_assessment_repeat_train.sum() / total_assessment_train.sum() * 100,
            x=lst_assessment_repeat_train.columns, ax=ax);
ax.set(ylabel='%')
ax.set_title('Train\nRelation number of repeated assessments to its initial number, %')
ax = set_annotation(ax)    


# In[44]:


import scipy 
def proportions_diff_confint_ind(*data, sample=None, proportion=None, alpha = 0.05):
    '''data - list of data; 
            if sample=True, data = [sample1, sample],
            if proportion=True, data[p1, p2, n1, n2] '''
    
    if proportion: p1, p2, n1, n2 = data
    if sample: 
        sample1, sample2 = data
        n1 = len(sample1)
        n2 = len(sample2)
        p1 = float(sum(sample1)) / n1
        p2 = float(sum(sample2)) / n2
    if sample is None and proportion is None:
        print('Choose sample or proportion')
        return None
    
    z = scipy.stats.norm.ppf(1 - alpha / 2.)
    left_boundary = (p1 - p2) - z * np.sqrt(p1 * (1 - p1)/n1 + p2 * (1 - p2)/ n2)
    right_boundary = (p1 - p2) + z * np.sqrt(p1 * (1 - p1)/ n1 + p2 * (1 - p2)/ n1)
    
    return (left_boundary, right_boundary)


def proportions_diff_z_stat_ind(*data, sample=None, proportion=None):
    '''data - list of data; 
            if sample=True, data = [sample1, sample],
            if proportion=True, data[p1, p2, n1, n2]  '''
    
    if proportion: p1, p2, n1, n2 = data
    if sample: 
        sample1, sample2 = data
        n1 = len(sample1)
        n2 = len(sample2)
        p1 = float(sum(sample1)) / n1
        p2 = float(sum(sample2)) / n2
    if sample is None and proportion is None:
        print('Choose sample or proportion')
        return None
    
    P = float(p1*n1 + p2*n2) / (n1 + n2)
    return (p1 - p2) / np.sqrt(P * (1 - P) * (1. / n1 + 1. / n2))


def proportions_diff_z_test(z_stat, alternative = 'two-sided'):
    if alternative not in ('two-sided', 'less', 'greater'):
        raise ValueError("alternative not recognized\n"
                         "should be 'two-sided', 'less' or 'greater'")
    
    if alternative == 'two-sided':
        return 2 * (1 - scipy.stats.norm.cdf(np.abs(z_stat)))
    
    if alternative == 'less':
        return scipy.stats.norm.cdf(z_stat)

    if alternative == 'greater':
        return 1 - scipy.stats.norm.cdf(z_stat)


# In[45]:


for test,total_test, train,total_train, idx in zip(lst_assessment_repeat_test.sum(), 
                                                   total_assessment_test.sum(),
                                                   lst_assessment_repeat_train.sum(), 
                                                   total_assessment_train.sum(),
                                                   total_assessment_test.columns):
    print(idx)
    print('ci p_test-p_train:', 
          proportions_diff_confint_ind(test/total_test, train/total_train, total_test, total_train,
                                       proportion=True))
    statistics = proportions_diff_z_stat_ind(test/total_test, train/total_train, 
                                             total_test, total_train,
                                             proportion=True)
    print('p_value:', proportions_diff_z_test(statistics), '\n')
    


# In[46]:


fig = plt.figure(figsize=(10,12))

temp = (lst_assessment_repeat_test/total_assessment_test *100).unstack().reset_index()                                        .rename(columns={'level_0': 'action',
                                                        0: 'count'})
ax = plt.subplot(211)
sns.barplot(data=temp, x='action', y='count', hue='title', ax=ax);
ax.set(ylabel='%')
ax.set_title('Test\nRelation number of repeated assessments to its initial number, %')
ax = set_annotation(ax)

temp = (lst_assessment_repeat_train/total_assessment_train *100).unstack().reset_index()                                        .rename(columns={'level_0': 'action',
                                                        0: 'count'})
ax = plt.subplot(212)
sns.barplot(data=temp, x='action', y='count', hue='title', ax=ax);
ax.set(ylabel='%')
ax.set_title('Train\nRelation number of repeated assessments to its initial number, %')
ax = set_annotation(ax)


# In[47]:


total_repeat_train = total_repeat_train.reset_index()                                       .rename(columns={'last_game_session': 'game_session'})

assert total_repeat_train.game_session.nunique() == total_repeat_train.shape[0]
assert labels.game_session.nunique() == labels.shape[0]

# add accuracy_group to repeating info
print(total_repeat_train.shape)
total_repeat_train = total_repeat_train.merge(labels[['game_session', 'accuracy_group']], on='game_session', how='left')
total_repeat_train = total_repeat_train.rename(columns={'accuracy_group': ('accuracy_group', '')})                                       .drop('game_session',axis=1)
total_repeat_train.columns = pd.MultiIndex.from_tuples(total_repeat_train.columns)
assert total_repeat_train.accuracy_group.isna().sum() == 0
total_repeat_train.shape


# In[48]:


# sum along title assessments
temp = total_repeat_train.sum(axis=1, level=0)

#plot
_, axes = plt.subplots(5, 3, gridspec_kw={'height_ratios': [4, 1,1,1,1]}, figsize=(15,10))
axes = axes.flatten()
for i, action in enumerate(['assessment_start', 'got_attempt', 'correct']):
#     ax = plt.subplot(2,3,i+1)
    sns.barplot(x='accuracy_group', y=action, data=temp, ax=axes[i])
    
    colors=sns.color_palette()
    for acc in range(0,4):
        sns.distplot(temp.loc[temp.accuracy_group==acc, action], kde=False, 
                     color=colors[acc],
                     label='acc_group'+str(acc), ax=axes[i+3+acc*3])
        axes[i+3+acc*3].legend()
    


# In[49]:


acc0 = temp.loc[temp.accuracy_group == 0]
acc1 = temp.loc[temp.accuracy_group == 1]#, 'attempt_start']
acc2 = temp.loc[temp.accuracy_group == 2]#, 'attempt_start']
acc3 = temp.loc[temp.accuracy_group == 3]#, 'attempt_start']

for action in ['assessment_start', 'got_attempt', 'correct']:
    p = scipy.stats.f_oneway(acc0[action], acc1[action],
                               acc2[action],acc3[action])[1]
    reject = '' if p > 0.05 else '!!reject'
    print(f'{action}  p-value: {p} {reject}')


# In[50]:


from statsmodels.stats.multicomp import pairwise_tukeyhsd
pairwise_tukeyhsd(temp.correct,temp.accuracy_group).plot_simultaneous();#.summary()


# In[51]:


titles = df_test[df_test.type=='Assessment'].groupby('world').title.unique()
print('Test')
for i in titles.index:
    print(i, temp[i])
    
titles = df[df.type=='Assessment'].groupby('world').title.unique()
print('Train')
for i in titles.index:
    print(i, temp[i])

